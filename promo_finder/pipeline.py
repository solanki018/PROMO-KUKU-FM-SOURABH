"""Stages 1-5. Stage 0 = parser.py, Stage 3 (validation) = segments.py.

1  per-episode LLM call   -> key_facts, clips, moments  (ids only, never timestamps)
2  one LLM call           -> cross-event / cross-episode combos
3  code                   -> real timestamps, 20-60s rule, one retry, then drop
4  3 fresh LLM calls each -> scorer, blind recap, recap-vs-facts
5  code                   -> flags, rank, dedupe, per-episode cap
"""
from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from .config import Settings
from .debug import Check, health_checks, render_checks
from .llm import FatalLLMError, LLMClient, classify_error
from .parser import episode_view, event_line
from .prompts import COMBINE, COMPARE, EXTRACT, RECAP, RETRY, SCORE, render
from .schemas import (Candidate, ComboOut, Episode, EpisodeAnalysis, EpisodeOut, Evaluation, FactCheck,
                      MomentSeg, Promo, Recap, Score, Show)
from .segments import ResolveError, candidate_text, check_combo, resolve, rule_ending
from .timeutil import fmt_ts

Progress = Callable[[str, int, int], None]


def _clamp(v, lo=1, hi=5, default=3) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ Stage 1
def _resolve_out(ep: Episode, out: EpisodeOut, cfg: Settings, an: EpisodeAnalysis, errors: list[str]) -> None:
    seen = {(c.segments[0].start_id, c.segments[0].end_id) for c in an.clips}
    for c in out.clips:
        try:
            seg = resolve(ep, c.start_id, c.end_id, cfg.min_s, cfg.max_s, cfg)
        except ResolveError as e:
            errors.append(f"clip {c.start_id}..{c.end_id}: {e}")
            continue
        if (seg.start_id, seg.end_id) in seen:
            continue
        seen.add((seg.start_id, seg.end_id))
        an.repaired += int(seg.repaired)
        an.clips.append(Candidate(cid=f"{ep.id}_c{len(an.clips) + 1}", source="llm_clip", segments=[seg],
                                  why=c.why, strength=_clamp(c.strength)))
    seen_m = {(m.segment.start_id, m.segment.end_id) for m in an.moments}
    for m in out.moments:
        try:
            seg = resolve(ep, m.start_id, m.end_id, cfg.moment_min_s, cfg.moment_max_s, cfg)
        except ResolveError as e:
            errors.append(f"moment {m.start_id}..{m.end_id}: {e}")
            continue
        if (seg.start_id, seg.end_id) in seen_m:
            continue
        seen_m.add((seg.start_id, seg.end_id))
        an.repaired += int(seg.repaired)
        an.moments.append(MomentSeg(id=f"{ep.id}_m{len(an.moments) + 1}", role=m.role, thread=m.thread.strip(),
                                    summary=m.summary.strip(), strength=_clamp(m.strength), segment=seg))


def stage1_episode(ep: Episode, llm: LLMClient, cfg: Settings) -> EpisodeAnalysis:
    system = render(EXTRACT, cfg)
    user = f"EPISODE {ep.id} ({ep.duration:.0f}s)\nEVENTS:\n{episode_view(ep, cfg)}"
    out = llm.invoke(EpisodeOut, system, user)
    an = EpisodeAnalysis(ep=ep.id, summary=out.summary, key_facts=[f.strip() for f in out.key_facts if f.strip()])
    errors: list[str] = []
    _resolve_out(ep, out, cfg, an, errors)
    if errors:                                   # ONE retry with the error messages, then drop
        retry = user + "\n\n" + render(RETRY, cfg, errors="\n".join(f"- {e}" for e in errors))
        try:
            out2 = llm.invoke(EpisodeOut, system, retry)
            errors2: list[str] = []
            _resolve_out(ep, out2, cfg, an, errors2)
            an.dropped += [f"{e} (after retry)" for e in errors2]
        except RuntimeError as e:
            an.dropped += errors + [f"retry failed: {e}"]
    if cfg.rule_ending:                          # backup: most episodes end on a cliffhanger
        seg = rule_ending(ep, cfg)
        if seg and not any(c.segments[0].start_id == seg.start_id for c in an.clips):
            an.clips.append(Candidate(cid=f"{ep.id}_end", source="rule_ending", segments=[seg],
                                      why="Last seconds of the episode (rule-based backup).", strength=3))
    return an


# ------------------------------------------------------------------ Stage 2
def stage2_combine(show: Show, analyses: dict[str, EpisodeAnalysis], llm: LLMClient,
                   cfg: Settings) -> tuple[list[Candidate], list[str]]:
    """Cards are short summaries (not raw text) so the whole show fits in one call."""
    moments = [m for e in show.episodes if e.id in analyses for m in analyses[e.id].moments]
    if not moments:
        return [], []
    by_id = {m.id: m for m in moments}
    cards = "\n".join(f"{m.id} | {m.segment.ep} | {m.segment.dur:.0f}s | {m.role} | thread: {m.thread} | "
                      f"\"{m.summary[:140]}\"" for m in moments)
    dropped: list[str] = []
    try:
        out = llm.invoke(ComboOut, render(COMBINE, cfg), "CARDS:\n" + cards)
    except RuntimeError as e:
        return [], [f"combine call failed: {e}"]
    cands: list[Candidate] = []
    seen: set[tuple[str, ...]] = set()
    for item in out.combos[: cfg.max_combos]:
        ids = [i.strip().strip("`'\"[]() ") for i in item.moment_ids]
        if not (cfg.combo_min_moments <= len(ids) <= cfg.combo_max_moments) or len(set(ids)) != len(ids):
            dropped.append(f"combo {ids}: needs {cfg.combo_min_moments}-{cfg.combo_max_moments} distinct moments")
            continue
        if any(i not in by_id for i in ids):
            dropped.append(f"combo {ids}: unknown moment id")
            continue
        segs = [by_id[i].segment for i in ids]
        why = check_combo(segs, cfg)
        if why:
            dropped.append(f"combo {ids}: {why}")
            continue
        if tuple(ids) in seen:
            continue
        seen.add(tuple(ids))
        cands.append(Candidate(cid=f"combo_{len(cands) + 1}", source="combo", segments=segs, why=item.why,
                               strength=sum(by_id[i].strength for i in ids) / len(ids), ref_ids=ids))
    return cands, dropped


def prefilter(cands: list[Candidate], cfg: Settings) -> list[Candidate]:
    """Cheap filter before the 3 evaluation calls: top-N by Stage-1 strength + a few per episode + all combos."""
    ranked = sorted(cands, key=lambda c: (-c.strength, c.source != "llm_clip", c.cid))
    keep = {c.cid for c in ranked[: cfg.top_n]} | {c.cid for c in ranked if c.source == "combo"}
    per_ep: dict[str, int] = {}
    for c in ranked:
        if per_ep.get(c.episodes[0], 0) < cfg.min_per_episode:
            keep.add(c.cid)
        per_ep[c.episodes[0]] = per_ep.get(c.episodes[0], 0) + 1
    return [c for c in ranked if c.cid in keep]


# ------------------------------------------------------------------ Stage 4
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def verify_quote(quote: str, line_id: str, cand: Candidate) -> bool:
    """The scorer's quoted evidence must really exist in that event (guards against invented evidence)."""
    q = _norm(quote)
    if len(q) < 3:
        return False
    for s in cand.segments:
        for ev in s.events:
            if _norm(ev.id) == _norm(line_id):
                t = _norm(ev.text)
                # exact substring, or the model quoted with a speaker prefix ("JULIANA: ...")
                return q in t or (len(t) >= 3 and t in q)
    return False


def evaluate(cand: Candidate, facts: list[str], llm: LLMClient, cfg: Settings) -> Evaluation:
    """3 independent calls. No logline / summary / key_facts is ever shown to the scorer or the recap."""
    text = candidate_text(cand, lambda e: event_line(e, cfg))
    ev = Evaluation()
    try:
        score = llm.invoke(Score, render(SCORE, cfg), text)                  # 4a scorer
        recap = llm.invoke(Recap, render(RECAP, cfg), text)                  # 4b blind recap
        ev.score, ev.recap = score, recap
        ev.hook_verified = verify_quote(score.hook_quote, score.hook_line_id, cand)
        ev.cliff_verified = verify_quote(score.cliff_quote, score.cliff_line_id, cand)
        ev.hook_score, ev.cliff_score = _clamp(score.hook_score), _clamp(score.cliff_score)
        if not ev.hook_verified:
            ev.hook_score = min(ev.hook_score, cfg.unverified_cap)
        if not ev.cliff_verified:
            ev.cliff_score = min(ev.cliff_score, cfg.unverified_cap)
        if facts:                                                            # 4c compare
            numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
            user = (f"GROUND TRUTH FACTS:\n{numbered}\n\nRECAP:\nwho: {recap.who}\nwhat: {recap.what}\n"
                    f"stakes: {recap.stakes}")
            ev.check = llm.invoke(FactCheck, render(COMPARE, cfg), user)
            verdict = {v.fact_no: v.verdict for v in ev.check.verdicts}
            ok = [i for i in range(1, len(facts) + 1) if verdict.get(i) == "correct"]
            ev.match = len(ok) / len(facts)                                  # computed by code, not by the model
            ev.missing_facts = [facts[i - 1] for i in range(1, len(facts) + 1) if i not in ok]
    except RuntimeError as e:
        ev.error = str(e)
    return ev


# ------------------------------------------------------------------ Stage 5
def _dialogue_line(c: Candidate, last: bool = False) -> str:
    evs = [e for s in c.segments for e in s.events]
    pool = [e for e in evs if e.kind == "DIALOGUE"] or evs
    e = pool[-1] if last else pool[0]
    return f'{e.speaker}: "{e.text}"' if e.kind == "DIALOGUE" else e.text


def build_promo(c: Candidate, ev: Evaluation, cfg: Settings) -> Promo:
    flags: list[str] = []
    match = ev.match if ev.match is not None else 0.5
    if ev.match is None:
        flags.append("no_key_facts")
    elif ev.match < cfg.needs_context_below:
        flags.append("needs_context")
    if ev.hook_score <= cfg.weak_hook_max:
        flags.append("weak_hook")
    if ev.cliff_score <= cfg.resolved_ending_max:
        flags.append("resolved_ending")
    if c.segments[-1].events[-1].text.rstrip()[-1:] not in cfg.terminal_chars:
        flags.append("cut_mid_sentence")
    if c.cross_episode:
        flags.append("cross_episode")
    if ev.score and not ev.hook_verified:
        flags.append("hook_quote_unverified")
    if ev.score and not ev.cliff_verified:
        flags.append("cliff_quote_unverified")
    standalone = match * 5
    overall = cfg.w_standalone * standalone + cfg.w_hook * ev.hook_score + cfg.w_cliff * ev.cliff_score
    return Promo(
        candidate=c, evaluation=ev, overall=round(overall, 3), standalone=round(standalone, 2), flags=flags,
        hook_text=ev.score.hook_quote if (ev.score and ev.hook_verified) else _dialogue_line(c),
        cliff_text=ev.score.cliff_quote if (ev.score and ev.cliff_verified) else _dialogue_line(c, True),
        rank_reason=(f"hook {ev.hook_score}/5, cliffhanger {ev.cliff_score}/5, standalone {standalone:.1f}/5 "
                     f"(blind recap matched {match:.0%} of key facts)"))


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    inter = sum(max(0.0, min(x.end, y.end) - max(x.start, y.start))
                for x in a.segments for y in b.segments if x.ep == y.ep)
    base = min(a.dur, b.dur)
    return inter / base if base > 0 else 0.0


def rank(promos: list[Promo], cfg: Settings) -> list[Promo]:
    """Flagged promos are only marked, never pushed down: the editor decides."""
    kept: list[Promo] = []
    per_ep: dict[str, int] = {}
    for p in sorted(promos, key=lambda p: (-p.overall, p.candidate.cid)):
        c = p.candidate
        if any(overlap_ratio(c, k.candidate) > cfg.overlap_max for k in kept):
            continue
        if per_ep.get(c.episodes[0], 0) >= cfg.per_episode_max:   # counted against the hook episode
            continue
        per_ep[c.episodes[0]] = per_ep.get(c.episodes[0], 0) + 1
        kept.append(p)
    for i, p in enumerate(kept, 1):
        p.rank = i
    return kept


# ------------------------------------------------------------------ run + export
class RunResult(BaseModel):
    show: str
    model: str
    created: str
    promos: list[Promo]
    failed: list[dict] = []
    analyses: dict[str, EpisodeAnalysis] = {}
    stats: dict = {}
    eval_rows: list[dict] = []
    llm_log: list[dict] = []
    samples: dict = {}
    stage_seconds: dict = {}
    checks: list[Check] = []


def _pmap(fn, items: list, workers: int, stage: str, progress: Optional[Progress]) -> list:
    out: list = [None] * len(items)
    if not items:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                out[futs[f]] = (True, f.result())
            except Exception as e:  # noqa: BLE001 - one bad call must not kill the run
                out[futs[f]] = (False, repr(e))
            if progress:
                progress(stage, n, len(items))
    return out


def first_n(show: Show, n: int) -> Show:
    """Sample mode: only the first n episodes (cheap way to test a key / model / prompt change)."""
    return show if n <= 0 or n >= len(show.episodes) else show.model_copy(update={"episodes": show.episodes[:n]})


def _first_error(failed: list[dict]) -> str:
    e = failed[0]["error"] if failed else "unknown"
    return e[:700] if "HINT:" in e else f"{e[:500]}\nHINT: {classify_error(e)[1]}"


def estimate_calls(show: Show, cfg: Settings) -> int:
    n = len(show.episodes)
    cands = n * (cfg.clips_max + int(cfg.rule_ending)) + cfg.max_combos
    ev = min(cands, cfg.top_n + n * cfg.min_per_episode + cfg.max_combos)
    return n + 1 + ev * 3


def run_pipeline(show: Show, cfg: Settings, llm: LLMClient, progress: Optional[Progress] = None) -> RunResult:
    failed: list[dict] = []
    seconds: dict[str, float] = {}
    llm.warmup()                                   # build all models on the main thread; fails here if misconfigured
    t0 = time.time()
    res = _pmap(lambda ep: stage1_episode(ep, llm, cfg), show.episodes, cfg.workers, "Stage 1: episodes", progress)
    seconds["stage1"] = round(time.time() - t0, 1)
    analyses: dict[str, EpisodeAnalysis] = {}
    for ep, (ok, val) in zip(show.episodes, res):
        if ok:
            analyses[ep.id] = val
        else:
            failed.append({"stage": "stage1", "id": ep.id, "error": val})
    if not analyses:                               # never "succeed" with nothing: say what went wrong
        raise RuntimeError(f"Stage 1 failed for all {len(show.episodes)} episodes.\nFirst error: {_first_error(failed)}")
    if progress:
        progress("Stage 2: combos", 0, 1)
    t0 = time.time()
    combos, combo_drops = stage2_combine(show, analyses, llm, cfg)
    seconds["stage2"] = round(time.time() - t0, 1)
    if progress:
        progress("Stage 2: combos", 1, 1)

    cands = [c for a in analyses.values() for c in a.clips] + combos
    chosen = prefilter(cands, cfg)
    facts_for = lambda c: [f for ep in c.episodes for f in analyses[ep].key_facts]  # noqa: E731
    t0 = time.time()
    res4 = _pmap(lambda c: evaluate(c, facts_for(c), llm, cfg), chosen, cfg.workers, "Stage 4: evaluation", progress)
    seconds["stage4"] = round(time.time() - t0, 1)
    promos: list[Promo] = []
    eval_rows: list[dict] = []
    for c, (ok, val) in zip(chosen, res4):
        ev = val if ok else Evaluation(error=val)
        if ev.error:
            failed.append({"stage": "stage4", "id": c.cid, "error": ev.error})
        else:
            promos.append(build_promo(c, ev, cfg))
            eval_rows.append({"cid": c.cid, "hook_score": ev.hook_score, "cliff_score": ev.cliff_score,
                              "hook_verified": ev.hook_verified, "cliff_verified": ev.cliff_verified,
                              "match": ev.match, "recap_unknown": (ev.recap.who or "").strip().lower() in ("unknown", "")})
    if chosen and not promos:
        raise RuntimeError(f"Stage 4 failed for all {len(chosen)} candidates.\nFirst error: {_first_error(failed)}")
    kept = rank(promos, cfg)

    flags: dict[str, int] = {}
    for p in kept:
        for f in p.flags:
            flags[f] = flags.get(f, 0) + 1
    stats = {
        "episodes": len(show.episodes), "events": sum(len(e.events) for e in show.episodes),
        "clips_accepted": sum(1 for c in cands if c.source == "llm_clip"),
        "rule_endings": sum(1 for c in cands if c.source == "rule_ending"),
        "fixed_by_code": sum(a.repaired for a in analyses.values()),
        "dropped_invalid": sum(len(a.dropped) for a in analyses.values()), "combos_valid": len(combos),
        "combos_dropped": len(combo_drops), "evaluated": len(chosen), "ranked": len(kept),
        "flag_counts": flags, "llm_calls": llm.stats["calls"], "cache_hits": llm.stats["cache_hits"],
        "failed_calls": len(failed), "moments": sum(len(a.moments) for a in analyses.values()),
        "combo_drop_reasons": combo_drops[:20],
    }
    result = RunResult(show=show.name, model=llm.describe(), created=datetime.now().isoformat(timespec="seconds"),
                       promos=kept, failed=failed, analyses=analyses, stats=stats, eval_rows=eval_rows,
                       llm_log=llm.log, samples=llm.samples, stage_seconds=seconds)
    result.checks = health_checks(show, result, cfg)
    return result


def promo_to_dict(p: Promo) -> dict:
    c, ev = p.candidate, p.evaluation
    return {
        "rank": p.rank, "promo_id": c.cid, "type": "combo" if len(c.segments) > 1 else "single",
        "source": c.source, "duration_s": round(c.dur, 2),
        "segments": [{"episode": s.ep, "start": fmt_ts(s.start), "end": fmt_ts(s.end), "start_sec": s.start,
                      "end_sec": s.end, "start_id": s.start_id, "end_id": s.end_id,
                      "repaired_by_code": s.repaired} for s in c.segments],
        "hook": p.hook_text, "cliffhanger": p.cliff_text, "why_it_works": c.why, "rank_reason": p.rank_reason,
        "overall": p.overall,
        "scores": {"hook": ev.hook_score, "cliffhanger": ev.cliff_score, "standalone": p.standalone, "match": ev.match},
        "flags": p.flags, "blind_recap": ev.recap.model_dump() if ev.recap else None,
        "missing_facts": ev.missing_facts,
    }


def save_outputs(result: RunResult, show: Show, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    (out / "parsed").mkdir(parents=True, exist_ok=True)
    for ep in show.episodes:                                   # Stage 0 output
        (out / "parsed" / f"{ep.id}.json").write_text(ep.model_dump_json(indent=1), encoding="utf-8")
    rows = [promo_to_dict(p) for p in result.promos]
    (out / "promos.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "type", "segments", "duration_s", "hook", "cliffhanger", "why_it_works", "overall",
                    "hook_score", "cliff_score", "match", "flags", "human_ok(y/n)", "human_note"])
        for r in rows:
            segs = " + ".join(f"{s['episode']} {s['start']}-{s['end']}" for s in r["segments"])
            w.writerow([r["rank"], r["type"], segs, r["duration_s"], r["hook"], r["cliffhanger"], r["why_it_works"],
                        r["overall"], r["scores"]["hook"], r["scores"]["cliffhanger"], r["scores"]["match"],
                        ";".join(r["flags"]), "", ""])
    with open(out / "llm_calls.jsonl", "w", encoding="utf-8") as f:       # one row per API attempt
        for row in result.llm_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "debug_samples.json").write_text(json.dumps(result.samples, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "health_report.txt").write_text(render_checks(result.checks), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(
        {"show": result.show, "model": result.model, "created": result.created, "stats": result.stats,
         "stage_seconds": result.stage_seconds, "failed": result.failed}, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
