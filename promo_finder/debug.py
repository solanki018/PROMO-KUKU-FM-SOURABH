"""Debug helpers: tell, stage by stage, whether the pipeline did something sensible.

    parse_checks(show, cfg)             -> after Stage 0 (no API needed)
    health_checks(show, result, cfg)    -> after a run: Stage 1..5, from the real outputs
Every check is PASS / WARN / FAIL with a plain-language detail and a hint for the fix.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Literal

from pydantic import BaseModel

from .config import Settings
from .schemas import Show


class Check(BaseModel):
    stage: str
    name: str
    status: Literal["PASS", "WARN", "FAIL"]
    detail: str


ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}


def _c(stage, name, ok, detail, warn=False) -> Check:
    return Check(stage=stage, name=name, status="PASS" if ok else ("WARN" if warn else "FAIL"), detail=detail)


def render_checks(checks: list[Check]) -> str:
    return "\n".join(f"{ICON[c.status]} [{c.stage}] {c.name}: {c.detail}" for c in checks)


# ------------------------------------------------------------------ Stage 0
def parse_checks(show: Show, cfg: Settings) -> list[Check]:
    st = "Stage 0 parse"
    out: list[Check] = []
    n_ev = [len(e.events) for e in show.episodes]
    out.append(_c(st, "episodes parsed", len(show.episodes) > 0, f"{len(show.episodes)} episodes, "
                  f"{sum(n_ev)} events (min {min(n_ev, default=0)}, max {max(n_ev, default=0)} per episode)"))
    short = [e.id for e in show.episodes if e.duration < cfg.min_s]
    out.append(_c(st, "episodes long enough for a promo", not short,
                  "all >= min promo length" if not short else f"shorter than {cfg.min_s:g}s: {short[:8]}", warn=True))
    kinds = Counter(e.kind for ep in show.episodes for e in ep.events)
    out.append(_c(st, "event kinds", kinds["DIALOGUE"] > 0,
                  ", ".join(f"{k}={v}" for k, v in kinds.items()) +
                  ("" if kinds["DIALOGUE"] else "  <- no dialogue: check speaker label format / parser settings")))
    bad = [e.id for ep in show.episodes for e in ep.events if e.end < e.start]
    out.append(_c(st, "timestamps valid (end >= start)", not bad, "ok" if not bad else f"bad events: {bad[:5]}"))
    no_logline = [e.id for e in show.episodes if not e.logline]
    out.append(_c(st, "logline found (never sent to evaluation)", not no_logline,
                  "all episodes" if not no_logline else f"missing in {len(no_logline)} episodes (harmless)", warn=True))
    out.append(_c(st, "parser warnings", not show.warnings,
                  "none" if not show.warnings else f"{len(show.warnings)}: {show.warnings[:3]}", warn=True))
    return out


# ------------------------------------------------------------------ Stage 1-5
def health_checks(show: Show, res, cfg: Settings) -> list[Check]:
    s = res.stats
    out: list[Check] = []
    n_eps = len(show.episodes)

    # ---- Stage 1
    st = "Stage 1 extract"
    ok_eps = [a for a in res.analyses.values() if not a.error]
    out.append(_c(st, "episodes analysed", len(ok_eps) == n_eps,
                  f"{len(ok_eps)}/{n_eps} episodes returned a valid answer", warn=len(ok_eps) > 0))
    if ok_eps:
        kf = mean(len(a.key_facts) for a in ok_eps)
        out.append(_c(st, "key_facts per episode", kf >= 3, f"avg {kf:.1f} (asked {cfg.facts_min}-{cfg.facts_max})",
                      warn=True))
        cl = mean(sum(1 for c in a.clips if c.source == "llm_clip") for a in ok_eps)
        mo = mean(len(a.moments) for a in ok_eps)
        out.append(_c(st, "clips / moments accepted per episode", cl >= 1 and mo >= 1,
                      f"avg {cl:.1f} clips, {mo:.1f} moments (asked {cfg.clips_min}-{cfg.clips_max} / "
                      f"{cfg.moments_min}-{cfg.moments_max})", warn=True))
        total_out = s["clips_accepted"] + s["moments"] + s["dropped_invalid"]
        drop = s["dropped_invalid"] / total_out if total_out else 0
        reasons = Counter()
        for a in ok_eps:
            for d in a.dropped:
                reasons["unknown id" if "does not exist" in d else "duration" if "lasts" in d else
                        "order" if "before" in d else "other"] += 1
        out.append(_c(st, "invalid items dropped after retry", drop < 0.3,
                      f"{s['dropped_invalid']} of {total_out} ({drop:.0%}); reasons {dict(reasons)}"
                      + ("" if drop < 0.3 else "  <- model returns bad ids/durations: use a stronger model"),
                      warn=True))
        ex = next((d for a in ok_eps for d in a.dropped), None)
        if ex:
            out.append(Check(stage=st, name="example dropped item", status="PASS", detail=ex[:200]))

    # ---- Stage 2
    st = "Stage 2 combos"
    out.append(_c(st, "combos valid", s["combos_valid"] > 0,
                  f"{s['combos_valid']} valid, {s['combos_dropped']} dropped"
                  + (f"; e.g. {s['combo_drop_reasons'][0]}" if s.get("combo_drop_reasons") else ""), warn=True))

    # ---- Stage 3: re-verify the bounds on real segments
    st = "Stage 3 validate"
    cands = [c for a in res.analyses.values() for c in a.clips]
    durs = [c.dur for c in cands]
    bad = [c.cid for c in cands if not (cfg.min_s - 1e-6 <= c.dur <= cfg.max_s + 1e-6)]
    out.append(_c(st, f"every clip is {cfg.min_s:g}-{cfg.max_s:g}s", not bad and bool(durs),
                  f"{len(durs)} clips, {min(durs, default=0):.0f}-{max(durs, default=0):.0f}s"
                  + ("" if not bad else f"; OUT OF RANGE: {bad[:5]}")))
    out.append(_c(st, "fixed automatically by code", True, f"{s['fixed_by_code']} clips/moments trimmed or extended"))

    # ---- Stage 4
    st = "Stage 4 evaluate"
    rows = res.eval_rows
    out.append(_c(st, "candidates evaluated", len(rows) > 0 and s["failed_calls"] == 0,
                  f"{len(rows)} of {s['evaluated']} evaluated, {s['failed_calls']} failed calls",
                  warn=len(rows) > 0))
    if rows:
        hv = mean(1 if r["hook_verified"] else 0 for r in rows)
        cv = mean(1 if r["cliff_verified"] else 0 for r in rows)
        out.append(_c(st, "scorer quotes found in the clip", min(hv, cv) >= 0.7,
                      f"hook {hv:.0%}, cliffhanger {cv:.0%}"
                      + ("" if min(hv, cv) >= 0.7 else "  <- model paraphrases / invents lines; scores are capped"),
                      warn=True))
        hs = Counter(r["hook_score"] for r in rows)
        cs = Counter(r["cliff_score"] for r in rows)
        spread = len(hs) > 1 and len(cs) > 1
        out.append(_c(st, "scores discriminate", spread,
                      f"hook {dict(sorted(hs.items()))}, cliff {dict(sorted(cs.items()))}"
                      + ("" if spread else "  <- all the same value: judge is not separating good from bad"),
                      warn=True))
        ms = [r["match"] for r in rows if r["match"] is not None]
        if ms:
            ok = 0.05 < mean(ms) < 0.95
            out.append(_c(st, "blind recap vs key facts (match)", ok,
                          f"mean {mean(ms):.2f}, min {min(ms):.2f}, max {max(ms):.2f}"
                          + ("" if ok else "  <- ~0: recap or compare is broken; ~1: facts too easy / leaking"),
                          warn=True))
        unk = mean(1 if r["recap_unknown"] else 0 for r in rows)
        out.append(_c(st, "recaps that say 'unknown' for who", unk < 0.5, f"{unk:.0%} of recaps", warn=True))

    # ---- Stage 5
    st = "Stage 5 rank"
    out.append(_c(st, "promos ranked", len(res.promos) > 0, f"{len(res.promos)} promos"))
    per_ep = Counter(p.candidate.episodes[0] for p in res.promos)
    out.append(_c(st, f"max {cfg.per_episode_max} per episode respected",
                  max(per_ep.values(), default=0) <= cfg.per_episode_max, f"max found {max(per_ep.values(), default=0)}"))
    combos = [p for p in res.promos if p.candidate.cross_episode]
    out.append(_c(st, "cross-episode combos in ranking", bool(combos), f"{len(combos)} combos ranked", warn=True))
    fc = s.get("flag_counts", {})
    out.append(Check(stage=st, name="flag counts", status="PASS",
                     detail=", ".join(f"{k}={v}" for k, v in sorted(fc.items(), key=lambda x: -x[1])) or "none"))
    return out
