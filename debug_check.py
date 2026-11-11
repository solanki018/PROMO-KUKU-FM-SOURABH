"""Step-by-step check that every stage works, on a few episodes (cheap: ~10 API calls).

    python debug_check.py --input pack.zip                       # 2 episodes
    python debug_check.py --input pack.zip --episodes 1 --provider openai --model gpt-4o-mini

Stops at the first failing step and prints what to fix.
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from promo_finder.config import load_settings
from promo_finder.debug import ICON, parse_checks, render_checks
from promo_finder.llm import LLMClient, classify_error, preflight
from promo_finder.parser import load_show
from promo_finder.pipeline import evaluate, first_n, stage1_episode
from promo_finder.timeutil import fmt_ts


def head(t: str) -> None:
    print(f"\n=== {t} " + "=" * max(3, 60 - len(t)))


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--episodes", type=int, default=2, help="how many episodes to try")
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    ov = {k: v for k, v in {"provider": a.provider, "model": a.model, "base_url": a.base_url}.items() if v}
    if a.no_cache:
        ov["use_cache"] = False
    cfg = load_settings(ov)
    print(f"provider={cfg.provider} model={cfg.model} base_url={cfg.base_url or '-'} cache={cfg.use_cache}")

    head("1. Stage 0: parse input (no API)")
    show = load_show(a.input, cfg)
    print(render_checks(parse_checks(show, cfg)))

    head("2. API connection (one tiny call, no retries)")
    pf = preflight(cfg)
    for s in pf["steps"]:
        print(f"{ICON['PASS' if s['ok'] else 'FAIL']} {s['name']}: {s['detail']}")
    if not pf["ok"]:
        print("\nFIX THIS FIRST. Nothing after this can work until the connection test passes.")
        return 1

    sub = first_n(show, a.episodes)
    llm = LLMClient(cfg)
    llm.warmup()

    head(f"3. Stage 1 + 3: extract + validate on {len(sub.episodes)} episode(s)")
    analyses = []
    for ep in sub.episodes:
        try:
            an = stage1_episode(ep, llm, cfg)
        except RuntimeError as e:
            print(f"❌ {ep.id}: {e}\nHINT: {classify_error(str(e))[1]}")
            return 1
        analyses.append(an)
        llm_clips = [c for c in an.clips if c.source == "llm_clip"]
        print(f"✅ {ep.id}: {len(an.key_facts)} key_facts, {len(llm_clips)} clips, {len(an.moments)} moments, "
              f"{len(an.dropped)} dropped, {an.repaired} fixed by code")
        for f in an.key_facts[:3]:
            print(f"     fact: {f[:100]}")
        for c in llm_clips[:2]:
            s = c.segments[0]
            print(f"     clip {c.cid}: {fmt_ts(s.start)}-{fmt_ts(s.end)} ({s.dur:.0f}s) strength {c.strength:.0f} | {c.why[:70]}")
        for d in an.dropped[:2]:
            print(f"     dropped: {d[:110]}")
        if not llm_clips:
            print("⚠️  no valid clips: see debug_samples in the output folder / the dropped reasons above")

    head("4. Stage 4: scorer + blind recap + compare on 2 candidates")
    cands = [c for an in analyses for c in an.clips][:2]
    facts = {an.ep: an.key_facts for an in analyses}
    for c in cands:
        ev = evaluate(c, [f for ep in c.episodes for f in facts[ep]], llm, cfg)
        if ev.error:
            print(f"❌ {c.cid}: {ev.error[:300]}")
            return 1
        print(f"✅ {c.cid}: hook {ev.hook_score} (quote verified: {ev.hook_verified}), "
              f"cliff {ev.cliff_score} (verified: {ev.cliff_verified}), match {ev.match}")
        print(f"     hook quote: {ev.score.hook_quote[:90]}")
        print(f"     blind recap who: {ev.recap.who[:90]}")
        if ev.missing_facts:
            print(f"     facts the recap missed: {len(ev.missing_facts)}")

    head("5. Summary")
    errs = [r for r in llm.log if r["status"].startswith("error")]
    print(f"API calls ok: {llm.stats['calls']}, cache hits: {llm.stats['cache_hits']}, errors/retries: {len(errs)}")
    for r in errs[:3]:
        print(f"   {r['schema']} {r['status']}: {r['error'][:150]}")
    print("\nALL STAGES WORK. Now run the full app: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
