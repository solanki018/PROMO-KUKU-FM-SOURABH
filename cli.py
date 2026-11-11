"""python cli.py --input pack.zip [--provider openai --model gpt-4o-mini] [--set max_s=45] [--episodes 2] [--dry-run]"""
from __future__ import annotations

import argparse
import sys
import time

import yaml
from dotenv import load_dotenv

from promo_finder.config import load_settings, resolve_dir
from promo_finder.debug import parse_checks, render_checks
from promo_finder.llm import LLMClient, classify_error
from promo_finder.parser import load_show
from promo_finder.pipeline import estimate_calls, first_n, run_pipeline, save_outputs


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Find promo clips in microdrama screenplays")
    ap.add_argument("--input", required=True, help="folder, .zip or single .txt of episode scripts")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", type=int, default=0, help="sample mode: only the first N episodes")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL", help="e.g. max_s=45")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="parse + estimate calls, no LLM")
    a = ap.parse_args()

    ov: dict = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        ov[k.strip()] = yaml.safe_load(v)
    if a.provider:
        ov["provider"] = a.provider
    if a.model:
        ov["model"] = a.model
    if a.no_cache:
        ov["use_cache"] = False
    cfg = load_settings(ov, a.config)

    show = first_n(load_show(a.input, cfg), a.episodes)
    print(f"Parsed {len(show.episodes)} episodes, {sum(len(e.events) for e in show.episodes)} events "
          f"({len(show.warnings)} warnings)")
    print(render_checks(parse_checks(show, cfg)))
    print("Estimated LLM calls (max):", estimate_calls(show, cfg))
    if a.dry_run:
        return 0

    llm = LLMClient(cfg)
    t0 = time.time()
    last = {"s": ""}

    def progress(stage, done, total):
        if stage != last["s"] or done == total:
            print(f"  {stage}: {done}/{total}", flush=True)
            last["s"] = stage

    try:
        res = run_pipeline(show, cfg, llm, progress)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"\nFAILED: {msg}" + ("" if "HINT:" in msg else f"\nHINT: {classify_error(msg)[1]}"))
        for r in llm.log[-5:]:
            print(f"  {r['schema']} {r['status']}: {r['error'][:160]}")
        print("\nTip: python debug_check.py --input", a.input, " (step-by-step diagnosis)")
        return 1
    out = save_outputs(res, show, a.out or resolve_dir(cfg.output_dir))
    print(f"\nDone in {time.time() - t0:.0f}s with {llm.describe()}. {len(res.promos)} promos -> {out}")
    print(render_checks(res.checks))
    for p in res.promos[:5]:
        print(f"  #{p.rank} {p.candidate.cid} overall {p.overall:.2f} flags={p.flags}")
    return 0 if res.promos else 2


if __name__ == "__main__":
    sys.exit(main())
