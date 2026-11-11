"""streamlit run app.py"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from promo_finder.config import ROOT, load_settings
from promo_finder.debug import ICON, parse_checks
from promo_finder.llm import PROVIDER_KEY_ENV, LLMClient, classify_error, preflight
from promo_finder.parser import episode_view, load_show
from promo_finder.pipeline import estimate_calls, first_n, promo_to_dict, run_pipeline, save_outputs
from promo_finder.timeutil import fmt_ts

load_dotenv()
st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Promo Finder")
st.caption("Screenplays → events JSON → clips & moments → combos → scored, flagged, ranked promos")
base = load_settings()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    providers = list(PROVIDER_KEY_ENV)
    prov = st.selectbox("Provider", providers, index=providers.index(base.provider) if base.provider in providers else 0)
    model = st.text_input("Model", base.model)
    base_url = st.text_input("Base URL (OpenAI-compatible APIs only)", base.base_url)
    key_env = PROVIDER_KEY_ENV.get(prov)
    if key_env:
        key = st.text_input(key_env, type="password", value=os.getenv(key_env, ""))
        if key:
            os.environ[key_env] = key
    workers = st.slider("Parallel calls", 1, 16, base.workers)
    rpm = st.number_input("Max requests / minute (0 = off; ~10 for free tiers)", 0, 600, base.rpm)
    use_cache = st.checkbox("Use cache", base.use_cache)

    st.header("Promo rules")
    min_s, max_s = st.slider("Length (s)", 5, 120, (int(base.min_s), int(base.max_s)))
    top_n = st.slider("Candidates to evaluate (cost)", 5, 300, base.top_n)
    min_pe = st.slider("Min candidates kept per episode", 0, 5, base.min_per_episode)
    per_ep = st.slider("Max promos / episode", 1, 10, base.per_episode_max)
    st.caption("Rank weights")
    w1 = st.slider("standalone", 0.0, 1.0, base.w_standalone)
    w2 = st.slider("hook", 0.0, 1.0, base.w_hook)
    w3 = st.slider("cliffhanger", 0.0, 1.0, base.w_cliff)

    st.header("This show")
    audience = st.text_input("Audience", base.audience)
    genre = st.text_input("Genre hint", base.genre_hint)
    extra = st.text_area("Extra instructions", base.extra_instructions, height=70)

cfg = load_settings({
    "provider": prov, "model": model, "base_url": base_url, "workers": workers, "rpm": int(rpm),
    "min_per_episode": min_pe, "use_cache": use_cache,
    "min_s": min_s, "max_s": max_s, "top_n": top_n, "per_episode_max": per_ep, "w_standalone": w1, "w_hook": w2,
    "w_cliff": w3, "audience": audience, "genre_hint": genre, "extra_instructions": extra})

# ---------------------------------------------------------------- debug: API connection test
with st.expander("🔧 Debug: test the API connection first", expanded=False):
    st.caption("One tiny call, no retries. Shows the real error and what to fix.")
    if st.button("Test API connection"):
        with st.spinner("calling the model..."):
            pf = preflight(cfg)
        for stp in pf["steps"]:
            (st.success if stp["ok"] else st.error)(f"**{stp['name']}**: {stp['detail']}")
        if pf["ok"]:
            st.success("Connection works. Now run the pipeline.")

# ---------------------------------------------------------------- input
up = st.file_uploader("Upload a .zip of episode scripts (or several .txt files)", type=["zip", "txt", "md"],
                      accept_multiple_files=True)
folder = st.text_input("...or a local folder / zip path", "")
source = [(f.name, f.getvalue()) for f in up] if up else (folder.strip() or None)

show = None
if source:
    try:
        show = load_show(source, cfg)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not parse input: {e}")

if show:
    a, b, c = st.columns(3)
    a.metric("Episodes", len(show.episodes))
    b.metric("Parsed events", sum(len(e.events) for e in show.episodes))
    c.metric("Est. LLM calls (max)", estimate_calls(show, cfg))
    if show.warnings:
        with st.expander(f"{len(show.warnings)} parser warnings"):
            st.write(show.warnings)
    with st.expander("🔧 Debug: Stage 0 checks"):
        for c in parse_checks(show, cfg):
            st.write(f"{ICON[c.status]} **{c.name}**: {c.detail}")
    with st.expander("Stage 0: parsed JSON and the exact view the LLM sees"):
        ep = show.episode(st.selectbox("Episode", [e.id for e in show.episodes]))
        t1, t2 = st.tabs(["LLM view", "JSON"])
        t1.code(episode_view(ep, cfg), language="text")
        t2.json(json.loads(ep.model_dump_json()), expanded=1)

    sample_n = st.number_input("Sample mode: run only the first N episodes (0 = all). Use 2-3 to test cheaply.",
                               0, len(show.episodes), 0)
    if st.button("▶ Find promos", type="primary"):
        run_show = first_n(show, int(sample_n))
        bar = st.progress(0.0, text="starting")
        order = ["Stage 1: episodes", "Stage 2: combos", "Stage 4: evaluation"]

        def progress(stage, done, total):
            i = order.index(stage) if stage in order else 0
            bar.progress(min(1.0, (i + done / max(total, 1)) / len(order)), text=f"{stage} {done}/{total}")

        try:
            llm = LLMClient(cfg)
            res = run_pipeline(run_show, cfg, llm, progress)
            out = save_outputs(res, run_show, ROOT / "output")
            st.session_state["res"], st.session_state["out"] = res, out
            bar.progress(1.0, text="done")
        except Exception as e:  # noqa: BLE001
            st.error(f"Run failed:\n\n{e}")
            st.info(f"Hint: {classify_error(str(e))[1]}")
            if 'llm' in locals() and llm.log:
                st.caption("Last API attempts:")
                st.dataframe(pd.DataFrame(llm.log[-8:])[["schema", "status", "seconds", "error"]], width="stretch")

# ---------------------------------------------------------------- results
res = st.session_state.get("res")
if res:
    st.divider()
    rows = [promo_to_dict(p) for p in res.promos]
    st.subheader(f"{len(rows)} ranked promos · {res.model}")
    s = res.stats
    st.caption(f"{s['llm_calls']} LLM calls ({s['cache_hits']} cached) · fixed by code: {s['fixed_by_code']} · "
               f"dropped invalid: {s['dropped_invalid']} · combos valid/dropped: {s['combos_valid']}/{s['combos_dropped']} "
               f"· failed calls: {s['failed_calls']}")
    if s["flag_counts"]:
        st.markdown("**Flags (where the tool may pick badly):** " +
                    " · ".join(f"`{k}` {v}" for k, v in sorted(s["flag_counts"].items(), key=lambda x: -x[1])))
    with st.expander("🔧 Debug: pipeline health (is every stage doing something sensible?)", expanded=True):
        for c in res.checks:
            st.write(f"{ICON[c.status]} **[{c.stage}] {c.name}**: {c.detail}")
        st.caption("Seconds per stage: " + ", ".join(f"{k} {v}s" for k, v in res.stage_seconds.items()))
    if res.failed:
        with st.expander(f"🔧 Debug: {len(res.failed)} failed calls"):
            st.dataframe(pd.DataFrame([{**f, "hint": classify_error(f["error"])[1][:90]} for f in res.failed]),
                         width="stretch")
    with st.expander("🔧 Debug: every API call (stage, status, seconds, error)"):
        if res.llm_log:
            st.dataframe(pd.DataFrame(res.llm_log), width="stretch")
    with st.expander("🔧 Debug: what the model saw and returned (first sample per schema)"):
        st.json(res.samples, expanded=False)
    hide = st.checkbox("Hide flagged promos", False)
    rows = [r for r in rows if not (hide and set(r["flags"]) - {"cross_episode"})]
    st.dataframe(pd.DataFrame([{
        "#": r["rank"], "type": r["type"],
        "where": " + ".join(f"{x['episode']} {x['start']}-{x['end']}" for x in r["segments"]),
        "dur": r["duration_s"], "hook": r["scores"]["hook"], "cliff": r["scores"]["cliffhanger"],
        "standalone": r["scores"]["standalone"], "overall": r["overall"], "flags": ", ".join(r["flags"]),
    } for r in rows]), width="stretch", hide_index=True)

    if rows:
        pick = st.selectbox("Inspect", [f"#{r['rank']} {r['promo_id']}" for r in rows])
        r = next(x for x in rows if f"#{x['rank']} {x['promo_id']}" == pick)
        p = next(x for x in res.promos if x.rank == r["rank"])
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"**Hook:** {r['hook']}\n\n**Cliffhanger:** {r['cliffhanger']}\n\n**Why:** {r['why_it_works']}")
            st.caption(r["rank_reason"])
            for seg in p.candidate.segments:
                st.markdown(f"`{seg.ep}` **{fmt_ts(seg.start)} → {fmt_ts(seg.end)}**")
                st.code("\n".join(f"[{fmt_ts(e.start, 1)}] {(e.speaker + ': ') if e.speaker else e.kind + ' '}{e.text}"
                                  for e in seg.events), language="text")
        with right:
            st.markdown("**Flags:** " + (", ".join(r["flags"]) or "none"))
            if r["blind_recap"]:
                st.markdown("**Blind recap (what a new viewer gets)**")
                st.json(r["blind_recap"])
            if r["missing_facts"]:
                st.markdown("**Key facts the recap missed**")
                st.write(r["missing_facts"])

    out = Path(st.session_state["out"])
    d1, d2 = st.columns(2)
    d1.download_button("⬇ promos.json", (out / "promos.json").read_bytes(), "promos.json")
    d2.download_button("⬇ promos.csv", (out / "promos.csv").read_bytes(), "promos.csv")
    st.caption(f"Raw outputs (promos.json/csv, run_meta.json, parsed/*.json) are saved in: {out}")
