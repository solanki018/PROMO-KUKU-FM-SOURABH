# Promo Finder

Reads microdrama screenplays and finds 20-60s promo clips (hook in the first 3s, ends on a cliffhanger,
makes sense alone), including **cross-episode combos**. Streamlit + LangChain + Pydantic. Model is swappable.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env            # put ONE api key (Gemini / Groq have free tiers)

streamlit run app.py            # UI: upload the zip, press "Find promos"
# or CLI
python cli.py --input pack.zip --out output
python debug_check.py --input pack.zip              # FIRST: step-by-step check (parse -> API -> each stage), ~10 calls
python cli.py --input pack.zip --episodes 1         # sample mode: run only the first episode
python cli.py --input pack.zip --dry-run            # parse + estimate calls, no LLM
pytest -q                                           # tests, no key needed
```

Swap model with no code change: `LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini python cli.py --input pack.zip`
(or the sidebar in the UI). `provider=mock` is an offline stand-in for plumbing tests only - its scores mean nothing.
Free tier rate limits: `--set rpm=10 --set workers=2 --set top_n=40`.

## Pipeline

| Stage | Who | What |
|---|---|---|
| 0 parse | code | scripts -> `Episode/Event` JSON with stable ids (`ep12_e020`), sorted by time |
| 1 extract | LLM, 1 call/episode | key_facts, 3-5 clips, 4-8 reusable moments - **as ids, never timestamps** |
| 2 combine | LLM, 1 call (chunked if >50 eps) | 2-3 moment cards -> cross-event / cross-episode promos |
| 3 validate | code | ids exist, order ok, 20-60s; deterministic trim/extend; one LLM retry with the error text; then drop |
| 4 evaluate | LLM, 3 fresh calls/candidate | scorer (quote line + id, then score), blind recap, recap vs key_facts |
| 5 rank | code | flags, `0.35*standalone + 0.35*hook + 0.30*cliffhanger`, overlap dedupe, max 3/episode |

Why it hallucinates less:
- the model only chooses ids; code looks up the real timestamps
- scorer must quote the exact line + id; code checks the quote exists in that event, else the score is capped and flagged
- standalone-ness is measured by a *blind* recap (no logline/summary/facts in that call), then compared to key facts;
  `match` is computed by code from per-fact verdicts, not self-reported by the model
- flagged promos are marked (`needs_context`, `weak_hook`, `resolved_ending`, `cut_mid_sentence`,
  `cross_episode`, `*_quote_unverified`), never silently hidden

## Nothing is hardcoded to one show

- `config.yaml`: durations, counts, weights, thresholds, model, workers, rate limit (+ UI sliders, `--set key=value` (config is flat: `--set max_s=45`; a misspelled key now raises an error))
- `prompts/*.md`: all prompts, with placeholders filled from config; `show.audience / genre_hint / extra_instructions`
  steer a new show without code changes
- parser labels / header names / extensions are config (`parser.*`); unparseable blocks become warnings, not crashes
- input can be a zip, folder, or loose .txt files; episode count and length are free

## Cost (50 episodes)

~50 (stage 1) + 1 (stage 2) + up to ~(top_n + combos) x 3 (stage 4) calls; default ~600-700 small calls, cached by
hash(model+prompt), so re-runs are free. Biggest lever: `evaluate.top_n` (cheap Stage-1 strength pre-filter).

## Known limits

- Reads text, not video: acting, music, visuals are invisible; timestamps are only as good as the script's.
- LLM-as-judge scores are noisy; `standalone` is a proxy (recap model may fill gaps from genre tropes).
- Moments the model never proposes are never found (rule-based ending only covers episode ends).
- Combos are chunked by episode window for very long shows, so cross-chunk combos are missed.
- `human_ok` column in `promos.csv` is there so an editor can mark the top 20 and measure precision.

## If it "returns nothing"
1. `python debug_check.py --input your_script.txt` - stops at the first failing step and prints the fix.
2. In the UI: sidebar -> *Debug: test the API connection* (one tiny call, shows the real error).
3. After a run, open `output/health_report.txt`, `output/llm_calls.jsonl` (every API attempt) and `output/debug_samples.json` (what the model saw / returned).
Fatal errors (bad key, wrong model name, no quota) now stop the run immediately with a plain-language hint; an empty model answer counts as an error and is never cached.
