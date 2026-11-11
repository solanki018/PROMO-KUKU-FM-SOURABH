"""Run: pytest -q   (no API key needed)"""
from promo_finder.config import load_settings
from promo_finder.llm import FatalLLMError, LLMClient, classify_error
from promo_finder.parser import load_show, event_line
from promo_finder.pipeline import overlap_ratio, run_pipeline, verify_quote as _verify
from promo_finder.segments import ResolveError, check_combo, resolve, rule_ending
from promo_finder.schemas import Candidate, EpisodeOut, Score
import pytest

SCRIPT = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: secret logline
CONTINUITY NOTES: notes

--- SCRIPT START ---

SCENE 01
Scene heading: INT. HALL - DAY
Scene start: 00:00.00
Scene end: 00:50.00

**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of the hall.

**ANNA (Furious, Shouting):**
Start: (00:01.00) | End: (00:03.00)
"You lied to me!"

**ACTION:**
Start: (00:04.00) | End: (00:08.00)
Anna slaps Ben.

**BEN (O.S.) (Cold):**
Start: (00:08.00) | End: (00:12.00)
"Say that again."

**SOUND FX:**
Start: (00:12.00) | End: (00:14.00)
Door slams.

**ANNA (Low):**
Start: (00:14.00) | End: (00:30.00)
"I know what you did. And so does the Duke."

**BEN (Whispering):**
Start: (00:30.00) | End: (00:50.00)
"Then he must never leave this room"

--- SCRIPT END ---
"""


@pytest.fixture
def show():
    cfg = load_settings()
    return load_show([("episode-007-x.txt", SCRIPT.encode())], cfg), cfg


def test_parser_kinds_ids_and_order(show):
    s, cfg = show
    ep = s.episodes[0]
    assert ep.id == "ep07" and ep.logline == "secret logline"
    assert [e.kind for e in ep.events] == ["CAMERA", "DIALOGUE", "ACTION", "DIALOGUE", "SOUND", "DIALOGUE", "DIALOGUE"]
    assert ep.events[1].speaker == "ANNA" and ep.events[1].tag == "Furious, Shouting"
    assert ep.events[3].voice == "O.S." and ep.events[3].tag == "Cold"
    assert ep.events[1].text == "You lied to me!"          # quotes stripped
    assert ep.events[0].id == "ep07_e000" and ep.duration == 50
    assert '"You lied to me!"' in event_line(ep.events[1], cfg)


def test_resolve_real_times_and_errors(show):
    s, cfg = show
    ep = s.episodes[0]
    seg = resolve(ep, "ep07_e001", "ep07_e006", 20, 60, cfg)
    assert seg.start == 1.0 and seg.end == 50.0
    with pytest.raises(ResolveError):
        resolve(ep, "ep07_e099", "ep07_e006", 20, 60, cfg)     # hallucinated id
    with pytest.raises(ResolveError):
        resolve(ep, "ep07_e006", "ep07_e001", 20, 60, cfg)     # reversed


def test_repair_extends_and_trims(show):
    s, cfg = show
    ep = s.episodes[0]
    short = resolve(ep, "ep07_e001", "ep07_e003", 20, 60, cfg)  # 11s -> extended to >=20
    assert short.repaired and short.dur >= 20
    cfg.repair = False
    with pytest.raises(ResolveError):
        resolve(ep, "ep07_e001", "ep07_e003", 20, 60, cfg)


def test_rule_ending(show):
    s, cfg = show
    seg = rule_ending(s.episodes[0], cfg)
    assert seg and seg.end == 50.0 and 20 <= seg.dur <= 60


def test_combo_rules(show):
    s, cfg = show
    ep = s.episodes[0]
    a = resolve(ep, "ep07_e001", "ep07_e003", 5, 25, cfg)
    b = resolve(ep, "ep07_e005", "ep07_e006", 5, 25, cfg)
    assert check_combo([a, b], cfg) is None or "total" in check_combo([a, b], cfg)
    assert "order" in check_combo([b, a], cfg) or "total" in check_combo([b, a], cfg)


def test_quote_verification(show):
    s, cfg = show
    ep = s.episodes[0]
    seg = resolve(ep, "ep07_e001", "ep07_e006", 20, 60, cfg)
    c = Candidate(cid="x", source="llm_clip", segments=[seg])
    assert _verify("you lied to me", "ep07_e001", c)
    assert not _verify("you never lied", "ep07_e001", c)      # invented quote
    assert not _verify("you lied to me", "ep07_e003", c)      # right text, wrong id


def test_mock_pipeline_runs_and_respects_rules(show, tmp_path):
    s, cfg = show
    cfg.provider, cfg.model, cfg.use_cache = "mock", "mock", False
    res = run_pipeline(s, cfg, LLMClient(cfg))
    assert res.promos
    for p in res.promos:
        assert cfg.min_s <= p.candidate.dur <= cfg.max_s
        assert 1 <= p.evaluation.hook_score <= 5
    assert overlap_ratio(res.promos[0].candidate, res.promos[0].candidate) == 1.0


def test_logline_never_reaches_evaluation_prompts(show):
    s, cfg = show
    cfg.provider, cfg.model, cfg.use_cache = "mock", "mock", False
    seen = []
    client = LLMClient(cfg)
    orig = client.invoke
    client.invoke = lambda schema, system, user: (seen.append((schema.__name__, system + user)), orig(schema, system, user))[1]
    run_pipeline(s, cfg, client)
    for name, text in seen:
        if name in ("Score", "Recap"):
            assert "secret logline" not in text


# ------------------------------------------------------------------ regression tests for the "returns nothing" bugs
def test_config_typo_is_rejected():
    """`cips_max: 5l` used to be silently ignored; now it must fail loudly."""
    with pytest.raises(Exception):
        load_settings({"cips_max": 5})


def test_classify_error_hints():
    assert classify_error("401 API_KEY_INVALID")[0] == "bad_key"
    assert classify_error("404 models/foo is not found")[0] == "bad_model"
    assert classify_error("429 RESOURCE_EXHAUSTED")[0] == "rate_limit"
    assert classify_error("empty structured answer")[0] == "empty"


def test_empty_model_answer_is_an_error_not_a_result(tmp_path, monkeypatch):
    """A model that 'succeeds' with all-empty fields must not be treated (or cached) as a valid answer."""
    cfg = load_settings({"provider": "mock", "use_cache": False, "call_retries": 1})
    client = LLMClient(cfg)
    import promo_finder.mock_llm as m
    monkeypatch.setattr(m, "mock_respond", lambda schema, system, user: EpisodeOut(summary="", key_facts=[], clips=[], moments=[]))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="empty"):
        client.invoke(EpisodeOut, "s", "u")
    assert any("error" in r["status"] for r in client.log)


def test_fatal_error_fails_fast_and_is_remembered(monkeypatch):
    cfg = load_settings({"provider": "mock", "use_cache": False, "call_retries": 5})
    client = LLMClient(cfg)
    calls = {"n": 0}
    import promo_finder.mock_llm as m

    def boom(schema, system, user):
        calls["n"] += 1
        raise RuntimeError("401 API_KEY_INVALID")
    monkeypatch.setattr(m, "mock_respond", boom)
    with pytest.raises(FatalLLMError):
        client.invoke(EpisodeOut, "s", "u")
    assert calls["n"] == 1                      # no 6 pointless retries
    with pytest.raises(FatalLLMError):
        client.invoke(EpisodeOut, "s", "u2")
    assert calls["n"] == 1                      # later calls do not even hit the API


def test_llm_ids_in_sloppy_format_still_resolve(show):
    s, cfg = show
    ep = s.episodes[0]
    seg = resolve(ep, "EP7_E1", "`ep07_e006`", 20, 60, cfg)   # ep7_e1 / backticks / upper-case
    assert seg.start_id == "ep07_e001" and seg.end_id == "ep07_e006"
