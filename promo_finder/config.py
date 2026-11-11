"""All tunables live here (and in config.yaml). Nothing about a specific show is hardcoded.

FLAT settings: every knob is a top-level attribute  (cfg.min_s, cfg.workers, cfg.provider ...).
Override order:  config.yaml  <  env (LLM_PROVIDER / LLM_MODEL / LLM_BASE_URL)  <  overrides dict.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # extra="forbid": a typo in config.yaml (e.g. `cips_max`) now fails loudly instead of being ignored
    model_config = ConfigDict(extra="forbid")

    # ---- LLM
    provider: str = "google_genai"      # any provider supported by langchain init_chat_model, or "mock"
    model: str = "gemini-2.5-flash"
    base_url: str = ""                  # only for OpenAI-compatible endpoints (OpenRouter, Together, local ...)
    temperature: float = 0.0
    max_retries: int = 2                # SDK-level retries (network / 429)
    call_retries: int = 2               # app-level retries (empty / invalid structured output)
    workers: int = 4                    # parallel calls
    rpm: int = 0                        # requests per minute cap, 0 = off (set ~10 for free tiers)
    use_cache: bool = True
    cache_dir: str = ".cache"

    # ---- per-show hints injected into the prompts
    audience: str = "US"
    format: str = "vertical microdrama, 1-2 minute episodes"
    genre_hint: str = ""
    extra_instructions: str = ""

    # ---- parser
    extensions: list[str] = [".txt", ".md"]
    camera_labels: list[str] = ["CAMERA", "SHOT"]
    action_labels: list[str] = ["ACTION"]
    sound_labels: list[str] = ["SOUND FX", "SOUND", "SFX", "MUSIC"]
    voice_tags: list[str] = ["O.S.", "V.O.", "O.C.", "CONT'D", "CONT’D", "OFF-SCREEN"]
    header_episode_number: str = "EPISODE NUMBER"
    header_logline: str = "EPISODE LOGLINE"
    header_continuity: str = "CONTINUITY NOTES"
    line_max_chars: int = 220
    include_kinds: list[str] = ["DIALOGUE", "ACTION", "CAMERA", "SOUND"]

    # ---- promo rules
    min_s: float = 20
    max_s: float = 60
    moment_min_s: float = 5
    moment_max_s: float = 25
    hook_window_s: float = 3
    facts_min: int = 4
    facts_max: int = 6
    clips_min: int = 3
    clips_max: int = 5
    moments_min: int = 4
    moments_max: int = 8
    combo_min_moments: int = 2
    combo_max_moments: int = 3
    max_combos: int = 15
    combine_episodes_per_call: int = 50
    repair: bool = True                 # deterministic trim / extend before giving up on a clip
    retry_invalid: bool = True
    rule_ending: bool = True            # add last-N-seconds clip of every episode
    rule_ending_target_s: float = 35
    terminal_chars: str = ".?!…\"'”’"

    # ---- evaluation
    top_n: int = 100
    min_per_episode: int = 2
    always_include_combos: bool = True
    w_standalone: float = 0.35
    w_hook: float = 0.35
    w_cliff: float = 0.30
    needs_context_below: float = 0.5
    weak_hook_max: int = 2
    resolved_ending_max: int = 2
    unverified_cap: int = 3
    neutral_match: float = 0.5

    # ---- ranking
    overlap_max: float = 0.5
    per_episode_max: int = 3

    prompts_dir: str = "prompts"
    output_dir: str = "output"


def load_settings(overrides: dict[str, Any] | None = None, path: str | Path | None = None) -> Settings:
    """config.yaml  <  env  <  overrides, e.g. load_settings({"max_s": 45})."""
    path = Path(path) if path else ROOT / "config.yaml"
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for env, key in (("LLM_PROVIDER", "provider"), ("LLM_MODEL", "model"), ("LLM_BASE_URL", "base_url")):
        if os.getenv(env):
            data[key] = os.environ[env]
    for k, v in (overrides or {}).items():
        if v is not None:
            data[k] = v
    return Settings.model_validate(data)


def resolve_dir(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q
