"""Prompts are plain .md files in prompts/ so they can be edited (or swapped for another show)
without touching code. {placeholders} are filled from the settings."""
from __future__ import annotations

from .config import Settings, resolve_dir

EXTRACT, COMBINE, SCORE, RECAP, COMPARE, RETRY = "extract", "combine", "score", "recap", "compare", "retry"


class _Safe(dict):
    def __missing__(self, key):  # leave unknown {x} untouched instead of crashing
        return "{" + key + "}"


def render(name: str, cfg: Settings, **extra) -> str:
    text = (resolve_dir(cfg.prompts_dir) / f"{name}.md").read_text(encoding="utf-8")
    v = dict(
        audience=cfg.audience, format=cfg.format,
        genre_hint=(f"Genre: {cfg.genre_hint}." if cfg.genre_hint else ""),
        extra_instructions=cfg.extra_instructions,
        min_s=f"{cfg.min_s:g}", max_s=f"{cfg.max_s:g}",
        moment_min_s=f"{cfg.moment_min_s:g}", moment_max_s=f"{cfg.moment_max_s:g}",
        hook_window=f"{cfg.hook_window_s:g}",
        facts_min=cfg.facts_min, facts_max=cfg.facts_max,
        clips_min=cfg.clips_min, clips_max=cfg.clips_max,
        moments_min=cfg.moments_min, moments_max=cfg.moments_max,
        combo_min=cfg.combo_min_moments, combo_max=cfg.combo_max_moments, max_combos=cfg.max_combos,
    )
    return text.format_map(_Safe({**v, **extra})).strip()
