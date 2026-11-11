"""Stage 3 (pure code): turn LLM-chosen ids into real timestamps and validate them.

The LLM never writes a timestamp. It only picks event ids; code looks the times up, so a
hallucinated time is impossible and a hallucinated id is caught here.
"""
from __future__ import annotations

from typing import Optional

from .config import Settings
from .schemas import Candidate, Episode, Event, Segment


class ResolveError(Exception):
    pass


def _dur(evs: list[Event], s: int, e: int) -> float:
    return max(x.end for x in evs[s : e + 1]) - evs[s].start


def ends_clean(ev: Event, terminal_chars: str) -> bool:
    t = ev.text.rstrip()
    return bool(t) and t[-1] in terminal_chars


def _make(ep: Episode, s: int, e: int, repaired: bool = False) -> Segment:
    seg = ep.events[s : e + 1]
    return Segment(ep=ep.id, start_id=seg[0].id, end_id=seg[-1].id, start=seg[0].start,
                   end=max(x.end for x in seg), events=seg, repaired=repaired)


def fit(ep: Episode, s: int, e: int, lo: float, hi: float, terminal: str) -> Optional[tuple[int, int]]:
    """Deterministic repair: trim the end back / extend it forward so duration lands in [lo, hi],
    preferring an end that finishes a sentence. Returns None if impossible."""
    evs = ep.events
    d = _dur(evs, s, e)
    if lo <= d <= hi:
        return s, e
    if d > hi:
        cands = [k for k in range(e, s - 1, -1) if _dur(evs, s, k) <= hi]
        if not cands:
            return None
        for k in cands[:6]:
            if _dur(evs, s, k) >= lo and ends_clean(evs[k], terminal):
                return s, k
        k = cands[0]
        return (s, k) if _dur(evs, s, k) >= lo else None
    first_ok = None
    for k in range(e + 1, len(evs)):
        dk = _dur(evs, s, k)
        if dk > hi:
            break
        if dk >= lo:
            if ends_clean(evs[k], terminal):
                return s, k
            first_ok = first_ok if first_ok is not None else k
    return (s, first_ok) if first_ok is not None else None


def resolve(ep: Episode, start_id: str, end_id: str, lo: float, hi: float, cfg: Settings) -> Segment:
    """ids -> Segment with real times. Raises ResolveError with a message the LLM can act on."""
    s, e = ep.pos(start_id), ep.pos(end_id)
    if s is None or e is None:
        bad = start_id if s is None else end_id
        raise ResolveError(f"id '{bad}' does not exist in {ep.id}")
    if e < s:
        raise ResolveError(f"end_id {end_id} comes before start_id {start_id}")
    d = _dur(ep.events, s, e)
    if lo <= d <= hi:
        return _make(ep, s, e)
    if cfg.repair:
        fixed = fit(ep, s, e, lo, hi, cfg.terminal_chars)
        if fixed:
            return _make(ep, fixed[0], fixed[1], repaired=True)
    raise ResolveError(f"{start_id}..{end_id} lasts {d:.1f}s, allowed {lo:.0f}-{hi:.0f}s")


def rule_ending(ep: Episode, cfg: Settings) -> Optional[Segment]:
    """Backup clip: the last ~N seconds of the episode (most episodes end on a cliffhanger)."""
    if not ep.events:
        return None
    target = ep.duration - cfg.rule_ending_target_s
    s = next((i for i, x in enumerate(ep.events) if x.start >= target), None)
    if s is None:
        return None
    last = len(ep.events) - 1
    fixed = fit(ep, s, last, cfg.min_s, cfg.max_s, cfg.terminal_chars)
    return _make(ep, *fixed, repaired=False) if fixed else None


def check_combo(segs: list[Segment], cfg: Settings) -> Optional[str]:
    """None if OK, else the reason it is invalid."""
    total = sum(s.dur for s in segs)
    if not (cfg.min_s <= total <= cfg.max_s):
        return f"total {total:.1f}s outside {cfg.min_s:.0f}-{cfg.max_s:.0f}s"
    last_end: dict[str, float] = {}
    for s in segs:  # same-episode pieces must be chronological and non-overlapping
        if s.ep in last_end and s.start < last_end[s.ep] - 1e-6:
            return f"pieces from {s.ep} are out of time order or overlap"
        last_end[s.ep] = s.end
    return None


def candidate_text(c: Candidate, event_line) -> str:
    """What the evaluation calls see: only the clip text, pieces separated by --- CUT ---."""
    parts = ["\n".join(event_line(e) for e in s.events) for s in c.segments]
    return "\n--- CUT ---\n".join(parts)
