"""Offline heuristic stand-in for the LLM (LLM_PROVIDER=mock).

ONLY for testing that the plumbing works without an API key. It reads the same prompts a real
model would, but its "judgement" is a few string heuristics, so its scores mean nothing.
"""
from __future__ import annotations

import hashlib
import re

from .parser import LINE_RE
from .schemas import (Clip, ComboItem, ComboOut, EpisodeOut, FactCheck, FactVerdict, Moment, Recap, Score)
from .timeutil import parse_ts

_CARD = re.compile(r"^(?P<id>ep\d+_m\d+) \| (?P<ep>ep\d+) \| (?P<dur>\d+)s \| (?P<role>\w+) \|")


def _events(user: str) -> list[dict]:
    out = []
    for line in user.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            d = m.groupdict()
            d["s"], d["e"] = parse_ts(d["s"]), parse_ts(d["e"])
            d["rest"] = d.get("rest") or ""
            out.append(d)
    return out


def _window(evs: list[dict], i: int, target: float) -> int:
    j = i
    while j + 1 < len(evs) and evs[j]["e"] - evs[i]["s"] < target:
        j += 1
    return j


def mock_respond(schema, system: str, user: str):
    name = schema.__name__
    h = int(hashlib.md5(user.encode()).hexdigest(), 16)
    if name == "EpisodeOut":
        evs = _events(user)
        dl = [e for e in evs if e["kind"] == "DIALOGUE"]
        facts = [re.sub(r'^\S+ \(?.*?\)?: ', '', e["rest"])[:90] for e in dl[:4]] or ["unknown"]
        clips, moments = [], []
        step = max(1, len(evs) // 4)
        for k, i in enumerate(range(0, len(evs), step)):
            j = _window(evs, i, 35)
            clips.append(Clip(start_id=evs[i]["id"], end_id=evs[j]["id"], why="mock heuristic", strength=3 + (k % 3)))
        roles = ["hook", "setup", "payoff", "cliffhanger"]
        step = max(1, len(evs) // 6)
        for k, i in enumerate(range(0, len(evs), step)):
            j = _window(evs, i, 12)
            moments.append(Moment(start_id=evs[i]["id"], end_id=evs[j]["id"], role=roles[k % 4],
                                  thread="main conflict", summary=evs[i]["rest"][:80], strength=3))
        if moments:  # one deliberately bad item so the validation / retry path is exercised
            clips.append(Clip(start_id="ep99_e999", end_id=evs[-1]["id"], why="bad id", strength=1))
        return EpisodeOut(summary="mock summary", key_facts=facts, clips=clips[:5], moments=moments[:8])

    if name == "ComboOut":
        cards = [m.groupdict() for m in (_CARD.match(l.strip()) for l in user.splitlines()) if m]
        hooks = [c for c in cards if c["role"] == "hook"]
        cliffs = [c for c in cards if c["role"] == "cliffhanger"]
        combos = []
        for a, b in zip(hooks, cliffs[1:]):
            combos.append(ComboItem(moment_ids=[a["id"], b["id"]], why="mock pairing"))
        return ComboOut(combos=combos[:15])

    if name == "Score":
        evs = _events(user)
        dl = [e for e in evs if e["kind"] == "DIALOGUE"] or evs

        def text(e):
            m = re.search(r'"(.*)"', e["rest"])
            return m.group(1) if m else e["rest"]
        first, last = dl[0], dl[-1]
        hs = 5 if re.search(r"[!?]", text(first)) else 3
        cs = 4 if re.search(r"(\.\.\.|…|\?)$", text(last)) else 2
        return Score(hook_line_id=first["id"], hook_quote=text(first), hook_reason="mock", hook_score=hs,
                     cliff_line_id=last["id"], cliff_quote=text(last), cliff_reason="mock", cliff_score=cs)

    if name == "Recap":
        return Recap(who="unknown", what="unknown", stakes="unknown")

    if name == "FactCheck":
        n = len(re.findall(r"^\d+\. ", user, re.M))
        return FactCheck(verdicts=[FactVerdict(fact_no=i, verdict="correct" if (h >> i) & 1 else "missing")
                                   for i in range(1, n + 1)], notes="mock")
    raise ValueError(f"mock has no handler for {name}")
