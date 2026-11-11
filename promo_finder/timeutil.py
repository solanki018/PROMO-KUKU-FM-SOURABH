from __future__ import annotations

import re

_TS = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def parse_ts(s: str) -> float:
    """'01:27.21' -> 87.21 ; '1:02:03.5' -> 3723.5. The decimals are fractions of a second."""
    m = _TS.match(s.strip().strip("()"))
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mnt, sec = m.groups()
    return round(int(h or 0) * 3600 + int(mnt) * 60 + float(sec), 3)


def fmt_ts(sec: float, decimals: int = 2) -> str:
    """87.21 -> '01:27.21' (hours folded into minutes: episodes are short)."""
    sec = max(sec, 0.0)
    m = int(sec // 60)
    s = sec - m * 60
    width = 3 + decimals if decimals else 2
    return f"{m:02d}:{s:0{width}.{decimals}f}"
