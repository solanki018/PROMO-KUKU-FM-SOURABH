"""Stage 0 (pure code, no LLM): screenplay text files -> Show / Episode / Event with stable ids.

Expected block format (what the Kuku input pack uses):

    EPISODE NUMBER: Episode 12
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    SCENE 01
    Scene heading: INT. COTTAGE - CONTINUOUS
    **CAMERA:**
    Start: (00:00.00) | End: (00:04.60)
    Medium shot of ...
    **JULIANA (Furious, Defiant):**
    Start: (00:05.70) | End: (00:08.80)
    "No! That watch was not in his coat!"

Labels, header names and extensions are config driven (ParserCfg), so a differently named
show / format only needs a config change. Anything that cannot be parsed becomes a warning,
never a crash.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

from .config import Settings
from .schemas import Episode, Event, Show
from .timeutil import fmt_ts, parse_ts

_TIME = r"(?:\d+:)?\d+:\d+(?:\.\d+)?"
RE_BLOCK = re.compile(r"^\*\*(?P<label>.+?)\*\*\s*:?\s*$")
RE_TIMES = re.compile(rf"Start:\s*\(?(?P<s>{_TIME})\)?\s*\|\s*End:\s*\(?(?P<e>{_TIME})\)?", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(?P<n>\d+)", re.I)
RE_END = re.compile(r"^---\s*SCRIPT END\s*---", re.I)
RE_PAREN = re.compile(r"\(([^)]*)\)")

# The exact shape of an event line in the LLM view (the mock LLM and tests import this)
LINE_RE = re.compile(r"^(?P<id>ep\d+_e\d+) \[(?P<s>[\d:.]+)-(?P<e>[\d:.]+)\] (?P<kind>[A-Z]+)(?: (?P<rest>.*))?$")


# ------------------------------------------------------------------ reading sources
def read_sources(source: str | Path | Iterable[tuple[str, bytes]], cfg: Settings) -> tuple[str, list[tuple[str, str]]]:
    """Return (show_name, [(filename, text), ...]) from a folder, a .zip, a single file,
    or an iterable of (name, bytes) pairs (what Streamlit uploads give us)."""
    exts = tuple(x.lower() for x in cfg.extensions)

    def ok(name: str) -> bool:
        base = Path(name).name
        return name.lower().endswith(exts) and not base.startswith((".", "__"))

    def dec(b: bytes) -> str:
        return b.decode("utf-8-sig", errors="replace")

    files: list[tuple[str, str]] = []
    name = "show"
    if isinstance(source, (str, Path)):
        p = Path(source)
        name = p.stem if p.is_file() else p.name
        if p.is_dir():
            files = [(f.name, dec(f.read_bytes())) for f in sorted(p.rglob("*")) if f.is_file() and ok(f.name)]
        elif p.suffix.lower() == ".zip":
            files = _from_zip(p.read_bytes(), ok, dec)
        elif p.is_file():
            files = [(p.name, dec(p.read_bytes()))]
        else:
            raise FileNotFoundError(f"input not found: {p}")
    else:
        for fname, data in source:
            if fname.lower().endswith(".zip"):
                name = Path(fname).stem
                files += _from_zip(data, ok, dec)
            elif ok(fname):
                files.append((Path(fname).name, dec(data)))
    return name, files


def _from_zip(data: bytes, ok, dec) -> list[tuple[str, str]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if not info.is_dir() and ok(info.filename):
                out.append((Path(info.filename).name, dec(z.read(info))))
    return out


# ------------------------------------------------------------------ parsing one file
def _classify(label: str, cfg: Settings):
    """label text -> (kind, speaker, tag, voice)"""
    label = label.strip().rstrip(":").strip()
    base = re.split(r"\s*\(", label, maxsplit=1)[0].strip()
    up = base.upper()
    if up in {x.upper() for x in cfg.camera_labels}:
        return "CAMERA", None, None, None
    if up in {x.upper() for x in cfg.action_labels}:
        return "ACTION", None, None, None
    if up in {x.upper() for x in cfg.sound_labels}:
        return "SOUND", None, None, None
    voice, tag = None, None
    voices = {v.upper() for v in cfg.voice_tags}
    for p in RE_PAREN.findall(label):
        p = p.strip()
        if p.upper() in voices:
            voice = p
        elif p:
            tag = p
    return "DIALOGUE", base, tag, voice


def _header(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, re.I | re.M)
    return m.group(1).strip() if m else ""


def parse_episode(text: str, filename: str, cfg: Settings, order: int, warnings: list[str]) -> Episode:
    head = _header(text, cfg.header_episode_number)
    m = re.search(r"(\d+)", head) or re.search(r"episode[-_ ]?0*(\d+)", filename, re.I)
    number = int(m.group(1)) if m else order
    ep_id = f"ep{number:02d}"

    raw: list[dict] = []
    scene = 0
    block: dict | None = None

    def flush():
        nonlocal block
        if block is not None:
            raw.append(block)
        block = None

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if RE_END.match(s):
            flush()
            break
        sm = RE_SCENE.match(s)
        if sm:
            flush()
            scene = int(sm.group("n"))
            continue
        bm = RE_BLOCK.match(s)
        if bm:
            flush()
            block = {"label": bm.group("label"), "scene": scene, "times": None, "text": []}
            continue
        if block is None:
            continue  # header / scene metadata lines
        tm = RE_TIMES.search(s)
        if tm and block["times"] is None:
            block["times"] = (parse_ts(tm.group("s")), parse_ts(tm.group("e")))
            continue
        block["text"].append(s)
    flush()

    events: list[Event] = []
    for b in raw:
        if b["times"] is None or not b["text"]:
            warnings.append(f"{ep_id}: skipped block '{b['label'][:30]}' (no timestamps or text)")
            continue
        kind, speaker, tag, voice = _classify(b["label"], cfg)
        txt = " ".join(b["text"]).strip()
        if kind == "DIALOGUE":
            txt = txt.strip('"“”').strip()
        start, end = b["times"]
        if end < start:
            warnings.append(f"{ep_id}: end < start in block '{b['label'][:30]}', swapped")
            start, end = end, start
        events.append(Event(id="", ep=ep_id, idx=0, scene=b["scene"], kind=kind, speaker=speaker,
                            tag=tag, voice=voice, text=txt, start=start, end=end))

    # file order is almost chronological; a stable sort by start makes it exact
    events.sort(key=lambda e: e.start)
    for i, e in enumerate(events):
        e.idx = i
        e.id = f"{ep_id}_e{i:03d}"
    if not events:
        warnings.append(f"{ep_id}: no events parsed from {filename}")

    return Episode(id=ep_id, number=number, logline=_header(text, cfg.header_logline),
                   continuity=_header(text, cfg.header_continuity), source_file=filename, events=events)


def load_show(source, cfg: Settings) -> Show:
    name, files = read_sources(source, cfg)
    warnings: list[str] = []
    episodes: list[Episode] = []
    for i, (fname, text) in enumerate(files, start=1):
        ep = parse_episode(text, fname, cfg, i, warnings)
        if ep.events:
            episodes.append(ep)
    episodes.sort(key=lambda e: e.number)
    seen: set[str] = set()
    for ep in episodes:  # duplicate episode numbers would collide on ids
        if ep.id in seen:
            warnings.append(f"duplicate episode id {ep.id} ({ep.source_file}); ids will collide")
        seen.add(ep.id)
    if not episodes:
        raise ValueError("No episodes could be parsed. Check parser settings in config.yaml.")
    return Show(name=name, episodes=episodes, warnings=warnings)


# ------------------------------------------------------------------ LLM view
def event_line(e: Event, cfg: Settings) -> str:
    """ep12_e020 [00:11.3-00:13.0] DIALOGUE LADY JULIANA (Desperate): "Gerald is waiting." """
    head = f"{e.id} [{fmt_ts(e.start, 1)}-{fmt_ts(e.end, 1)}] {e.kind}"
    text = e.text if len(e.text) <= cfg.line_max_chars else e.text[: cfg.line_max_chars - 1] + "…"
    if e.kind == "DIALOGUE":
        who = e.speaker or "?"
        extra = " ".join(f"({x})" for x in (e.voice, e.tag) if x)
        return f'{head} {who}{(" " + extra) if extra else ""}: "{text}"'
    return f"{head} {text}"


def episode_view(ep: Episode, cfg: Settings) -> str:
    return "\n".join(event_line(e, cfg) for e in ep.events if e.kind in cfg.include_kinds)
