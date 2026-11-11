"""Pydantic models. Three groups:
1. parsed data (Event, Episode, Show)       - produced by code (Stage 0)
2. LLM structured outputs                   - what the model must return
3. pipeline objects (Segment, Candidate...) - produced by code after validation
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr

Kind = Literal["DIALOGUE", "ACTION", "CAMERA", "SOUND"]


# ---------------------------------------------------------------- 1. parsed data
def _canon(i: str) -> str:
    i = i.strip().strip("`'\"[]() ").lower()
    m = re.match(r"^ep0*(\d+)_e0*(\d+)$", i)
    return f"ep{int(m.group(1)):02d}_e{int(m.group(2)):03d}" if m else i


class Event(BaseModel):
    id: str
    ep: str
    idx: int
    scene: int = 0
    kind: Kind
    speaker: Optional[str] = None
    tag: Optional[str] = None       # emotion / delivery, e.g. "Furious, Defiant"
    voice: Optional[str] = None     # O.S. / V.O.
    text: str
    start: float
    end: float


class Episode(BaseModel):
    id: str
    number: int
    logline: str = ""               # NEVER sent to the evaluation calls (leak risk)
    continuity: str = ""
    source_file: str = ""
    events: list[Event] = []
    _index: Optional[dict[str, int]] = PrivateAttr(default=None)

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)

    def pos(self, event_id: str) -> Optional[int]:
        """Index of an event id. Tolerant of what LLMs write: ep12_e20 / EP12_E020 / `ep12_e020` / spaces."""
        if self._index is None:
            self._index = {_canon(e.id): i for i, e in enumerate(self.events)}
        return self._index.get(_canon(event_id))


class Show(BaseModel):
    name: str
    episodes: list[Episode]
    warnings: list[str] = []

    def episode(self, ep_id: str) -> Episode:
        for e in self.episodes:
            if e.id == ep_id:
                return e
        raise KeyError(ep_id)


# ---------------------------------------------------------------- 2. LLM outputs
class Moment(BaseModel):
    id: str = Field(default="", description="any short id; it is replaced by code")
    start_id: str = Field(description="first event id of the piece")
    end_id: str = Field(description="last event id of the piece")
    role: Literal["hook", "setup", "payoff", "cliffhanger"]
    thread: str = Field(description="short conflict tag, e.g. 'Juliana jilted at altar'")
    summary: str
    strength: int = Field(default=3, description="1-5, how good this piece is for a promo")


class Clip(BaseModel):
    start_id: str
    end_id: str
    why: str
    strength: int = Field(default=3, description="1-5, how good this promo is")


class EpisodeOut(BaseModel):
    summary: str
    key_facts: list[str]
    clips: list[Clip]
    moments: list[Moment]


class ComboItem(BaseModel):
    moment_ids: list[str]
    why: str


class ComboOut(BaseModel):
    combos: list[ComboItem]


class Score(BaseModel):
    # reasoning first, score last: the model must find evidence before it scores
    hook_line_id: str = Field(description="event id of the line that is the hook")
    hook_quote: str = Field(description="exact text of that line")
    hook_reason: str
    hook_score: int = Field(description="1-5")
    cliff_line_id: str = Field(description="event id of the line that is the cliffhanger")
    cliff_quote: str = Field(description="exact text of that line")
    cliff_reason: str
    cliff_score: int = Field(description="1-5")


class Recap(BaseModel):
    who: str
    what: str
    stakes: str


class FactVerdict(BaseModel):
    fact_no: int
    verdict: Literal["correct", "wrong", "missing"]


class FactCheck(BaseModel):
    verdicts: list[FactVerdict]
    notes: str = ""


# ---------------------------------------------------------------- 3. pipeline objects
class Segment(BaseModel):
    ep: str
    start_id: str
    end_id: str
    start: float
    end: float
    events: list[Event]
    repaired: bool = False

    @property
    def dur(self) -> float:
        return self.end - self.start


class Candidate(BaseModel):
    cid: str
    source: Literal["llm_clip", "rule_ending", "combo"]
    segments: list[Segment]
    why: str = ""
    strength: float = 3.0
    ref_ids: list[str] = []          # moment ids for combos

    @property
    def dur(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def episodes(self) -> list[str]:
        out: list[str] = []
        for s in self.segments:
            if s.ep not in out:
                out.append(s.ep)
        return out

    @property
    def cross_episode(self) -> bool:
        return len(self.episodes) > 1


class MomentSeg(BaseModel):
    id: str
    role: str
    thread: str
    summary: str
    strength: float
    segment: Segment


class EpisodeAnalysis(BaseModel):
    ep: str
    summary: str = ""
    key_facts: list[str] = []
    clips: list[Candidate] = []
    moments: list[MomentSeg] = []
    dropped: list[str] = []          # reasons, for the quality report
    repaired: int = 0
    error: Optional[str] = None


class Evaluation(BaseModel):
    score: Optional[Score] = None
    recap: Optional[Recap] = None
    check: Optional[FactCheck] = None
    hook_score: int = 0
    cliff_score: int = 0
    hook_verified: bool = False
    cliff_verified: bool = False
    match: Optional[float] = None
    missing_facts: list[str] = []
    error: Optional[str] = None


class Promo(BaseModel):
    rank: int = 0
    candidate: Candidate
    evaluation: Evaluation
    overall: float = 0.0
    standalone: float = 0.0
    flags: list[str] = []
    hook_text: str = ""
    cliff_text: str = ""
    rank_reason: str = ""
