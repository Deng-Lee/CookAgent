from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_LOCK_LOG = Path("logs/parent_locking.log")
DEFAULT_EVIDENCE_LOG = Path("logs/evidence_driven.log")
DEFAULT_SESSION_DIR = Path("logs/sessions")


@dataclass
class ChunkHit:
    score: float
    rank: int
    text: str
    metadata: Dict


@dataclass
class ParentHit:
    parent_id: str
    rrf_sum: float = 0.0
    max_chunk_score: float = 0.0
    coverage: int = 0  # hit count
    total_chunks: int = 0
    coverage_ratio: float = 0.0
    rrf_sum_norm: float = 0.0
    max_chunk_norm: float = 0.0
    overall_score: float = 0.0
    low_evidence: bool = False
    good_but_ambiguous: bool = False
    auto_recommend: bool = False
    hits: List[ChunkHit] = field(default_factory=list)
    parent_doc: Optional[str] = None


@dataclass
class ParentLock:
    status: str  # locked | pending | none
    parent_id: Optional[str]
    lock_reason: Optional[str]
    lock_score: Optional[float]
    locked_at_turn: Optional[int]
    pending_reason: Optional[str]


@dataclass
class RetrievalState:
    parents: List[ParentHit]
    parent_lock: ParentLock
    evidence_set: Optional[Dict]
