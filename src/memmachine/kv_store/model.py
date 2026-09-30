"""Records stored in the role-scoped key-value log."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class KvEntry:
    """One appended value for a key."""

    id: str
    key: str
    value: str
    created_at: datetime


@dataclass(frozen=True)
class KvList:
    """Newest-first page of values for one key."""

    key: str
    entries: list[KvEntry]
    total: int
