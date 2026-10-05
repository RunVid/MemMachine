"""Records stored in the role-scoped library."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LibraryFile:
    """One document. ``id`` stays fixed when the title changes."""

    id: str
    name: str
    content: str
    description: str
    always_loaded: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class LibraryName:
    """A file id, title, and one-line summary, without the body."""

    id: str
    name: str
    description: str
    updated_at: datetime
