"""Records stored in the role-scoped library."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LibraryFile:
    """One named document."""

    name: str
    content: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class LibraryName:
    """A file name without its body."""

    name: str
    updated_at: datetime
