"""Storage interface for the role-scoped key-value log."""

from datetime import datetime
from typing import Protocol

from memmachine.kv_store.model import KvEntry, KvList


class KvStore(Protocol):
    """Append values under a key and read the newest ones back by exact key."""

    async def startup(self) -> None:
        """Create tables when the database does not already have them."""

    async def append(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        key: str,
        value: str,
        created_at: datetime | None = None,
    ) -> KvEntry:
        """Append one value. Existing values for the same key stay in place."""

    async def list_latest(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        key: str,
        limit: int | None,
    ) -> KvList:
        """Return newest values for one key. ``limit=None`` returns the full log."""

    async def delete_project(self, *, org_id: str, project_id: str) -> None:
        """Delete every key-value row in one project."""
