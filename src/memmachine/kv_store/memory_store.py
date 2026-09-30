"""In-memory key-value log used by unit tests."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from memmachine.kv_store.model import KvEntry, KvList
from memmachine.kv_store.protocol import KvStore


class InMemoryKvStore(KvStore):
    """Process-local append log. Rows are not visible to semantic ingestion."""

    def __init__(self) -> None:
        """Create an empty in-memory log."""
        self._rows: list[tuple[str, str, str, KvEntry]] = []
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        return None

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
        entry = KvEntry(
            id=str(uuid4()),
            key=key,
            value=value,
            created_at=created_at or datetime.now(UTC),
        )
        async with self._lock:
            self._rows.append((org_id, project_id, role_id, entry))
        return entry

    async def list_latest(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        key: str,
        limit: int | None,
    ) -> KvList:
        async with self._lock:
            matched = [
                entry
                for row_org, row_project, row_role, entry in self._rows
                if row_org == org_id
                and row_project == project_id
                and row_role == role_id
                and entry.key == key
            ]
        newest_first = list(reversed(matched))
        page = newest_first if limit is None else newest_first[:limit]
        return KvList(key=key, entries=page, total=len(matched))

    async def delete_project(self, *, org_id: str, project_id: str) -> None:
        async with self._lock:
            self._rows = [
                row for row in self._rows if row[0] != org_id or row[1] != project_id
            ]
