"""In-memory library used by unit tests."""

import asyncio
from datetime import UTC, datetime

from memmachine.common.errors import LibraryNameExistsError, ResourceNotFoundError
from memmachine.library_store.model import LibraryFile, LibraryName
from memmachine.library_store.protocol import LibraryStore

_Scope = tuple[str, str, str, str]


class InMemoryLibraryStore(LibraryStore):
    """Process-local documents. Rows are not visible to semantic ingestion."""

    def __init__(self) -> None:
        """Create an empty library."""
        self._files: dict[_Scope, LibraryFile] = {}
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        return None

    def _scope(self, org_id: str, project_id: str, role_id: str, name: str) -> _Scope:
        return (org_id, project_id, role_id, name)

    async def create(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        name: str,
        content: str,
    ) -> LibraryFile:
        now = datetime.now(UTC)
        file = LibraryFile(
            name=name,
            content=content,
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            key = self._scope(org_id, project_id, role_id, name)
            if key in self._files:
                raise LibraryNameExistsError(name)
            self._files[key] = file
        return file

    async def update(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        name: str,
        content: str,
        new_name: str | None = None,
    ) -> LibraryFile:
        target = name if new_name is None or new_name == name else new_name
        async with self._lock:
            key = self._scope(org_id, project_id, role_id, name)
            current = self._files.get(key)
            if current is None:
                raise ResourceNotFoundError(f"Library file '{name}' not found")
            if target != name:
                renamed = self._scope(org_id, project_id, role_id, target)
                if renamed in self._files:
                    raise LibraryNameExistsError(target)
                del self._files[key]
                key = renamed
            updated = LibraryFile(
                name=target,
                content=content,
                created_at=current.created_at,
                updated_at=datetime.now(UTC),
            )
            self._files[key] = updated
        return updated

    async def get(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        name: str,
    ) -> LibraryFile:
        async with self._lock:
            file = self._files.get(self._scope(org_id, project_id, role_id, name))
        if file is None:
            raise ResourceNotFoundError(f"Library file '{name}' not found")
        return file

    async def delete(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        name: str,
    ) -> None:
        async with self._lock:
            key = self._scope(org_id, project_id, role_id, name)
            if key not in self._files:
                raise ResourceNotFoundError(f"Library file '{name}' not found")
            del self._files[key]

    async def list_names(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
    ) -> list[LibraryName]:
        async with self._lock:
            names = [
                LibraryName(name=file.name, updated_at=file.updated_at)
                for (row_org, row_project, row_role, _), file in self._files.items()
                if row_org == org_id
                and row_project == project_id
                and row_role == role_id
            ]
        return sorted(names, key=lambda item: item.updated_at, reverse=True)

    async def delete_project(self, *, org_id: str, project_id: str) -> None:
        async with self._lock:
            self._files = {
                key: file
                for key, file in self._files.items()
                if key[0] != org_id or key[1] != project_id
            }
