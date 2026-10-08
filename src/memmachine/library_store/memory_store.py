"""In-memory library used by unit tests."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

from memmachine.common.api.spec import LIBRARY_MAX_FILES_PER_SCOPE
from memmachine.common.errors import (
    LibraryFileLimitError,
    LibraryNameExistsError,
    ResourceNotFoundError,
)
from memmachine.library_store.model import LibraryFile, LibraryName
from memmachine.library_store.protocol import LibraryStore

_Key = tuple[str, str, str, str]


class InMemoryLibraryStore(LibraryStore):
    """Process-local documents. Rows are not visible to semantic ingestion."""

    def __init__(self) -> None:
        """Create an empty library."""
        self._files: dict[_Key, LibraryFile] = {}
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        return None

    def _key(self, org_id: str, project_id: str, role_id: str, file_id: str) -> _Key:
        return (org_id, project_id, role_id, file_id)

    def _find(
        self, org_id: str, project_id: str, role_id: str, file_id: str
    ) -> _Key | None:
        key = self._key(org_id, project_id, role_id, file_id)
        if key in self._files:
            return key
        return None

    async def create(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        name: str,
        content: str,
        description: str = "",
    ) -> LibraryFile:
        now = datetime.now(UTC)
        file = LibraryFile(
            id=file_id,
            name=name,
            content=content,
            description=description,
            always_loaded=False,
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            in_scope = sum(
                1
                for (row_org, row_project, row_role, _) in self._files
                if row_org == org_id
                and row_project == project_id
                and row_role == role_id
            )
            if in_scope >= LIBRARY_MAX_FILES_PER_SCOPE:
                raise LibraryFileLimitError(LIBRARY_MAX_FILES_PER_SCOPE)
            self._reject_taken_name(org_id, project_id, role_id, name, file_id)
            self._files[self._key(org_id, project_id, role_id, file_id)] = file
        return file

    async def update_content(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        content: str,
    ) -> LibraryFile:
        async with self._lock:
            key = self._find(org_id, project_id, role_id, file_id)
            if key is None:
                raise ResourceNotFoundError(f"Library file '{file_id}' not found")
            current = self._files[key]
            updated = replace(
                current,
                content=content,
                updated_at=datetime.now(UTC),
            )
            self._files[key] = updated
        return updated

    async def rename(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        name: str,
    ) -> LibraryFile:
        async with self._lock:
            key = self._find(org_id, project_id, role_id, file_id)
            if key is None:
                raise ResourceNotFoundError(f"Library file '{file_id}' not found")
            current = self._files[key]
            if current.name != name:
                self._reject_taken_name(org_id, project_id, role_id, name, file_id)
            updated = replace(
                current,
                name=name,
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
        file_id: str,
    ) -> LibraryFile:
        async with self._lock:
            key = self._find(org_id, project_id, role_id, file_id)
            file = self._files.get(key) if key is not None else None
        if file is None:
            raise ResourceNotFoundError(f"Library file '{file_id}' not found")
        return file

    async def delete(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
    ) -> None:
        async with self._lock:
            key = self._find(org_id, project_id, role_id, file_id)
            if key is not None:
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
                LibraryName(
                    id=file.id,
                    name=file.name,
                    description=file.description,
                    updated_at=file.updated_at,
                )
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

    def _reject_taken_name(
        self,
        org_id: str,
        project_id: str,
        role_id: str,
        name: str,
        file_id: str,
    ) -> None:
        for (row_org, row_project, row_role, row_id), file in self._files.items():
            same_scope = (
                row_org == org_id and row_project == project_id and row_role == role_id
            )
            if same_scope and file.name == name and row_id != file_id:
                raise LibraryNameExistsError(name)
