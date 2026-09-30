"""Storage interface for the role-scoped library."""

from typing import Protocol

from memmachine.library_store.model import LibraryFile, LibraryName


class LibraryStore(Protocol):
    """Store one document per file name."""

    async def startup(self) -> None:
        """Create tables when the database does not already have them."""

    async def create(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        name: str,
        content: str,
    ) -> LibraryFile:
        """Create a file. The same name in this scope must not already exist."""

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
        """Replace the body of an existing file, and its name when requested."""

    async def get(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        name: str,
    ) -> LibraryFile:
        """Return one file."""

    async def delete(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        name: str,
    ) -> None:
        """Delete one file."""

    async def list_names(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
    ) -> list[LibraryName]:
        """Return file names and update times, newest first."""

    async def delete_project(self, *, org_id: str, project_id: str) -> None:
        """Delete every library file in one project."""
