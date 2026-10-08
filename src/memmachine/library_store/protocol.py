"""Storage interface for the role-scoped library."""

from typing import Protocol

from memmachine.library_store.model import LibraryFile, LibraryName


class LibraryStore(Protocol):
    """Store one document per id. The title is unique inside one role."""

    async def startup(self) -> None:
        """Create tables when the database does not already have them."""

    async def create(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        name: str,
        content: str,
        description: str,
        category: str,
    ) -> LibraryFile:
        """Insert a finished document. The title must be free in this scope."""

    async def update_content(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        content: str,
    ) -> LibraryFile:
        """Replace the body. The title stays the same."""

    async def update_category(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        category: str,
    ) -> LibraryFile:
        """Replace the category. The title and body stay the same."""

    async def rename(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        name: str,
    ) -> LibraryFile:
        """Replace the title. A taken title is left unchanged."""

    async def get(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
    ) -> LibraryFile:
        """Return one file."""

    async def delete(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
    ) -> None:
        """Delete one file. A missing id is already gone."""

    async def list_names(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
    ) -> list[LibraryName]:
        """Return ids, titles, and update times, newest first."""

    async def delete_project(self, *, org_id: str, project_id: str) -> None:
        """Delete every library file in one project."""
