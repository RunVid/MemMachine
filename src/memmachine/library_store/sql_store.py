"""Postgres library stored beside semantic memory."""

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    ColumnElement,
    DateTime,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    delete,
    false,
    func,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from memmachine.common.api.spec import LIBRARY_MAX_FILES_PER_SCOPE
from memmachine.common.errors import (
    LibraryFileLimitError,
    LibraryNameExistsError,
    ResourceNotFoundError,
)
from memmachine.library_store.model import LibraryFile, LibraryName


def _result_rowcount(result: object) -> int:
    rowcount = getattr(result, "rowcount", 0)
    if rowcount is None:
        return 0
    return int(rowcount)


metadata = MetaData()

library_entry_table = Table(
    "library_entry",
    metadata,
    Column("id", String, primary_key=True),
    Column("org_id", String, nullable=False),
    Column("project_id", String, nullable=False),
    Column("role_id", String, nullable=False),
    Column("name", String, nullable=False),
    Column("content", String, nullable=False),
    Column("description", String, nullable=False, server_default=""),
    Column("always_loaded", Boolean, nullable=False, server_default=false()),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "org_id",
        "project_id",
        "role_id",
        "name",
        name="uq_library_entry_name",
    ),
)


def _file_from_mapping(mapping: object) -> LibraryFile:
    row = dict(mapping)  # type: ignore[call-overload]
    return LibraryFile(
        id=str(row["id"]),
        name=str(row["name"]),
        content=str(row["content"]),
        description=str(row.get("description", "") or ""),
        always_loaded=bool(row.get("always_loaded", False)),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _missing(file_id: str) -> ResourceNotFoundError:
    return ResourceNotFoundError(f"Library file '{file_id}' not found")


class SqlLibraryStore:
    """One document per id in the ``library_entry`` table."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Bind the store to the semantic memory SQL engine."""
        self._engine = engine
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    def _session(self) -> AsyncSession:
        return self._session_factory()

    def _by_id(
        self,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
    ) -> ColumnElement[bool]:
        table = library_entry_table.c
        return (
            (table.org_id == org_id)
            & (table.project_id == project_id)
            & (table.role_id == role_id)
            & (table.id == file_id)
        )

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

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
        table = library_entry_table.c
        count_stmt = (
            select(func.count())
            .select_from(library_entry_table)
            .where(
                (table.org_id == org_id)
                & (table.project_id == project_id)
                & (table.role_id == role_id)
            )
        )
        stmt = library_entry_table.insert().values(
            id=file_id,
            org_id=org_id,
            project_id=project_id,
            role_id=role_id,
            name=name,
            content=content,
            description=description,
            always_loaded=False,
            created_at=now,
            updated_at=now,
        )
        async with self._session() as session:
            count = int((await session.execute(count_stmt)).scalar_one())
            if count >= LIBRARY_MAX_FILES_PER_SCOPE:
                raise LibraryFileLimitError(LIBRARY_MAX_FILES_PER_SCOPE)
            try:
                await session.execute(stmt)
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise LibraryNameExistsError(name) from error
        return LibraryFile(
            id=file_id,
            name=name,
            content=content,
            description=description,
            always_loaded=False,
            created_at=now,
            updated_at=now,
        )

    async def update_content(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        content: str,
    ) -> LibraryFile:
        now = datetime.now(UTC)
        stmt = (
            update(library_entry_table)
            .where(self._by_id(org_id, project_id, role_id, file_id))
            .values(content=content, updated_at=now)
        )
        async with self._session() as session:
            result = await session.execute(stmt)
            if _result_rowcount(result) == 0:
                await session.rollback()
                raise _missing(file_id)
            await session.commit()
        return await self.get(
            org_id=org_id,
            project_id=project_id,
            role_id=role_id,
            file_id=file_id,
        )

    async def rename(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
        name: str,
    ) -> LibraryFile:
        now = datetime.now(UTC)
        stmt = (
            update(library_entry_table)
            .where(self._by_id(org_id, project_id, role_id, file_id))
            .values(name=name, updated_at=now)
        )
        async with self._session() as session:
            try:
                result = await session.execute(stmt)
                if _result_rowcount(result) == 0:
                    await session.rollback()
                    raise _missing(file_id)
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise LibraryNameExistsError(name) from error
        return await self.get(
            org_id=org_id,
            project_id=project_id,
            role_id=role_id,
            file_id=file_id,
        )

    async def get(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
    ) -> LibraryFile:
        stmt = select(library_entry_table).where(
            self._by_id(org_id, project_id, role_id, file_id)
        )
        async with self._session() as session:
            row = (await session.execute(stmt)).mappings().first()
        if row is None:
            raise _missing(file_id)
        return _file_from_mapping(row)

    async def delete(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
        file_id: str,
    ) -> None:
        stmt = delete(library_entry_table).where(
            self._by_id(org_id, project_id, role_id, file_id)
        )
        async with self._session() as session:
            await session.execute(stmt)
            await session.commit()

    async def list_names(
        self,
        *,
        org_id: str,
        project_id: str,
        role_id: str,
    ) -> list[LibraryName]:
        table = library_entry_table.c
        stmt = (
            select(table.id, table.name, table.description, table.updated_at)
            .where(
                (table.org_id == org_id)
                & (table.project_id == project_id)
                & (table.role_id == role_id)
            )
            .order_by(table.updated_at.desc(), table.name.asc())
        )
        async with self._session() as session:
            rows = (await session.execute(stmt)).mappings().all()
        return [
            LibraryName(
                id=str(row["id"]),
                name=str(row["name"]),
                description=str(row.get("description", "") or ""),
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def delete_project(self, *, org_id: str, project_id: str) -> None:
        table = library_entry_table.c
        stmt = delete(library_entry_table).where(
            (table.org_id == org_id) & (table.project_id == project_id)
        )
        async with self._session() as session:
            await session.execute(stmt)
            await session.commit()
