"""Postgres key-value log stored beside semantic memory."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from memmachine.kv_store.model import KvEntry, KvList

metadata = MetaData()

kv_entry_table = Table(
    "kv_entry",
    metadata,
    Column("id", String, primary_key=True),
    Column("org_id", String, nullable=False),
    Column("project_id", String, nullable=False),
    Column("role_id", String, nullable=False),
    Column("key", String, nullable=False),
    Column("value", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index(
        "idx_kv_entry_lookup",
        "org_id",
        "project_id",
        "role_id",
        "key",
        "created_at",
    ),
)


def _entry_from_mapping(mapping: object) -> KvEntry:
    row = dict(mapping)  # type: ignore[call-overload]
    return KvEntry(
        id=str(row["id"]),
        key=str(row["key"]),
        value=str(row["value"]),
        created_at=row["created_at"],
    )


class SqlKvStore:
    """Append-only ``kv_entry`` table in the semantic memory database."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Bind the store to the semantic memory SQL engine."""
        self._engine = engine
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    def _session(self) -> AsyncSession:
        return self._session_factory()

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

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
        stmt = insert(kv_entry_table).values(
            id=entry.id,
            org_id=org_id,
            project_id=project_id,
            role_id=role_id,
            key=entry.key,
            value=entry.value,
            created_at=entry.created_at,
        )
        async with self._session() as session:
            await session.execute(stmt)
            await session.commit()
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
        scope = (
            (kv_entry_table.c.org_id == org_id)
            & (kv_entry_table.c.project_id == project_id)
            & (kv_entry_table.c.role_id == role_id)
            & (kv_entry_table.c.key == key)
        )
        count_stmt = select(func.count()).select_from(kv_entry_table).where(scope)
        list_stmt = (
            select(kv_entry_table)
            .where(scope)
            .order_by(kv_entry_table.c.created_at.desc(), kv_entry_table.c.id.desc())
        )
        if limit is not None:
            list_stmt = list_stmt.limit(limit)

        async with self._session() as session:
            total = int((await session.execute(count_stmt)).scalar_one())
            rows = (await session.execute(list_stmt)).mappings().all()
        return KvList(
            key=key,
            entries=[_entry_from_mapping(row) for row in rows],
            total=total,
        )

    async def delete_project(self, *, org_id: str, project_id: str) -> None:
        stmt = delete(kv_entry_table).where(
            kv_entry_table.c.org_id == org_id,
            kv_entry_table.c.project_id == project_id,
        )
        async with self._session() as session:
            await session.execute(stmt)
            await session.commit()
