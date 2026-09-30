"""Tests for the role-scoped key-value log."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from memmachine.common.api.spec import AppendKvSpec, GetKvSpec
from memmachine.kv_store.memory_store import InMemoryKvStore
from memmachine.kv_store.sql_store import SqlKvStore


def test_append_spec_keeps_phone_key_text():
    spec = AppendKvSpec(
        org_id="org",
        project_id="project",
        role_id="call_log",
        key="  calls from +1510123123  ",
        value="  2026-09-07, caller claimed to be Dr Smith's office.  ",
    )
    assert spec.key == "calls from +1510123123"
    assert spec.value.startswith("2026-09-07")


def test_get_spec_defaults_to_prepared_context_limit():
    spec = GetKvSpec(
        org_id="org",
        project_id="project",
        role_id="call_log",
        key="calls from +1510123123",
    )
    assert spec.limit == 5


def test_get_spec_rejects_a_non_positive_limit():
    with pytest.raises(ValidationError):
        GetKvSpec(
            org_id="org",
            project_id="project",
            role_id="call_log",
            key="calls from +1510123123",
            limit=0,
        )


async def _append_three(store: InMemoryKvStore | SqlKvStore) -> None:
    start = datetime(2026, 9, 7, tzinfo=UTC)
    for offset, value in enumerate(["first", "second", "third"]):
        await store.append(
            org_id="org",
            project_id="project",
            role_id="call_log",
            key="calls from +1510123123",
            value=value,
            created_at=start + timedelta(minutes=offset),
        )


@pytest.mark.asyncio
async def test_memory_store_returns_newest_for_the_same_role_and_key():
    store = InMemoryKvStore()
    await _append_three(store)
    await store.append(
        org_id="org",
        project_id="project",
        role_id="other_role",
        key="calls from +1510123123",
        value="other role",
        created_at=datetime(2026, 9, 8, tzinfo=UTC),
    )

    page = await store.list_latest(
        org_id="org",
        project_id="project",
        role_id="call_log",
        key="calls from +1510123123",
        limit=2,
    )
    assert page.total == 3
    assert [entry.value for entry in page.entries] == ["third", "second"]


@pytest.mark.asyncio
async def test_memory_store_null_limit_returns_the_full_log():
    store = InMemoryKvStore()
    await _append_three(store)
    page = await store.list_latest(
        org_id="org",
        project_id="project",
        role_id="call_log",
        key="calls from +1510123123",
        limit=None,
    )
    assert page.total == 3
    assert [entry.value for entry in page.entries] == ["third", "second", "first"]


@pytest.mark.asyncio
async def test_sql_store_appends_in_sqlite_without_semantic_tables():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    store = SqlKvStore(engine)
    await store.startup()
    await _append_three(store)

    page = await store.list_latest(
        org_id="org",
        project_id="project",
        role_id="call_log",
        key="calls from +1510123123",
        limit=2,
    )
    assert page.total == 3
    assert [entry.value for entry in page.entries] == ["third", "second"]
    await engine.dispose()


async def _assert_project_delete_keeps_other_projects(
    store: InMemoryKvStore | SqlKvStore,
) -> None:
    await store.append(
        org_id="org",
        project_id="project",
        role_id="call_log",
        key="calls from +1510123123",
        value="kept until delete",
    )
    await store.append(
        org_id="org",
        project_id="project",
        role_id="other_role",
        key="calls from +1510123999",
        value="same project",
    )
    await store.append(
        org_id="org",
        project_id="other",
        role_id="call_log",
        key="calls from +1510123123",
        value="other project",
    )

    await store.delete_project(org_id="org", project_id="project")

    deleted = await store.list_latest(
        org_id="org",
        project_id="project",
        role_id="call_log",
        key="calls from +1510123123",
        limit=None,
    )
    other_role = await store.list_latest(
        org_id="org",
        project_id="project",
        role_id="other_role",
        key="calls from +1510123999",
        limit=None,
    )
    kept = await store.list_latest(
        org_id="org",
        project_id="other",
        role_id="call_log",
        key="calls from +1510123123",
        limit=None,
    )
    assert deleted.total == 0
    assert other_role.total == 0
    assert [entry.value for entry in kept.entries] == ["other project"]


@pytest.mark.asyncio
async def test_memory_store_delete_project_removes_every_role():
    await _assert_project_delete_keeps_other_projects(InMemoryKvStore())


@pytest.mark.asyncio
async def test_sql_store_delete_project_removes_every_role():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    store = SqlKvStore(engine)
    await store.startup()
    await _assert_project_delete_keeps_other_projects(store)
    await engine.dispose()
