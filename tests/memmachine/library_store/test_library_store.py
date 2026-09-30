"""Tests for the role-scoped library."""

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from memmachine.common.api.spec import (
    LIBRARY_CONTENT_MAX_LENGTH,
    LibraryFileSpec,
    LibraryNameSpec,
    UpdateLibrarySpec,
)
from memmachine.common.errors import LibraryNameExistsError, ResourceNotFoundError
from memmachine.library_store.memory_store import InMemoryLibraryStore
from memmachine.library_store.protocol import LibraryStore
from memmachine.library_store.sql_store import SqlLibraryStore


def test_file_spec_strips_name_and_keeps_content():
    spec = LibraryFileSpec(
        org_id="org",
        project_id="project",
        role_id="library",
        name="  服务范围  ",
        content="  weekday coverage  ",
    )
    assert spec.name == "服务范围"
    assert spec.content == "  weekday coverage  "


def test_file_spec_rejects_blank_or_oversized_content():
    with pytest.raises(ValidationError):
        LibraryFileSpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="   ",
        )
    with pytest.raises(ValidationError):
        LibraryFileSpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="x" * (LIBRARY_CONTENT_MAX_LENGTH + 1),
        )


def test_update_spec_strips_the_new_name():
    spec = UpdateLibrarySpec(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
        content="revised",
        new_name="  营业时间  ",
    )
    assert spec.new_name == "营业时间"


def test_name_spec_rejects_a_blank_name():
    with pytest.raises(ValidationError):
        LibraryNameSpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="   ",
        )


async def _create_scope(store: LibraryStore) -> None:
    await store.create(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
        content="original",
    )


@pytest.mark.asyncio
async def test_memory_store_rejects_a_duplicate_name():
    store = InMemoryLibraryStore()
    await _create_scope(store)
    with pytest.raises(LibraryNameExistsError):
        await _create_scope(store)
    file = await store.get(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
    )
    assert file.content == "original"


@pytest.mark.asyncio
async def test_memory_store_updates_and_lists_without_bodies():
    store = InMemoryLibraryStore()
    await _create_scope(store)
    await store.create(
        org_id="org",
        project_id="project",
        role_id="library",
        name="价格",
        content="rates",
    )
    updated = await store.update(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
        content="revised",
    )
    assert updated.content == "revised"
    assert updated.created_at <= updated.updated_at

    names = await store.list_names(
        org_id="org",
        project_id="project",
        role_id="library",
    )
    assert names[0].name == "服务范围"
    assert {item.name for item in names} == {"服务范围", "价格"}


@pytest.mark.asyncio
async def test_memory_store_renames_without_overwriting():
    store = InMemoryLibraryStore()
    await _create_scope(store)
    await store.create(
        org_id="org",
        project_id="project",
        role_id="library",
        name="价格",
        content="rates",
    )
    renamed = await store.update(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
        content="revised",
        new_name="营业时间",
    )
    assert renamed.name == "营业时间"
    assert renamed.content == "revised"
    with pytest.raises(ResourceNotFoundError):
        await store.get(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
        )
    with pytest.raises(LibraryNameExistsError):
        await store.update(
            org_id="org",
            project_id="project",
            role_id="library",
            name="营业时间",
            content="again",
            new_name="价格",
        )
    kept = await store.get(
        org_id="org",
        project_id="project",
        role_id="library",
        name="价格",
    )
    assert kept.content == "rates"


@pytest.mark.asyncio
async def test_memory_store_missing_file_is_not_found():
    store = InMemoryLibraryStore()
    with pytest.raises(ResourceNotFoundError):
        await store.update(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="revised",
        )
    with pytest.raises(ResourceNotFoundError):
        await store.get(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
        )
    with pytest.raises(ResourceNotFoundError):
        await store.delete(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
        )


@pytest.mark.asyncio
async def test_memory_store_isolates_roles():
    store = InMemoryLibraryStore()
    await _create_scope(store)
    other = await store.list_names(
        org_id="org",
        project_id="project",
        role_id="other",
    )
    assert other == []


async def _assert_project_delete(store: LibraryStore) -> None:
    await _create_scope(store)
    await store.create(
        org_id="org",
        project_id="project",
        role_id="other",
        name="价格",
        content="same project",
    )
    await store.create(
        org_id="org",
        project_id="other",
        role_id="library",
        name="服务范围",
        content="other project",
    )
    await store.delete_project(org_id="org", project_id="project")
    assert (
        await store.list_names(org_id="org", project_id="project", role_id="library")
        == []
    )
    assert (
        await store.list_names(org_id="org", project_id="project", role_id="other")
        == []
    )
    kept = await store.get(
        org_id="org",
        project_id="other",
        role_id="library",
        name="服务范围",
    )
    assert kept.content == "other project"


@pytest.mark.asyncio
async def test_memory_store_delete_project_removes_every_role():
    await _assert_project_delete(InMemoryLibraryStore())


@pytest.mark.asyncio
async def test_sql_store_enforces_unique_names_and_project_delete():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    store = SqlLibraryStore(engine)
    await store.startup()
    await _create_scope(store)
    with pytest.raises(LibraryNameExistsError):
        await _create_scope(store)
    updated = await store.update(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
        content="revised",
    )
    assert updated.content == "revised"
    renamed = await store.update(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
        content="revised",
        new_name="营业时间",
    )
    assert renamed.name == "营业时间"
    with pytest.raises(ResourceNotFoundError):
        await store.get(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
        )
    await store.delete(
        org_id="org",
        project_id="project",
        role_id="library",
        name="营业时间",
    )
    with pytest.raises(ResourceNotFoundError):
        await store.get(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
        )
    await _assert_project_delete(store)
    await engine.dispose()
