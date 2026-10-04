"""Tests for the role-scoped library."""

import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from memmachine.common.api.spec import (
    LIBRARY_CONTENT_MAX_LENGTH,
    CreateLibrarySpec,
    LibraryIdSpec,
)
from memmachine.common.errors import (
    LibraryNameExistsError,
    LibraryTimeoutError,
    ResourceNotFoundError,
)
from memmachine.library_store.memory_store import InMemoryLibraryStore
from memmachine.library_store.naming import (
    LIBRARY_TITLE_MAX_CONCURRENT,
    choose_available_name,
    resolve_title,
    sample_for_title,
)
from memmachine.library_store.protocol import LibraryStore
from memmachine.library_store.sql_store import SqlLibraryStore


def test_create_spec_keeps_markdown_and_rejects_blank_content():
    spec = CreateLibrarySpec(
        org_id="org",
        project_id="project",
        role_id="library",
        content="  # 服务范围\n\n工作日覆盖前厅。  ",
        timeout=30,
    )
    assert spec.content.startswith("  #")
    assert spec.name is None
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            content="   ",
            timeout=30,
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            content="x" * (LIBRARY_CONTENT_MAX_LENGTH + 1),
            timeout=30,
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            content="正文",
            timeout=0,
        )


def test_id_spec_requires_a_uuid():
    LibraryIdSpec(
        org_id="org",
        project_id="project",
        role_id="library",
        id=str(uuid4()),
    )
    with pytest.raises(ValidationError):
        LibraryIdSpec(
            org_id="org",
            project_id="project",
            role_id="library",
            id="服务范围",
        )


def test_title_sample_keeps_edges_and_headings():
    body = "# 开头\n" + ("甲" * 5000) + "\n# 中间\n" + ("乙" * 5000) + "结尾标记"
    sample = sample_for_title(body)
    assert "# 开头" in sample
    assert "# 中间" in sample
    assert "结尾标记" in sample
    assert len(sample) < len(body)


def test_suggested_name_avoids_names_already_in_scope():
    assert choose_available_name("  服务范围  ", set()) == "服务范围"
    assert choose_available_name("服务范围", {"服务范围"}) == "服务范围 2"


@pytest.mark.asyncio
async def test_title_generation_stops_at_the_concurrency_limit():
    started = 0
    release = asyncio.Event()

    async def hold(_content: str, _existing: set[str]) -> str:
        nonlocal started
        started += 1
        await release.wait()
        return "服务范围"

    tasks = [
        asyncio.create_task(
            resolve_title(
                name=None,
                content="正文",
                existing=set(),
                seconds=2,
                suggest=hold,
            )
        )
        for _ in range(LIBRARY_TITLE_MAX_CONCURRENT + 1)
    ]
    await asyncio.sleep(0.05)
    assert started == LIBRARY_TITLE_MAX_CONCURRENT
    release.set()
    await asyncio.gather(*tasks)
    assert started == LIBRARY_TITLE_MAX_CONCURRENT + 1


@pytest.mark.asyncio
async def test_resolve_title_times_out_before_a_title_is_accepted():
    async def slow(_content: str, _existing: set[str]) -> str:
        await asyncio.sleep(0.05)
        return "服务范围"

    with pytest.raises(LibraryTimeoutError):
        await resolve_title(
            name=None,
            content="正文",
            existing=set(),
            seconds=0.01,
            suggest=slow,
        )


async def _create(store: LibraryStore, name: str, content: str = "正文") -> str:
    created = await store.create(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=str(uuid4()),
        name=name,
        content=content,
    )
    return created.id


@pytest.mark.asyncio
async def test_memory_store_keeps_id_when_content_and_title_change():
    store = InMemoryLibraryStore()
    file_id = await _create(store, "服务范围", "original")
    updated = await store.update_content(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
        content="# revised\n\nbody",
    )
    assert updated.id == file_id
    assert updated.name == "服务范围"
    assert updated.content.startswith("# revised")
    renamed = await store.rename(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
        name="营业时间",
    )
    assert renamed.id == file_id
    assert renamed.content.startswith("# revised")
    listed = await store.list_names(
        org_id="org", project_id="project", role_id="library"
    )
    assert [(item.id, item.name) for item in listed] == [(file_id, "营业时间")]


@pytest.mark.asyncio
async def test_memory_store_rename_conflict_keeps_the_old_title():
    store = InMemoryLibraryStore()
    kept_id = await _create(store, "价格", "rates")
    file_id = await _create(store, "服务范围")
    with pytest.raises(LibraryNameExistsError):
        await store.rename(
            org_id="org",
            project_id="project",
            role_id="library",
            file_id=file_id,
            name="价格",
        )
    current = await store.get(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
    )
    assert current.name == "服务范围"
    other = await store.get(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=kept_id,
    )
    assert other.content == "rates"


@pytest.mark.asyncio
async def test_memory_store_missing_id_and_repeat_delete():
    store = InMemoryLibraryStore()
    missing = str(uuid4())
    with pytest.raises(ResourceNotFoundError):
        await store.get(
            org_id="org",
            project_id="project",
            role_id="library",
            file_id=missing,
        )
    await store.delete(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=missing,
    )
    file_id = await _create(store, "服务范围")
    await store.delete(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
    )
    await store.delete(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
    )


@pytest.mark.asyncio
async def test_sql_store_uses_id_and_rejects_a_duplicate_title():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    store = SqlLibraryStore(engine)
    await store.startup()
    file_id = await _create(store, "服务范围", "original")
    with pytest.raises(LibraryNameExistsError):
        await _create(store, "服务范围", "other")
    updated = await store.update_content(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
        content="revised",
    )
    assert updated.id == file_id
    assert updated.name == "服务范围"
    renamed = await store.rename(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
        name="营业时间",
    )
    assert renamed.id == file_id
    await store.delete(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
    )
    await store.delete(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
    )
    with pytest.raises(ResourceNotFoundError):
        await store.get(
            org_id="org",
            project_id="project",
            role_id="library",
            file_id=file_id,
        )
    await engine.dispose()
