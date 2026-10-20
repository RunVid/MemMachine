"""Tests for the role-scoped library."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from memmachine.common.api.spec import (
    LIBRARY_CONTENT_MAX_LENGTH,
    LIBRARY_DESCRIPTION_MAX_LENGTH,
    LIBRARY_MAX_FILES_PER_CATEGORY,
    LIBRARY_NAME_MAX_LENGTH,
    CreateLibrarySpec,
    LibraryCategory,
    LibraryIdSpec,
    UpdateLibrarySpec,
)
from memmachine.common.errors import (
    LibraryFileLimitError,
    LibraryNameExistsError,
    ResourceNotFoundError,
)
from memmachine.library_store.memory_store import InMemoryLibraryStore
from memmachine.library_store.protocol import LibraryStore
from memmachine.library_store.sql_store import SqlLibraryStore

_SCOPE = {"org_id": "org", "project_id": "project", "role_id": "library"}


def test_create_spec_requires_name_and_validates_fields():
    spec = CreateLibrarySpec(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
        content="  # 服务范围\n\n工作日覆盖前厅。  ",
        description="Lobby coverage",
        category="business",
    )
    assert spec.content.startswith("  #")
    assert spec.category == LibraryCategory.BUSINESS
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="   ",
            content="正文",
            description="summary",
            category="personal",
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="x" * (LIBRARY_NAME_MAX_LENGTH + 1),
            content="正文",
            description="summary",
            category="personal",
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="   ",
            description="summary",
            category="personal",
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="正文",
            description="   ",
            category="personal",
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="x" * (LIBRARY_CONTENT_MAX_LENGTH + 1),
            description="summary",
            category="personal",
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="正文",
            description="line one\nline two",
            category="personal",
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="正文",
            description="x" * (LIBRARY_DESCRIPTION_MAX_LENGTH + 1),
            category="personal",
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="正文",
            description="summary",
        )
    with pytest.raises(ValidationError):
        CreateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            name="服务范围",
            content="正文",
            description="summary",
            category="work",
        )
    spec = CreateLibrarySpec(
        org_id="org",
        project_id="project",
        role_id="library",
        name="服务范围",
        content="正文",
        description="summary",
        category=" PERSONAL ",
    )
    assert spec.category == LibraryCategory.PERSONAL


def test_update_spec_overwrites_title_body_and_summary():
    file_id = str(uuid4())
    spec = UpdateLibrarySpec(
        org_id="org",
        project_id="project",
        role_id="library",
        id=file_id,
        name=" 营业时间 ",
        content="  # revised\n\nbody  ",
        description="  Lobby coverage  ",
    )
    assert spec.name == "营业时间"
    assert spec.content.startswith("  #")
    assert spec.description == "Lobby coverage"
    dumped = spec.model_dump()
    assert "category" not in dumped
    with pytest.raises(ValidationError):
        UpdateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            id=file_id,
            name="",
            content="正文",
            description="summary",
        )
    with pytest.raises(ValidationError):
        UpdateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            id=file_id,
            name="x" * (LIBRARY_NAME_MAX_LENGTH + 1),
            content="正文",
            description="summary",
        )
    with pytest.raises(ValidationError):
        UpdateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            id=file_id,
            name="标题",
            content="   ",
            description="summary",
        )
    with pytest.raises(ValidationError):
        UpdateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            id=file_id,
            name="标题",
            content="x" * (LIBRARY_CONTENT_MAX_LENGTH + 1),
            description="summary",
        )
    with pytest.raises(ValidationError):
        UpdateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            id=file_id,
            name="标题",
            content="正文",
            description="line one\nline two",
        )
    with pytest.raises(ValidationError):
        UpdateLibrarySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            id=file_id,
            name="标题",
            content="正文",
            description="x" * (LIBRARY_DESCRIPTION_MAX_LENGTH + 1),
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


@asynccontextmanager
async def _sql_store() -> AsyncIterator[SqlLibraryStore]:
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    store = SqlLibraryStore(engine)
    await store.startup()
    try:
        yield store
    finally:
        await engine.dispose()


@asynccontextmanager
async def _open_store(backend: str) -> AsyncIterator[LibraryStore]:
    if backend == "memory":
        yield InMemoryLibraryStore()
        return
    async with _sql_store() as store:
        yield store


async def _create(
    store: LibraryStore,
    name: str,
    content: str = "正文",
    description: str = "One-line summary",
    category: str = "personal",
    **scope: str,
) -> str:
    created = await store.create(
        org_id=scope.get("org_id", "org"),
        project_id=scope.get("project_id", "project"),
        role_id=scope.get("role_id", "library"),
        file_id=str(uuid4()),
        name=name,
        content=content,
        description=description,
        category=category,
    )
    return created.id


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_get_returns_stored_fields(backend: str):
    async with _open_store(backend) as store:
        file_id = await _create(
            store,
            "服务范围",
            content="body",
            description="Lobby coverage",
            category="business",
        )
        file = await store.get(**_SCOPE, file_id=file_id)
        assert file.id == file_id
        assert file.name == "服务范围"
        assert file.content == "body"
        assert file.description == "Lobby coverage"
        assert file.category == "business"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_update_overwrites_title_body_and_summary(backend: str):
    async with _open_store(backend) as store:
        file_id = await _create(
            store,
            "服务范围",
            "original",
            category="business",
        )
        updated = await store.update(
            **_SCOPE,
            file_id=file_id,
            name="营业时间",
            content="# revised\n\nbody",
            description="Revised summary",
        )
        assert updated.id == file_id
        assert updated.name == "营业时间"
        assert updated.content.startswith("# revised")
        assert updated.description == "Revised summary"
        assert updated.category == "business"
        listed = await store.list_names(**_SCOPE)
        assert len(listed) == 1
        assert listed[0].id == file_id
        assert listed[0].name == "营业时间"
        assert listed[0].description == "Revised summary"
        assert listed[0].category == "business"
        assert not hasattr(listed[0], "content")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_update_same_title_rewrites_body(backend: str):
    async with _open_store(backend) as store:
        file_id = await _create(store, "服务范围", "original")
        updated = await store.update(
            **_SCOPE,
            file_id=file_id,
            name="服务范围",
            content="revised",
            description="Still the same file",
        )
        assert updated.id == file_id
        assert updated.name == "服务范围"
        assert updated.content == "revised"
        assert updated.description == "Still the same file"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_update_duplicate_title_keeps_the_old_file(backend: str):
    async with _open_store(backend) as store:
        kept_id = await _create(store, "价格", "rates")
        file_id = await _create(store, "服务范围", "range")
        with pytest.raises(LibraryNameExistsError):
            await store.update(
                **_SCOPE,
                file_id=file_id,
                name="价格",
                content="new body",
                description="new summary",
            )
        current = await store.get(**_SCOPE, file_id=file_id)
        assert current.name == "服务范围"
        assert current.content == "range"
        other = await store.get(**_SCOPE, file_id=kept_id)
        assert other.content == "rates"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_create_duplicate_title_is_rejected(backend: str):
    async with _open_store(backend) as store:
        await _create(store, "服务范围", "original")
        with pytest.raises(LibraryNameExistsError):
            await _create(store, "服务范围", "other")
        await _create(store, "服务范围", "other-role", role_id="other")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_list_is_a_catalog_without_bodies(backend: str):
    async with _open_store(backend) as store:
        empty = await store.list_names(**_SCOPE)
        assert empty == []
        first = await _create(
            store,
            "a-old",
            content="secret body",
            description="Old summary",
        )
        second = await _create(
            store,
            "b-new",
            content="other body",
            description="New summary",
            category="business",
        )
        await store.update(
            **_SCOPE,
            file_id=first,
            name="a-old",
            content="secret body",
            description="Updated summary",
        )
        listed = await store.list_names(**_SCOPE)
        assert [item.id for item in listed] == [first, second]
        assert listed[0].description == "Updated summary"
        for item in listed:
            assert not hasattr(item, "content")
        fetched = await store.get(**_SCOPE, file_id=first)
        assert fetched.content == "secret body"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_missing_id_and_repeat_delete(backend: str):
    async with _open_store(backend) as store:
        missing = str(uuid4())
        with pytest.raises(ResourceNotFoundError):
            await store.get(**_SCOPE, file_id=missing)
        with pytest.raises(ResourceNotFoundError):
            await store.update(
                **_SCOPE,
                file_id=missing,
                name="标题",
                content="正文",
                description="summary",
            )
        await store.delete(**_SCOPE, file_id=missing)
        file_id = await _create(store, "服务范围")
        await store.delete(**_SCOPE, file_id=file_id)
        await store.delete(**_SCOPE, file_id=file_id)
        assert await store.list_names(**_SCOPE) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_delete_project_removes_library_files(backend: str):
    async with _open_store(backend) as store:
        await _create(store, "本项目")
        await _create(store, "其他项目", project_id="other")
        await store.delete_project(org_id="org", project_id="project")
        assert await store.list_names(**_SCOPE) == []
        kept = await store.list_names(
            org_id="org", project_id="other", role_id="library"
        )
        assert [item.name for item in kept] == ["其他项目"]


async def _fill_scope(
    store: LibraryStore,
    *,
    count: int,
    org_id: str = "org",
    project_id: str = "project",
    role_id: str = "library",
    category: str = "personal",
    name_prefix: str = "file",
) -> list[str]:
    ids: list[str] = []
    for i in range(count):
        created = await store.create(
            org_id=org_id,
            project_id=project_id,
            role_id=role_id,
            file_id=str(uuid4()),
            name=f"{name_prefix}-{i}",
            content="body",
            description="summary",
            category=category,
        )
        ids.append(created.id)
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sql"])
async def test_rejects_a_twenty_first_file_in_the_same_category(backend: str):
    async with _open_store(backend) as store:
        ids = await _fill_scope(store, count=LIBRARY_MAX_FILES_PER_CATEGORY)
        assert len(ids) == LIBRARY_MAX_FILES_PER_CATEGORY
        with pytest.raises(LibraryFileLimitError):
            await store.create(
                **_SCOPE,
                file_id=str(uuid4()),
                name="file-20",
                content="body",
                description="summary",
                category="personal",
            )
        business = await store.create(
            **_SCOPE,
            file_id=str(uuid4()),
            name="biz-0",
            content="body",
            description="summary",
            category="business",
        )
        assert business.category == "business"
        await store.create(
            org_id="org",
            project_id="other-project",
            role_id="library",
            file_id=str(uuid4()),
            name="file-0",
            content="body",
            description="summary",
            category="personal",
        )
        await store.delete(**_SCOPE, file_id=ids[0])
        created = await store.create(
            **_SCOPE,
            file_id=str(uuid4()),
            name="file-20",
            content="body",
            description="summary",
            category="personal",
        )
        assert created.name == "file-20"
        await store.update(
            **_SCOPE,
            file_id=created.id,
            name="file-20-renamed",
            content="still personal",
            description="quota unchanged",
        )
        with pytest.raises(LibraryFileLimitError):
            await store.create(
                **_SCOPE,
                file_id=str(uuid4()),
                name="file-21",
                content="body",
                description="summary",
                category="personal",
            )
