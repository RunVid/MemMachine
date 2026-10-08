"""Tests for the role-scoped library."""

from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from memmachine.common.api.spec import (
    LIBRARY_CONTENT_MAX_LENGTH,
    LIBRARY_DESCRIPTION_MAX_LENGTH,
    LIBRARY_MAX_FILES_PER_SCOPE,
    CreateLibrarySpec,
    LibraryCategory,
    LibraryIdSpec,
    UpdateLibraryCategorySpec,
)
from memmachine.common.errors import (
    LibraryFileLimitError,
    LibraryNameExistsError,
    ResourceNotFoundError,
)
from memmachine.library_store.memory_store import InMemoryLibraryStore
from memmachine.library_store.protocol import LibraryStore
from memmachine.library_store.sql_store import SqlLibraryStore


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


def test_update_category_spec_requires_personal_or_business():
    spec = UpdateLibraryCategorySpec(
        org_id="org",
        project_id="project",
        role_id="library",
        id=str(uuid4()),
        category=" BUSINESS ",
    )
    assert spec.category == LibraryCategory.BUSINESS
    with pytest.raises(ValidationError):
        UpdateLibraryCategorySpec(
            org_id="org",
            project_id="project",
            role_id="library",
            id=str(uuid4()),
            category="work",
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


async def _create(
    store: LibraryStore,
    name: str,
    content: str = "正文",
    description: str = "One-line summary",
    category: str = "personal",
) -> str:
    created = await store.create(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=str(uuid4()),
        name=name,
        content=content,
        description=description,
        category=category,
    )
    return created.id


@pytest.mark.asyncio
async def test_get_returns_name_content_and_description():
    store = InMemoryLibraryStore()
    file_id = await _create(
        store,
        "服务范围",
        content="body",
        description="Lobby coverage",
        category="business",
    )
    file = await store.get(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
    )
    assert file.name == "服务范围"
    assert file.content == "body"
    assert file.description == "Lobby coverage"
    assert file.category == "business"
    assert file.always_loaded is False


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
    recategorized = await store.update_category(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
        category="business",
    )
    assert recategorized.id == file_id
    assert recategorized.name == "服务范围"
    assert recategorized.category == "business"
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
    assert len(listed) == 1
    assert listed[0].id == file_id
    assert listed[0].name == "营业时间"
    assert listed[0].description == "One-line summary"
    assert listed[0].category == "business"


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
    with pytest.raises(ResourceNotFoundError):
        await store.update_category(
            org_id="org",
            project_id="project",
            role_id="library",
            file_id=missing,
            category="business",
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
    recategorized = await store.update_category(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=file_id,
        category="business",
    )
    assert recategorized.category == "business"
    assert recategorized.name == "服务范围"
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


async def _fill_scope(
    store: LibraryStore,
    *,
    count: int,
    org_id: str = "org",
    project_id: str = "project",
    role_id: str = "library",
) -> list[str]:
    ids: list[str] = []
    for i in range(count):
        created = await store.create(
            org_id=org_id,
            project_id=project_id,
            role_id=role_id,
            file_id=str(uuid4()),
            name=f"file-{i}",
            content="body",
            description="summary",
            category="personal",
        )
        ids.append(created.id)
    return ids


@pytest.mark.asyncio
async def test_memory_store_rejects_a_fifty_first_file_in_the_same_scope():
    store = InMemoryLibraryStore()
    ids = await _fill_scope(store, count=LIBRARY_MAX_FILES_PER_SCOPE)
    assert len(ids) == LIBRARY_MAX_FILES_PER_SCOPE
    with pytest.raises(LibraryFileLimitError):
        await store.create(
            org_id="org",
            project_id="project",
            role_id="library",
            file_id=str(uuid4()),
            name="file-50",
            content="body",
            description="summary",
            category="personal",
        )
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
    await store.delete(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=ids[0],
    )
    created = await store.create(
        org_id="org",
        project_id="project",
        role_id="library",
        file_id=str(uuid4()),
        name="file-50",
        content="body",
        description="summary",
        category="personal",
    )
    assert created.name == "file-50"


@pytest.mark.asyncio
async def test_sql_store_rejects_a_fifty_first_file_in_the_same_scope():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    store = SqlLibraryStore(engine)
    await store.startup()
    await _fill_scope(store, count=LIBRARY_MAX_FILES_PER_SCOPE)
    with pytest.raises(LibraryFileLimitError):
        await store.create(
            org_id="org",
            project_id="project",
            role_id="library",
            file_id=str(uuid4()),
            name="file-50",
            content="body",
            description="summary",
            category="personal",
        )
    await engine.dispose()
