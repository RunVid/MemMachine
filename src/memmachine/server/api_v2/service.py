"""API v2 service implementations."""

import logging
from dataclasses import dataclass

from fastapi import Request

from memmachine import MemMachine

logger = logging.getLogger(__name__)
from memmachine.common.api import MemoryType as MemoryTypeE
from memmachine.common.api.spec import (
    AddMemoriesSpec,
    AddMemoryResult,
    ConsolidateMemoriesResponse,
    Episode,
    EpisodicSearchResult,
    ListMemoriesSpec,
    ListResult,
    ListResultContent,
    SearchMemoriesSpec,
    SearchResult,
    SearchResultContent,
    SemanticFeature,
)
from memmachine.common.episode_store.episode_model import EpisodeEntry


# Placeholder dependency injection function
async def get_memmachine(request: Request) -> MemMachine:
    """Get session data manager instance."""
    return request.app.state.mem_machine


@dataclass
class _SessionData:
    org_id: str
    project_id: str
    user_id: str | None = None
    role_id: str | None = None
    session_id_override: str | None = None

    @property
    def session_key(self) -> str:
        return f"{self.org_id}/{self.project_id}"

    @property
    def user_profile_id(self) -> str | None:
        # Return user_id without prefix - SemanticSessionManager._generate_session_data
        # will add the "mem_user_" prefix automatically
        return self.user_id

    @property
    def role_profile_id(self) -> str | None:
        # Return role_id without prefix - SemanticSessionManager._generate_session_data
        # will add the "mem_role_" prefix automatically
        return self.role_id

    @property
    def session_id(self) -> str | None:
        return self.session_id_override if self.session_id_override else self.session_key


def _extract_ids_from_messages(
    messages: list, project_id: str
) -> tuple[str | None, str | None, str | None]:
    """
    Extract user_id, role_id, and session_id from message metadata.
    
    If user_id is not provided in metadata, use project_id as user_id
    (assuming one user per project).
    """
    user_id: str | None = None
    role_id: str | None = None
    session_id: str | None = None

    # Extract from first message's metadata (assuming all messages have consistent metadata)
    if messages and hasattr(messages[0], "metadata") and messages[0].metadata:
        metadata = messages[0].metadata
        user_id = metadata.get("user_id")
        role_id = metadata.get("role_id")
        session_id = metadata.get("session_id")

    # If user_id is not provided, use project_id as user_id (one user per project)
    if user_id is None or user_id == "":
        user_id = project_id
        logger.debug(
            "user_id not provided in metadata, using project_id as user_id: %s",
            user_id,
        )
    else:
        logger.debug("Extracted user_id from metadata: %s", user_id)

    logger.debug(
        "Extracted IDs - user_id: %s, role_id: %s, session_id: %s",
        user_id,
        role_id,
        session_id,
    )
    return user_id, role_id, session_id


async def _add_messages_to(
    target_memories: list[MemoryTypeE],
    spec: AddMemoriesSpec,
    memmachine: MemMachine,
) -> list[AddMemoryResult]:
    # Extract user_id, role_id, session_id from message metadata
    # If user_id is not provided, use project_id as user_id (one user per project)
    user_id, role_id, session_id = _extract_ids_from_messages(spec.messages, spec.project_id)

    episodes: list[EpisodeEntry] = [
        EpisodeEntry(
            content=message.content,
            producer_id=message.producer,
            produced_for_id=message.produced_for,
            producer_role=message.role,
            created_at=message.timestamp,
            metadata=message.metadata,
            episode_type=message.episode_type,
        )
        for message in spec.messages
    ]

    session_data = _SessionData(
        org_id=spec.org_id,
        project_id=spec.project_id,
        user_id=user_id,
        role_id=role_id,
        session_id_override=session_id,
    )
    logger.info(
        "Adding %d episodes to memories - org_id: %s, project_id: %s, user_id: %s, "
        "user_profile_id: %s, target_memories: %s",
        len(episodes),
        spec.org_id,
        spec.project_id,
        user_id,
        session_data.user_profile_id,
        [m.value for m in target_memories],
    )

    episode_ids = await memmachine.add_episodes(
        session_data=session_data,
        episode_entries=episodes,
        target_memories=target_memories,
    )
    logger.info(
        "Added %d episodes, returned %d episode_ids",
        len(episodes),
        len(episode_ids),
    )
    return [AddMemoryResult(uid=e_id) for e_id in episode_ids]


async def _search_target_memories(
    target_memories: list[MemoryTypeE],
    spec: SearchMemoriesSpec,
    memmachine: MemMachine,
) -> SearchResult:
    # For search, use project_id as user_id (one user per project)
    # This ensures semantic memory search targets the correct user profile
    user_id: str | None = spec.project_id
    role_id: str | None = None
    session_id: str | None = None

    results = await memmachine.query_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
            user_id=user_id,
            role_id=role_id,
            session_id_override=session_id,
        ),
        query=spec.query,
        target_memories=target_memories,
        search_filter=spec.filter,
        limit=spec.top_k,
        score_threshold=spec.score_threshold
        if spec.score_threshold is not None
        else -float("inf"),
    )
    content = SearchResultContent(
        episodic_memory=None,
        semantic_memory=None,
    )
    if results.episodic_memory is not None:
        content.episodic_memory = EpisodicSearchResult(
            **results.episodic_memory.model_dump(mode="json")
        )
    if results.semantic_memory is not None:
        content.semantic_memory = [
            SemanticFeature(**f.model_dump(mode="json"))
            for f in results.semantic_memory
        ]
    return SearchResult(
        status=0,
        content=content,
    )


async def _list_target_memories(
    target_memories: list[MemoryTypeE],
    spec: ListMemoriesSpec,
    memmachine: MemMachine,
) -> ListResult:
    # For list, use project_id as user_id (one user per project)
    # This ensures semantic memory list targets the correct user profile
    user_id: str | None = spec.project_id
    role_id: str | None = None
    session_id: str | None = None

    results = await memmachine.list_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
            user_id=user_id,
            role_id=role_id,
            session_id_override=session_id,
        ),
        target_memories=target_memories,
        search_filter=spec.filter,
        page_size=spec.page_size,
        page_num=spec.page_num,
    )

    content = ListResultContent(
        episodic_memory=None,
        semantic_memory=None,
    )
    if results.episodic_memory is not None:
        content.episodic_memory = [
            Episode(**e.model_dump(mode="json")) for e in results.episodic_memory
        ]
    if results.semantic_memory is not None:
        content.semantic_memory = [
            SemanticFeature(**f.model_dump(mode="json"))
            for f in results.semantic_memory
        ]

    return ListResult(
        status=0,
        content=content,
    )


async def _consolidate_memories(
    spec: "ConsolidateMemoriesSpec",
    memmachine: MemMachine,
) -> ConsolidateMemoriesResponse:
    """
    Trigger consolidation for a specific set_id.

    Args:
        spec: Consolidation specification with set_id and force flag
        memmachine: MemMachine instance

    Returns:
        ConsolidateMemoriesResponse with summary of consolidation results
    """
    from memmachine.common.api.spec import ConsolidateMemoriesSpec

    logger.info(
        "Consolidating memories - set_id: %s, force: %s",
        spec.set_id,
        spec.force,
    )

    lock_acquired, consolidated = await memmachine.trigger_consolidation(
        set_id=spec.set_id,
        force=spec.force,
    )

    # Build response message
    if not lock_acquired:
        message = (
            f"Consolidation skipped for set_id '{spec.set_id}': "
            "Lock is held by another process. Try again later."
        )
    elif not consolidated:
        message = (
            f"No consolidation applied for set_id '{spec.set_id}'. "
            "The set may have insufficient memories to consolidate. "
            "Try using force=true to bypass threshold checks."
        )
    else:
        message = f"Consolidation complete for set_id '{spec.set_id}'."

    return ConsolidateMemoriesResponse(
        message=message,
        set_id=spec.set_id,
        consolidated=consolidated,
        lock_acquired=lock_acquired,
    )

