"""Ingestion pipeline for converting episodes into semantic features."""

import asyncio
import itertools
import logging
from itertools import chain

import numpy as np
from pydantic import BaseModel, InstanceOf, TypeAdapter

from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import Episode, EpisodeIdT, EpisodeStorage
from memmachine.common.filter.filter_parser import And, Comparison
from memmachine.semantic_memory.semantic_llm import (
    LLMReducedFeature,
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.semantic_memory.semantic_model import (
    ResourceRetriever,
    Resources,
    SemanticCategory,
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
    SetIdT,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage

logger = logging.getLogger(__name__)


def _get_isolation_type(set_id: str) -> str:
    """Determine the isolation type from set_id prefix."""
    if set_id.startswith("mem_session_"):
        return "session"
    elif set_id.startswith("mem_user_"):
        return "user"
    elif set_id.startswith("mem_role_"):
        return "role"
    return "unknown"


class IngestionService:
    """
    Processes un-ingested history for each set_id and updates semantic features.

    The service pulls pending messages, invokes the LLM to generate mutation commands,
    applies the resulting changes, and optionally consolidates redundant memories.
    """

    class Params(BaseModel):
        """Dependencies and tuning knobs for the ingestion workflow."""

        semantic_storage: InstanceOf[SemanticStorage]
        history_store: InstanceOf[EpisodeStorage]
        resource_retriever: InstanceOf[ResourceRetriever]
        consolidated_threshold: int = 10
        debug_fail_loudly: bool = False

    def __init__(self, params: Params) -> None:
        """Initialize the ingestion service with storage backends and helpers."""
        self._semantic_storage = params.semantic_storage
        self._history_store = params.history_store
        self._resource_retriever = params.resource_retriever
        self._consolidation_threshold = params.consolidated_threshold
        self._debug_fail_loudly = params.debug_fail_loudly
        
        # Generate unique owner ID for this pod/process
        import os
        import socket
        hostname = socket.gethostname()
        pid = os.getpid()
        self._owner_id = f"{hostname}-{pid}"

    async def process_set_ids(self, set_ids: list[SetIdT]) -> None:
        logger.info("Starting ingestion processing for set ids: %s", set_ids)

        # Filter out sets that have no semantic categories configured
        # This prevents unnecessary processing of session/role sets when only profile memory is enabled
        valid_set_ids = []
        for set_id in set_ids:
            resources = self._resource_retriever.get_resources(set_id)
            if len(resources.semantic_categories) == 0:
                isolation_type = _get_isolation_type(set_id)
                logger.debug(
                    "Skipping set_id %s (%s) - no semantic categories configured",
                    set_id,
                    isolation_type,
                )
                continue
            valid_set_ids.append(set_id)
        
        if len(valid_set_ids) == 0:
            logger.debug("No valid set_ids to process after filtering")
            return

        logger.info("Processing %d set_ids (filtered from %d)", len(valid_set_ids), len(set_ids))
        
        results = await asyncio.gather(
            *[self._process_single_set(set_id) for set_id in valid_set_ids],
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) > 0:
            raise ExceptionGroup("Failed to process set ids", errors)

    async def _process_single_set(self, set_id: str) -> None:  # noqa: C901
        """
        Process all uningested messages for a single set_id.
        
        This method acquires a lock for the entire ingestion cycle to prevent
        race conditions when multiple pods try to process the same set_id.
        """
        # Try to acquire ingestion lock for this set_id
        logger.info(
            "Attempting to acquire ingestion lock for set_id=%s, owner=%s",
            set_id,
            self._owner_id,
        )
        
        lock_acquired = await self._semantic_storage.try_acquire_ingestion_lock(
            set_id=set_id,
            owner_id=self._owner_id,
            timeout_seconds=300,  # 5 minutes timeout
        )
        
        if not lock_acquired:
            logger.info(
                "SKIPPED set_id=%s - lock held by another pod, owner=%s will not process",
                set_id,
                self._owner_id,
            )
            return
        
        logger.info(
            "ACQUIRED lock for set_id=%s, owner=%s - starting ingestion",
            set_id,
            self._owner_id,
        )
        
        try:
            await self._process_single_set_with_lock(set_id)
        finally:
            # Always release the lock, even if processing fails
            await self._semantic_storage.release_ingestion_lock(
                set_id=set_id,
                owner_id=self._owner_id,
            )
            logger.info(
                "RELEASED lock for set_id=%s, owner=%s - ingestion complete",
                set_id,
                self._owner_id,
            )
    
    async def _process_single_set_with_lock(self, set_id: str) -> None:  # noqa: C901
        """
        Process all uningested messages for a set_id (called after lock is acquired).
        
        This processes messages in batches of 50 until all are processed,
        then performs consolidation once at the end.
        
        Note: This is only called for sets that have semantic categories configured.
        """
        logger.info("Processing semantic ingestion for set_id: %s", set_id)
        resources = self._resource_retriever.get_resources(set_id)

        # Process all uningested messages in batches of 50
        total_processed = 0
        while True:
            # Atomically claim next batch of messages
            history_ids = await self._semantic_storage.get_history_messages(
                set_ids=[set_id],
                limit=50,
                is_ingested=False,
            )
            
            if len(history_ids) == 0:
                logger.info(
                    "Finished processing %d total messages for set_id %s",
                    total_processed,
                    set_id,
                )
                break
            
            logger.info(
                "Processing batch of %d messages for set_id %s (total so far: %d)",
                len(history_ids),
                set_id,
                total_processed,
            )
            
            # Process this batch
            await self._process_message_batch(
                set_id=set_id,
                history_ids=history_ids,
                resources=resources,
            )
            
            total_processed += len(history_ids)
        
        # After processing all messages, consolidate once
        if total_processed > 0:
            logger.debug("Starting consolidation for set_id %s", set_id)
            await self._consolidate_set_memories_if_applicable(
                set_id=set_id,
                resources=resources,
            )
    
    async def _process_message_batch(
        self,
        set_id: str,
        history_ids: list[EpisodeIdT],
        resources: InstanceOf[Resources],
    ) -> None:
        """Process a batch of messages for a set_id."""
        raw_messages = await asyncio.gather(
            *[self._history_store.get_episode(h_id) for h_id in history_ids],
        )

        if len(raw_messages) != len([m for m in raw_messages if m is not None]):
            raise ValueError("Failed to retrieve messages. Invalid history_ids")

        messages = TypeAdapter(list[Episode]).validate_python(raw_messages)

        logger.info("Processing %d messages for set %s", len(messages), set_id)

        async def process_semantic_type(
            semantic_category: InstanceOf[SemanticCategory],
        ) -> None:
            logger.debug(
                "Processing semantic category '%s' for set_id %s with %d messages",
                semantic_category.name,
                set_id,
                len(messages),
            )
            for message in messages:
                if message.uid is None:
                    logger.error(
                        "Message ID is None for message %s", message.model_dump()
                    )

                    raise ValueError(
                        "Message ID is None for message %s",
                        message.model_dump(),
                    )

                filter_expr = And(
                    left=Comparison(field="set_id", op="=", value=set_id),
                    right=Comparison(
                        field="category", op="=", value=semantic_category.name
                    ),
                )

                features = await self._semantic_storage.get_feature_set(
                    filter_expr=filter_expr,
                )
                logger.debug(
                    "Found %d existing features for set_id %s, category %s",
                    len(features),
                    set_id,
                    semantic_category.name,
                )

                if semantic_category.name == "profile":
                    # Filter features to only include valid tags for profile category
                    valid_tags = {"basics", "contacts", "identities", "accounts", "preferences", "relationships", "services", "others"}
                    original_count = len(features)
                    features = [f for f in features if f.tag in valid_tags]
                    filtered_count = original_count - len(features)
                    
                    if filtered_count > 0:
                        logger.info(
                            "Filtered out %d features with invalid tags for profile category - set_id=%s, message_id=%s, kept=%d features",
                            filtered_count,
                            set_id,
                            message.uid,
                            len(features),
                        )
                elif semantic_category.name == "profile_life_context":
                    # Filter features to only include valid tags for profile_life_context category
                    valid_tags = {"interests", "lifestyle", "goals", "personality", "life_situation"}
                    original_count = len(features)
                    features = [f for f in features if f.tag in valid_tags]
                    filtered_count = original_count - len(features)
                    
                    if filtered_count > 0:
                        logger.info(
                            "Filtered out %d features with invalid tags for profile_life_context category - set_id=%s, message_id=%s, kept=%d features",
                            filtered_count,
                            set_id,
                            message.uid,
                            len(features),
                        )

                try:
                    commands = await llm_feature_update(
                        features=features,
                        message_content=message.content,
                        model=resources.language_model,
                        update_prompt=semantic_category.prompt.update_prompt,
                    )
                    logger.debug(
                        "LLM generated %d commands for message %s, category %s",
                        len(commands),
                        message.uid,
                        semantic_category.name,
                    )
                    
                    # Normalize and validate tags for profile category
                    if semantic_category.name == "profile":
                        valid_tags = {"basics", "contacts", "identities", "accounts", "preferences", "relationships", "services", "others"}
                        default_tag = "others"
                        corrected_count = 0
                        for cmd in commands:
                            original_tag = cmd.tag
                            # First, convert to lowercase
                            normalized_tag = cmd.tag.lower().strip()
                            # Then check if it's in valid tags
                            if normalized_tag not in valid_tags:
                                normalized_tag = default_tag
                            
                            if original_tag != normalized_tag:
                                corrected_count += 1
                                logger.info(
                                    "Corrected tag '%s' -> '%s' for command in message %s",
                                    original_tag,
                                    normalized_tag,
                                    message.uid,
                                )
                            cmd.tag = normalized_tag
                        
                        if corrected_count > 0:
                            logger.warning(
                                "Corrected %d tag(s) for profile category - set_id=%s, message_id=%s",
                                corrected_count,
                                set_id,
                                message.uid,
                            )
                    elif semantic_category.name == "profile_life_context":
                        valid_tags = {"interests", "lifestyle", "goals", "personality", "life_situation"}
                        default_tag = "interests"  # Default to interests as it's the most general
                        corrected_count = 0
                        for cmd in commands:
                            original_tag = cmd.tag
                            # First, convert to lowercase
                            normalized_tag = cmd.tag.lower().strip()
                            # Then check if it's in valid tags
                            if normalized_tag not in valid_tags:
                                normalized_tag = default_tag
                            
                            if original_tag != normalized_tag:
                                corrected_count += 1
                                logger.info(
                                    "Corrected tag '%s' -> '%s' for command in message %s",
                                    original_tag,
                                    normalized_tag,
                                    message.uid,
                                )
                            cmd.tag = normalized_tag
                        
                        if corrected_count > 0:
                            logger.warning(
                                "Corrected %d tag(s) for profile_life_context category - set_id=%s, message_id=%s",
                                corrected_count,
                                set_id,
                                message.uid,
                            )
                except Exception:
                    logger.exception(
                        "Failed to process message %s for semantic type %s",
                        message.uid,
                        semantic_category.name,
                    )
                    if self._debug_fail_loudly:
                        raise

                    continue

                await self._apply_commands(
                    commands=commands,
                    set_id=set_id,
                    category_name=semantic_category.name,
                    citation_id=message.uid,
                    embedder=resources.embedder,
                )
                logger.debug(
                    "Applied %d commands for message %s, category %s",
                    len(commands),
                    message.uid,
                    semantic_category.name,
                )

                mark_messages.append(message.uid)

        mark_messages: list[EpisodeIdT] = []
        semantic_category_runners = []
        for t in resources.semantic_categories:
            task = process_semantic_type(t)
            semantic_category_runners.append(task)

        await asyncio.gather(*semantic_category_runners)

        logger.info(
            "Finished processing %d messages out of %d for batch in set %s",
            len(mark_messages),
            len(messages),
            set_id,
        )

        if len(mark_messages) == 0:
            logger.warning(
                "No messages were successfully processed for set_id %s. "
                "This may indicate LLM processing errors.",
                set_id,
            )
            # Note: Messages are already marked as ingested when claimed via get_history_messages()
            # Even if processing failed, we don't want to retry them to avoid infinite loops
            return

        # Note: Messages are already marked as ingested atomically when claimed
        # via get_history_messages(). No need to mark them again here.
        # Consolidation will be performed once after all batches are processed.

    async def _apply_commands(
        self,
        *,
        commands: list[SemanticCommand],
        set_id: SetIdT,
        category_name: str,
        citation_id: EpisodeIdT | None,
        embedder: InstanceOf[Embedder],
    ) -> None:
        for command in commands:
            match command.command:
                case SemanticCommandType.ADD:
                    value_embedding = (await embedder.ingest_embed([command.value]))[0]

                    f_id = await self._semantic_storage.add_feature(
                        set_id=set_id,
                        category_name=category_name,
                        feature=command.feature,
                        value=command.value,
                        tag=command.tag,
                        embedding=np.array(value_embedding),
                    )

                    if citation_id is not None:
                        await self._semantic_storage.add_citations(f_id, [citation_id])

                case SemanticCommandType.DELETE:
                    filter_expr = And(
                        left=And(
                            left=Comparison(field="set_id", op="=", value=set_id),
                            right=Comparison(
                                field="category_name", op="=", value=category_name
                            ),
                        ),
                        right=And(
                            left=Comparison(
                                field="feature", op="=", value=command.feature
                            ),
                            right=Comparison(field="tag", op="=", value=command.tag),
                        ),
                    )

                    await self._semantic_storage.delete_feature_set(
                        filter_expr=filter_expr
                    )

                case _:
                    logger.error("Command with unknown action: %s", command.command)

    async def _consolidate_set_memories_if_applicable(
        self,
        *,
        set_id: SetIdT,
        resources: InstanceOf[Resources],
    ) -> None:
        async def _consolidate_type(
            semantic_category: InstanceOf[SemanticCategory],
        ) -> None:
            from memmachine.common.filter.filter_parser import And, Comparison

            filter_expr = And(
                left=Comparison(field="set_id", op="=", value=set_id),
                right=Comparison(
                    field="category_name", op="=", value=semantic_category.name
                ),
            )

            features = await self._semantic_storage.get_feature_set(
                filter_expr=filter_expr,
                tag_threshold=self._consolidation_threshold,
                load_citations=True,
            )

            consolidation_sections: list[list[SemanticFeature]] = list(
                SemanticFeature.group_features_by_tag_only(features).values(),
            )

            await asyncio.gather(
                *[
                    self._deduplicate_features(
                        set_id=set_id,
                        memories=section_features,
                        resources=resources,
                        semantic_category=semantic_category,
                    )
                    for section_features in consolidation_sections
                ],
            )

        category_tasks = []
        for t in resources.semantic_categories:
            task = _consolidate_type(t)
            category_tasks.append(task)

        await asyncio.gather(*category_tasks)

    async def _deduplicate_features(
        self,
        *,
        set_id: str,
        memories: list[SemanticFeature],
        semantic_category: InstanceOf[SemanticCategory],
        resources: InstanceOf[Resources],
    ) -> None:
        logger.info(
            "Deduplicating %d features for set_id=%s, category=%s",
            len(memories),
            set_id,
            semantic_category.name,
        )

        original_tag = memories[0].tag if len(memories) > 0 else None

        try:
            consolidate_resp = await llm_consolidate_features(
                features=memories,
                model=resources.language_model,
                consolidate_prompt=semantic_category.prompt.consolidation_prompt,
            )
        except (ValueError, TypeError):
            logger.exception("Failed to update features while calling LLM")
            if self._debug_fail_loudly:
                raise
            return

        if consolidate_resp is None or consolidate_resp.keep_memories is None:
            logger.warning("Failed to consolidate features")
            if self._debug_fail_loudly:
                raise ValueError("Failed to consolidate features")
            return

        logger.info(
            "Consolidation result: keep_memories=%s, consolidated_memories=%d",
            consolidate_resp.keep_memories,
            len(consolidate_resp.consolidated_memories),
        )

        memories_to_delete = [
            m
            for m in memories
            if m.metadata.id is not None
            and m.metadata.id not in consolidate_resp.keep_memories
        ]

        if memories_to_delete:
            delete_ids = [
                m.metadata.id for m in memories_to_delete if m.metadata.id is not None
            ]
            logger.info(
                "Deleting %d features: %s",
                len(delete_ids),
                delete_ids,
            )
            await self._semantic_storage.delete_features(delete_ids)

        merged_citations: chain[EpisodeIdT] = itertools.chain.from_iterable(
            [
                m.metadata.citations
                for m in memories_to_delete
                if m.metadata.citations is not None
            ],
        )
        citation_ids = TypeAdapter(list[EpisodeIdT]).validate_python(
            list(merged_citations),
        )

        # Ensure consolidated memories maintain the same tag as input memories
        # All input memories should have the same tag (grouped by tag)
        if original_tag and consolidate_resp.consolidated_memories:
            expected_tag = original_tag
            corrected_count = 0
            
            for f in consolidate_resp.consolidated_memories:
                if f.tag != expected_tag:
                    corrected_count += 1
                    logger.warning(
                        "Consolidation changed tag from '%s' to '%s' for feature '%s' - fixing to maintain input tag",
                        expected_tag,
                        f.tag,
                        f.feature,
                    )
                    f.tag = expected_tag
            
            if corrected_count > 0:
                logger.warning(
                    "Fixed %d consolidated tag(s) to maintain input tag '%s' - set_id=%s, category=%s",
                    corrected_count,
                    expected_tag,
                    set_id,
                    semantic_category.name,
                )

        async def _add_feature(f: LLMReducedFeature) -> None:
            value_embedding = (await resources.embedder.ingest_embed([f.value]))[0]

            f_id = await self._semantic_storage.add_feature(
                set_id=set_id,
                category_name=semantic_category.name,
                tag=f.tag,
                feature=f.feature,
                value=f.value,
                embedding=np.array(value_embedding),
            )
            logger.info(
                "Added consolidated feature: id=%s, tag=%s, feature=%s",
                f_id,
                f.tag,
                f.feature,
            )

            await self._semantic_storage.add_citations(f_id, citation_ids)

        if consolidate_resp.consolidated_memories:
            logger.info(
                "Adding %d consolidated features",
                len(consolidate_resp.consolidated_memories),
            )
            await asyncio.gather(
                *[
                    _add_feature(feature)
                    for feature in consolidate_resp.consolidated_memories
                ],
            )
