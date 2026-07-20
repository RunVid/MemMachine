"""
Live LLM prompt tests for agent_personality (manual write + ingest).

Requires OPENAI_API_KEY in the environment. Optional:
  OPENAI_BASE_URL (default https://api.openai.com/v1)
  OPENAI_MODEL (default gpt-4o-mini)

Run:
  export OPENAI_API_KEY=...
  pytest tests/memmachine/semantic_memory/test_agent_personality_llm_live.py --integration -v -s
"""

from __future__ import annotations

import os
from pprint import pprint

import openai
import pytest

from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine.semantic_memory.semantic_llm import (
    llm_feature_update,
    llm_parse_manual_instruction,
)
from memmachine.semantic_memory.semantic_manual_write import (
    build_manual_instruction_system_prompt,
)
from memmachine.semantic_memory.semantic_model import (
    SemanticCommandType,
    SemanticFeature,
)
from memmachine.server.prompt.agent_personality_prompt import (
    AgentPersonalitySemanticCategory,
)

pytestmark = pytest.mark.integration

_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_requires_openai = pytest.mark.skipif(
    not _API_KEY,
    reason="OPENAI_API_KEY not set",
)


def _feature(
    *,
    tag: str,
    feature_name: str,
    value: str,
    feature_id: str = "1",
) -> SemanticFeature:
    return SemanticFeature(
        metadata=SemanticFeature.Metadata(id=feature_id),
        category="agent_personality",
        tag=tag,
        feature_name=feature_name,
        value=value,
    )


def _enthusiasm_tone() -> SemanticFeature:
    return _feature(
        tag="tone",
        feature_name="ENTHUSIASM",
        value="Always convey excitement and positivity",
        feature_id="tone-enthusiasm",
    )


@pytest.fixture(scope="module")
def live_llm():
    api_key = os.environ["OPENAI_API_KEY"].strip()
    base_url = (
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        or "https://api.openai.com/v1"
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    print(f"\n=== live_llm model={model!r} base_url={base_url!r} ===")
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(client=client, model=model),
    )


@_requires_openai
@pytest.mark.asyncio
async def test_manual_write_accepts_additional_notification_criterion(live_llm):
    """Related notification criteria must append, not reject as conflict."""
    existing = [
        _enthusiasm_tone(),
        _feature(
            tag="boundaries",
            feature_name="NOTIFICATION SCOPE",
            value="Notify about emails from Mike",
            feature_id="boundaries-notify",
        ),
    ]
    system_prompt = build_manual_instruction_system_prompt(
        category_name="agent_personality",
    )
    parsed = await llm_parse_manual_instruction(
        instruction="Notify me about emails related to Pine tasks",
        existing_features=existing,
        model=live_llm,
        system_prompt=system_prompt,
    )

    print("\n=== manual_write LLM return ===")
    pprint(parsed.model_dump())

    assert parsed.accepted is True, parsed.rejection_reason
    assert parsed.tag.strip().lower() == "boundaries"
    assert parsed.feature_name.strip() != ""
    assert parsed.value.strip() != ""
    reason = parsed.rejection_reason.casefold()
    assert "conflict" not in reason


@_requires_openai
@pytest.mark.asyncio
async def test_manual_write_accepts_mike_then_peter(live_llm):
    """Mike then Peter: append another person criterion, do not reject as conflict."""
    existing = [
        _enthusiasm_tone(),
        _feature(
            tag="boundaries",
            feature_name="NOTIFICATION SCOPE",
            value="Notify about emails from Mike",
            feature_id="boundaries-notify",
        ),
    ]
    system_prompt = build_manual_instruction_system_prompt(
        category_name="agent_personality",
    )
    parsed = await llm_parse_manual_instruction(
        instruction="Notify me about emails from Peter",
        existing_features=existing,
        model=live_llm,
        system_prompt=system_prompt,
    )

    print("\n=== manual_write Mike→Peter LLM return ===")
    pprint(parsed.model_dump())

    assert parsed.accepted is True, parsed.rejection_reason
    assert parsed.tag.strip().lower() == "boundaries"
    assert parsed.feature_name.strip().casefold() != "notification scope", (
        "manual must append under a NEW feature_name, not reuse NOTIFICATION SCOPE"
    )
    assert "peter" in parsed.value.casefold(), parsed.value
    reason = parsed.rejection_reason.casefold()
    assert "conflict" not in reason
    assert "only" not in reason, (
        "must not invent exclusive 'only' into a non-only scope: "
        f"{parsed.rejection_reason}"
    )


@_requires_openai
@pytest.mark.asyncio
async def test_manual_write_accepts_peter_then_mike(live_llm):
    """Peter then Mike: include-Peter is not only-Peter; append Mike, no conflict."""
    existing = [
        _enthusiasm_tone(),
        _feature(
            tag="boundaries",
            feature_name="NOTIFICATION_SCOPE",
            value="Notify about emails from Peter",
            feature_id="boundaries-notify",
        ),
    ]
    system_prompt = build_manual_instruction_system_prompt(
        category_name="agent_personality",
    )
    parsed = await llm_parse_manual_instruction(
        instruction="Notify me about emails from Mike",
        existing_features=existing,
        model=live_llm,
        system_prompt=system_prompt,
    )

    print("\n=== manual_write Peter→Mike LLM return ===")
    pprint(parsed.model_dump())

    assert parsed.accepted is True, parsed.rejection_reason
    assert parsed.tag.strip().lower() == "boundaries"
    assert "mike" in parsed.value.casefold(), parsed.value
    reason = parsed.rejection_reason.casefold()
    assert "conflict" not in reason
    assert "only" not in reason, (
        "must not invent exclusive 'only' into a non-only scope: "
        f"{parsed.rejection_reason}"
    )


@_requires_openai
@pytest.mark.asyncio
async def test_ingest_merges_notification_scope_instead_of_replace(live_llm):
    """Same-topic extend must merge old+new, not replace with only the new clause."""
    existing = [
        _enthusiasm_tone(),
        _feature(
            tag="boundaries",
            feature_name="NOTIFICATION SCOPE",
            value="Only notify about emails related to Pine tasks",
            feature_id="boundaries-notify",
        ),
    ]
    update_prompt = AgentPersonalitySemanticCategory.prompt.update_prompt
    commands = await llm_feature_update(
        features=existing,
        message_contents=["Notify about emails from Mike"],
        model=live_llm,
        update_prompt=update_prompt,
    )

    print("\n=== ingest merge LLM return ===")
    pprint([c.model_dump() for c in commands])

    deletes = [c for c in commands if c.command == SemanticCommandType.DELETE]
    adds = [c for c in commands if c.command == SemanticCommandType.ADD]
    assert adds, f"expected at least one add, got {commands!r}"
    assert deletes, f"expected delete+add merge, got {commands!r}"

    merged = " ".join(a.value for a in adds).casefold()
    assert "pine" in merged or "task" in merged, merged
    assert "mike" in merged, merged


@_requires_openai
@pytest.mark.asyncio
async def test_ingest_merges_mike_then_peter(live_llm):
    """Mike then Peter: merge both people, do not replace Mike with only Peter."""
    existing = [
        _enthusiasm_tone(),
        _feature(
            tag="boundaries",
            feature_name="NOTIFICATION SCOPE",
            value="Notify about emails from Mike",
            feature_id="boundaries-notify",
        ),
    ]
    update_prompt = AgentPersonalitySemanticCategory.prompt.update_prompt
    commands = await llm_feature_update(
        features=existing,
        message_contents=["Notify about emails from Peter"],
        model=live_llm,
        update_prompt=update_prompt,
    )

    print("\n=== ingest Mike→Peter LLM return ===")
    pprint([c.model_dump() for c in commands])

    deletes = [c for c in commands if c.command == SemanticCommandType.DELETE]
    adds = [c for c in commands if c.command == SemanticCommandType.ADD]
    assert adds, f"expected at least one add, got {commands!r}"
    assert deletes, f"expected delete+add merge, got {commands!r}"
    # Must not silently drop Mike via add-only replace of Peter alone.
    assert not (
        len(adds) == 1
        and "peter" in adds[0].value.casefold()
        and "mike" not in adds[0].value.casefold()
        and not deletes
    ), f"replaced Mike with only Peter: {commands!r}"

    merged = " ".join(a.value for a in adds).casefold()
    assert "mike" in merged, merged
    assert "peter" in merged, merged


@_requires_openai
@pytest.mark.asyncio
async def test_ingest_exclusive_only_replaces_notification_scope(live_llm):
    """NEW exclusive 'only' may drop prior criteria."""
    existing = [
        _enthusiasm_tone(),
        _feature(
            tag="boundaries",
            feature_name="NOTIFICATION SCOPE",
            value="Notify about emails related to Pine tasks and emails from Mike",
            feature_id="boundaries-notify",
        ),
    ]
    update_prompt = AgentPersonalitySemanticCategory.prompt.update_prompt
    commands = await llm_feature_update(
        features=existing,
        message_contents=["Only notify about emails from Mike"],
        model=live_llm,
        update_prompt=update_prompt,
    )

    print("\n=== ingest exclusive-only LLM return ===")
    pprint([c.model_dump() for c in commands])

    adds = [c for c in commands if c.command == SemanticCommandType.ADD]
    assert adds, f"expected add, got {commands!r}"
    merged = " ".join(a.value for a in adds).casefold()
    assert "mike" in merged, merged
    # Exclusive only should not keep pine-task scope as a required criterion.
    assert "pine" not in merged, merged
