"""Tests for manual semantic write helpers."""

import pytest

from memmachine.semantic_memory.semantic_manual_write import (
    build_manual_instruction_system_prompt,
    find_manual_write_duplicate,
    resolve_instruction_write_target,
    validate_manual_write_content,
)
from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.server.prompt.agent_personality_prompt import (
    MANUAL_INSTRUCTION_RULES,
    SAFETY_RULES,
)


def _feature(*, tag: str, feature_name: str, value: str) -> SemanticFeature:
    return SemanticFeature(
        metadata=SemanticFeature.Metadata(id="1"),
        category="agent_personality",
        tag=tag,
        feature_name=feature_name,
        value=value,
    )


def test_build_manual_instruction_system_prompt_loads_rules_from_prompt_file():
    prompt = build_manual_instruction_system_prompt(category_name="agent_personality")

    assert "tone" in prompt
    assert "persona" in prompt
    assert MANUAL_INSTRUCTION_RULES.strip() in prompt
    assert "Sexual or explicit adult content" in prompt
    assert "merge into the existing feature" in prompt


def test_validate_manual_write_content_uses_prompt_file_blocklist():
    with pytest.raises(ValueError, match="disallowed"):
        validate_manual_write_content(
            category_name="agent_personality",
            value="Use explicit sexual language",
        )


def test_resolve_instruction_write_target_reuses_existing_feature_name():
    existing = [_feature(tag="tone", feature_name="FORMALITY", value="Formal")]

    feature_name, value = resolve_instruction_write_target(
        existing_features=existing,
        tag="tone",
        feature_name="formality",
        value="Professional but friendly",
    )

    assert feature_name == "FORMALITY"
    assert value == "Professional but friendly"


def test_resolve_instruction_write_target_merges_duplicate_value():
    existing = [
        _feature(tag="style", feature_name="RESPONSE FORMAT", value="Use bullet points"),
    ]

    feature_name, value = resolve_instruction_write_target(
        existing_features=existing,
        tag="style",
        feature_name="BULLETS",
        value="Use bullet points",
    )

    assert feature_name == "RESPONSE FORMAT"
    assert value == "Use bullet points"


def test_find_manual_write_duplicate_still_detects_structured_conflicts():
    existing = [
        _feature(tag="tone", feature_name="FORMALITY", value="Professional but friendly"),
    ]

    duplicate = find_manual_write_duplicate(
        existing_features=existing,
        tag="tone",
        feature_name="WARMTH",
        value="Professional but friendly",
    )

    assert duplicate is not None
    assert "FORMALITY" in duplicate
