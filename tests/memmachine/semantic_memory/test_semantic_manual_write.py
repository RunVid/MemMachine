"""Tests for manual semantic write helpers."""

import pytest

from memmachine.semantic_memory.semantic_manual_write import (
    build_manual_instruction_system_prompt,
    is_false_manual_conflict_rejection,
    normalize_manual_write_tag,
    unique_manual_feature_name,
    validate_manual_write_append,
    validate_manual_write_content,
)
from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.server.prompt.agent_personality_prompt import (
    MANUAL_INSTRUCTION_RULES,
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
    assert "STAGE A" in prompt
    assert "STAGE B" in prompt
    assert "default: NO CONFLICT" in prompt
    assert "VALUE text only" in prompt
    assert "Notify about emails related to Pine tasks" in prompt
    assert "cannot both be true" in prompt
    assert "Feature_name naming" in prompt
    assert "NOTIFICATION SCOPE PETER" in prompt


def test_validate_manual_write_content_uses_prompt_file_blocklist():
    with pytest.raises(ValueError, match="Validation error:"):
        validate_manual_write_content(
            category_name="agent_personality",
            value="Use explicit sexual language",
        )


def test_validate_manual_write_append_allows_same_feature_name_with_new_value():
    existing = [_feature(tag="tone", feature_name="FORMALITY", value="Formal")]

    error = validate_manual_write_append(
        existing_features=existing,
        tag="tone",
        feature_name="formality",
        value="Professional but friendly",
    )

    assert error is None


def test_validate_manual_write_append_rejects_duplicate_value():
    existing = [
        _feature(tag="style", feature_name="RESPONSE FORMAT", value="Use bullet points"),
    ]

    error = validate_manual_write_append(
        existing_features=existing,
        tag="style",
        feature_name="BULLETS",
        value="Use bullet points",
    )

    assert error is not None
    assert error.startswith("Duplicate:")
    assert "RESPONSE FORMAT" in error


def test_validate_manual_write_append_allows_new_feature():
    existing = [
        _feature(tag="tone", feature_name="FORMALITY", value="Formal"),
    ]

    error = validate_manual_write_append(
        existing_features=existing,
        tag="tone",
        feature_name="WARMTH",
        value="Friendly and approachable",
    )

    assert error is None


def test_normalize_manual_write_tag_lowercases_valid_tag():
    assert (
        normalize_manual_write_tag(category_name="agent_personality", tag=" Persona ")
        == "persona"
    )


def test_normalize_manual_write_tag_defaults_unknown_to_style():
    assert (
        normalize_manual_write_tag(
            category_name="agent_personality",
            tag="completely_unrelated",
        )
        == "style"
    )


def test_unique_manual_feature_name_keeps_unused_name():
    existing = [
        _feature(
            tag="boundaries",
            feature_name="NOTIFICATION_SCOPE",
            value="Notify about emails from Peter",
        ),
    ]

    assert (
        unique_manual_feature_name(
            existing_features=existing,
            tag="boundaries",
            feature_name="NOTIFICATION SCOPE PINE TASKS",
        )
        == "NOTIFICATION SCOPE PINE TASKS"
    )


def test_unique_manual_feature_name_adds_suffix_on_collision():
    existing = [
        _feature(
            tag="boundaries",
            feature_name="NOTIFICATION SCOPE",
            value="Notify about emails from Peter",
        ),
    ]

    assert (
        unique_manual_feature_name(
            existing_features=existing,
            tag="boundaries",
            feature_name="notification scope",
        )
        == "NOTIFICATION SCOPE 2"
    )
