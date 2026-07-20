"""Tests for manual semantic write helpers."""

import pytest

from memmachine.semantic_memory.semantic_manual_write import (
    build_manual_instruction_system_prompt,
    normalize_manual_write_tag,
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
    assert "append-only" in prompt
    assert "Do NOT merge or overwrite" in prompt
    assert "Do NOT reject for lacking a personality trait" in prompt


def test_validate_manual_write_content_uses_prompt_file_blocklist():
    with pytest.raises(ValueError, match="Validation error:"):
        validate_manual_write_content(
            category_name="agent_personality",
            value="Use explicit sexual language",
        )


def test_validate_manual_write_append_rejects_existing_feature_name():
    existing = [_feature(tag="tone", feature_name="FORMALITY", value="Formal")]

    error = validate_manual_write_append(
        existing_features=existing,
        tag="tone",
        feature_name="formality",
        value="Professional but friendly",
    )

    assert error is not None
    assert error.startswith("Conflict:")
    assert "FORMALITY" in error


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
