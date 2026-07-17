"""Validation helpers for direct semantic memory writes."""

from __future__ import annotations

import re

from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.server.prompt.agent_personality_prompt import (
    AGENT_PERSONALITY_TAGS,
    get_category_manual_instruction_config,
)

_CATEGORY_TAG_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "agent_personality": AGENT_PERSONALITY_TAGS,
}


def _compile_blocked_patterns(category_name: str) -> tuple[re.Pattern[str], ...]:
    config = get_category_manual_instruction_config(category_name)
    raw_patterns = config.get("blocked_patterns")
    if not isinstance(raw_patterns, tuple):
        return ()
    return tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in raw_patterns
        if isinstance(pattern, str)
    )


def normalize_manual_write_text(text: str) -> str:
    return " ".join(text.casefold().split())


def get_allowed_tags_for_category(category_name: str) -> set[str] | None:
    tag_descriptions = _CATEGORY_TAG_DESCRIPTIONS.get(category_name)
    if tag_descriptions is None:
        return None
    return set(tag_descriptions.keys())


def build_manual_instruction_system_prompt(*, category_name: str) -> str:
    config = get_category_manual_instruction_config(category_name)
    tag_descriptions = config["tags"]
    if not isinstance(tag_descriptions, dict):
        raise ValueError(
            f"Instruction writes are not supported for category '{category_name}'",
        )

    instruction_rules = config.get("instruction_rules", "")
    safety_rules = config.get("safety_rules", "")
    if not isinstance(instruction_rules, str) or not isinstance(safety_rules, str):
        raise ValueError(
            f"Invalid instruction prompt config for category '{category_name}'",
        )

    tag_lines = "\n".join(
        f"- {tag}: {description}" for tag, description in tag_descriptions.items()
    )
    allowed = ", ".join(sorted(tag_descriptions.keys()))
    return f"""
You convert a user's free-form instruction into one structured semantic memory feature.

Category: {category_name}

Allowed tags:
{tag_lines}

{instruction_rules}

{safety_rules}

Return JSON:
{{
  "accepted": true,
  "rejection_reason": "",
  "tag": "tone",
  "feature_name": "FORMALITY",
  "value": "Professional but friendly"
}}

If rejected for safety, category mismatch, or irreconcilable conflict:
{{
  "accepted": false,
  "rejection_reason": "reason",
  "tag": "",
  "feature_name": "",
  "value": ""
}}

Choose exactly one best tag from: {allowed}
"""


def validate_manual_write_tag(*, category_name: str, tag: str) -> None:
    allowed_tags = get_allowed_tags_for_category(category_name)
    if allowed_tags is None:
        return
    normalized_tag = tag.strip().lower()
    if normalized_tag not in allowed_tags:
        allowed = ", ".join(sorted(allowed_tags))
        raise ValueError(
            f"Invalid tag '{tag}' for category '{category_name}'. "
            f"Allowed tags: {allowed}",
        )


def validate_manual_write_content(*, category_name: str, value: str) -> None:
    normalized_value = value.strip()
    if normalized_value == "":
        raise ValueError("Semantic memory value cannot be empty")

    for pattern in _compile_blocked_patterns(category_name):
        if pattern.search(normalized_value):
            raise ValueError(
                "Semantic memory value contains disallowed sexual, violent, or illegal content",
            )


def find_manual_write_duplicate(
    *,
    existing_features: list[SemanticFeature],
    tag: str,
    feature_name: str,
    value: str,
) -> str | None:
    normalized_tag = tag.strip().lower()
    normalized_feature = normalize_manual_write_text(feature_name)
    normalized_value = normalize_manual_write_text(value)

    for feature in existing_features:
        if feature.tag.strip().lower() != normalized_tag:
            continue

        existing_feature = normalize_manual_write_text(feature.feature_name)
        existing_value = normalize_manual_write_text(feature.value)

        if existing_feature == normalized_feature:
            continue

        if existing_value == normalized_value:
            return (
                f"Duplicate value already exists under feature "
                f"'{feature.feature_name}' in tag '{feature.tag}'"
            )

    return None


def resolve_instruction_write_target(
    *,
    existing_features: list[SemanticFeature],
    tag: str,
    feature_name: str,
    value: str,
) -> tuple[str, str]:
    """Return the canonical feature name and value for an instruction upsert."""
    normalized_tag = tag.strip().lower()
    normalized_feature = normalize_manual_write_text(feature_name)
    normalized_value = normalize_manual_write_text(value)

    for feature in existing_features:
        if feature.tag.strip().lower() != normalized_tag:
            continue
        if normalize_manual_write_text(feature.feature_name) == normalized_feature:
            return feature.feature_name, value

    for feature in existing_features:
        if feature.tag.strip().lower() != normalized_tag:
            continue
        if normalize_manual_write_text(feature.feature_name) == normalized_feature:
            continue
        if normalize_manual_write_text(feature.value) == normalized_value:
            return feature.feature_name, value

    return feature_name, value
