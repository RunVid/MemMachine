"""Validation helpers for direct semantic memory writes."""

from __future__ import annotations

import re

from memmachine.semantic_memory.category_tags import normalize_category_tag
from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.server.prompt.agent_personality_prompt import (
    get_category_manual_instruction_config,
)


def _compile_blocked_patterns(category_name: str) -> tuple[re.Pattern[str], ...]:
    """Compile blocked-content regexes for a category."""
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
    """Collapse whitespace and case-fold instruction text."""
    return " ".join(text.casefold().split())


def build_manual_instruction_system_prompt(*, category_name: str) -> str:
    """Build the system prompt for instruction-to-feature conversion."""
    config = get_category_manual_instruction_config(category_name)
    tag_descriptions = config["tags"]
    if not isinstance(tag_descriptions, dict):
        raise ValueError(  # noqa: TRY004
            f"Instruction writes are not supported for category '{category_name}'",
        )

    instruction_rules = config.get("instruction_rules", "")
    safety_rules = config.get("safety_rules", "")
    if not isinstance(instruction_rules, str) or not isinstance(safety_rules, str):
        raise ValueError(  # noqa: TRY004
            f"Invalid instruction prompt config for category '{category_name}'",
        )

    tags = {str(tag): str(description) for tag, description in tag_descriptions.items()}
    tag_lines = "\n".join(
        f"- {tag}: {description}" for tag, description in tags.items()
    )
    allowed = ", ".join(sorted(tags))
    return f"""
You convert a user's free-form instruction into one structured semantic memory feature.

Category: {category_name}

Allowed tags:
{tag_lines}

{instruction_rules}

{safety_rules}

Accept example (existing value="Notify about emails related to Pine tasks";
new Peter — NOT conflict):
{{
  "accepted": true,
  "rejection_reason": "",
  "tag": "boundaries",
  "feature_name": "NOTIFICATION SCOPE PETER",
  "value": "Notify about emails from Peter"
}}

Reject example:
{{
  "accepted": false,
  "rejection_reason": "reason",
  "tag": "",
  "feature_name": "",
  "value": ""
}}

Choose exactly one best tag from: {allowed}
"""


def normalize_manual_write_tag(*, category_name: str, tag: str) -> str:
    """Normalize a tag onto the shared category tag policy."""
    return normalize_category_tag(category_name, tag)


def validate_manual_write_tag(*, category_name: str, tag: str) -> None:
    """Validate a write tag against the category policy."""
    normalize_manual_write_tag(category_name=category_name, tag=tag)


def validate_manual_write_content(*, category_name: str, value: str) -> None:
    """Reject empty or blocked instruction content."""
    normalized_value = value.strip()
    if normalized_value == "":
        raise ValueError("Validation error: semantic memory value cannot be empty")

    for pattern in _compile_blocked_patterns(category_name):
        if pattern.search(normalized_value):
            raise ValueError(
                "Validation error: value contains disallowed sexual, violent, or illegal content",
            )


def validate_manual_write_append(
    *,
    existing_features: list[SemanticFeature],
    tag: str,
    feature_name: str,
    value: str,
) -> str | None:
    """
    Return a duplicate message, or None if the write may proceed.

    Same feature_name is not a hard conflict; names are uniquified with a suffix
    before append. Only exact duplicate values are rejected here.
    Semantic conflicts are rejected by the LLM (accepted=false).
    """
    _ = feature_name
    normalized_tag = tag.strip().lower()
    normalized_value = normalize_manual_write_text(value)

    for feature in existing_features:
        if feature.tag.strip().lower() != normalized_tag:
            continue

        existing_value = normalize_manual_write_text(feature.value)
        if existing_value == normalized_value:
            return (
                f"Duplicate: value already exists under feature "
                f"'{feature.feature_name}' in tag '{feature.tag}'"
            )

    return None


def _normalize_feature_name(feature_name: str) -> str:
    """Uppercase and collapse whitespace in a feature name."""
    return " ".join(feature_name.strip().upper().split())


def unique_manual_feature_name(
    *,
    existing_features: list[SemanticFeature],
    tag: str,
    feature_name: str,
) -> str:
    """
    Ensure feature_name is unique under tag by appending a numeric suffix.

    Manual writes always append; colliding names get `` 2``, `` 3``, …
    """
    base = _normalize_feature_name(feature_name)
    if base == "":
        return base

    normalized_tag = tag.strip().lower()
    existing_names = {
        _normalize_feature_name(feature.feature_name)
        for feature in existing_features
        if feature.tag.strip().lower() == normalized_tag
    }
    if base not in existing_names:
        return base

    suffix = 2
    while f"{base} {suffix}" in existing_names:
        suffix += 1
    return f"{base} {suffix}"
