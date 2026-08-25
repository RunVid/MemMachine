"""Closed tag sets shared by extraction, consolidation, and manual writes.

Keys are ``SemanticCategory.name`` (not prompt file names):

- ``profile`` — user profile / task-assistant facts (profile memory)
- ``profile_life_context`` — user life-context (profile memory)
- ``agent_personality`` — agent behavior settings (role memory)

``default_tag`` is the grouping key when the LLM invents an unknown tag, so
consolidation does not fragment the same topic across extra tags.
"""

from dataclasses import dataclass

from memmachine.server.prompt.agent_personality_prompt import AGENT_PERSONALITY_TAGS
from memmachine.server.prompt.life_context_prompt import life_context_tags
from memmachine.server.prompt.task_assistant_prompt import task_assistant_tags


@dataclass(frozen=True)
class CategoryTagPolicy:
    """Allowed tags and fallback tag for a closed semantic category."""

    allowed_tags: frozenset[str]
    default_tag: str


# Keep in sync with the tag dicts in the prompt modules above.
CATEGORY_TAG_POLICIES: dict[str, CategoryTagPolicy] = {
    "profile": CategoryTagPolicy(
        allowed_tags=frozenset(task_assistant_tags),
        default_tag="others",
    ),
    "profile_life_context": CategoryTagPolicy(
        allowed_tags=frozenset(life_context_tags),
        default_tag="interests",
    ),
    "agent_personality": CategoryTagPolicy(
        allowed_tags=frozenset(AGENT_PERSONALITY_TAGS),
        default_tag="style",
    ),
}


def get_category_tag_policy(category_name: str) -> CategoryTagPolicy | None:
    """Return the closed tag policy, or None if the category is open-ended."""
    return CATEGORY_TAG_POLICIES.get(category_name)


def get_allowed_tags_for_category(category_name: str) -> set[str] | None:
    """Allowed tags for a closed category, or None if any tag is accepted."""
    policy = get_category_tag_policy(category_name)
    if policy is None:
        return None
    return set(policy.allowed_tags)


def normalize_category_tag(category_name: str, tag: str) -> str:
    """
    Lowercase and strip ``tag``.

    Closed categories remap unknown tags to ``default_tag`` so extract, ingest
    commands, and manual writes share one grouping key. Open categories are
    only lowercased.
    """
    normalized_tag = tag.strip().lower()
    policy = get_category_tag_policy(category_name)
    if policy is None:
        return normalized_tag
    if normalized_tag in policy.allowed_tags:
        return normalized_tag
    return policy.default_tag
