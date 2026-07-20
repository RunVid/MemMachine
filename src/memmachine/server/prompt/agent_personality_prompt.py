"""Prompt template for agent personality semantic memory."""

from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    SemanticCategory,
)
from memmachine.semantic_memory.util.semantic_prompt_template import (
    build_consolidation_prompt,
    build_update_prompt,
)

AGENT_PERSONALITY_TAGS: dict[str, str] = {
    "tone": "How the agent sounds: formality, warmth, directness, and emotional quality.",
    "persona": "Who the agent is: role, identity, character, and archetype.",
    "style": "How the agent responds: length, structure, formatting, and interaction habits.",
    "boundaries": (
        "Agent scope limits, notification filters, task/topic focus, and refusal patterns. "
        "Includes what to notify about, ignore, prioritize, or decline. "
        "Not for overriding global safety rules."
    ),
}

SAFETY_RULES = """
    ## SAFETY (MANDATORY)

    NEVER extract, store, update, or consolidate memories that involve:
    - Sexual or explicit adult content
    - Violence, harm, or instructions to hurt people
    - Illegal activity or instructions to break the law

    If the claim contains only unsafe content, return no add commands.
"""

MANUAL_WRITE_BLOCKED_CONTENT_PATTERNS: tuple[str, ...] = (
    r"\b(porn|pornography|sexual|explicit|nsfw)\b",
    r"\b(murder|kill people|violent|gore|torture)\b",
    r"\b(illegal|hack into|steal credentials|make bomb|child abuse)\b",
)

# Ingest + consolidation only (manual write stays append-only).
SAME_TOPIC_WRITE_RULES = """
    ## SAME-TOPIC LOGIC (CRITICAL — OVERRIDES GENERIC GUIDELINES)

    One topic → exactly one feature. Never leave an old same-topic feature behind.

    Decision:
    1) Exact duplicate of an existing value → no write
    2) Same topic, new claim extends or updates it → REPLACE that feature:
       delete the existing feature, then add one feature with the SAME feature_name
       and a value that keeps prior criteria and includes the new ones
    3) Same topic, NEW claim is exclusive "only …" or clearly conflicts → REPLACE:
       delete existing, add the new rule alone
    4) Different topic → add a new feature_name (do not touch unrelated features)

    Notes:
    - Old values that say "only …" can still be extended later; merge both criteria
    - Reuse the exact existing feature_name for same-topic writes
    - For ingest commands: same-topic change = delete THEN add (both required).
      A lone add that leaves the old feature is WRONG.
    - Ignore any generic guideline that says not to delete: for this category,
      delete+add is the required way to extend/update/replace a feature
"""

MANUAL_INSTRUCTION_RULES = """
    ## INSTRUCTION WRITE RULES

    - Manual writes are append-only: never update, merge, or reuse existing feature names
    - Map the instruction to exactly one best matching tag
    - Pick a new concise UPPERCASE feature name for each accepted instruction
    - Store a short stable value describing the preference or rule, not the raw instruction
    - Compare carefully with existing features before deciding
    - Accept tone, persona, style, and boundaries; boundaries need not sound like
      personality traits

    ## BOUNDARIES TAG

    - Scope limits, filters, task/topic focus, notify/ignore rules, refusal patterns
    - Do NOT reject for lacking a personality trait

    ## CONFLICT AND DUPLICATE HANDLING

    - Reject when the instruction would duplicate an existing value under the same tag
    - Reject when the proposed feature name already exists under the same tag
    - Reject when the instruction overlaps or conflicts with an existing rule
    - Set accepted=false with a clear rejection_reason for duplicates and conflicts
    - Do NOT merge or overwrite existing features on manual write

    ## CATEGORY REJECTION

    - Accept agent behavior / communication / scope instructions
    - Reject unrelated content, or sexual / violent / illegal content per SAFETY
"""

CATEGORY_MANUAL_INSTRUCTION_CONFIG: dict[str, dict[str, object]] = {
    "agent_personality": {
        "tags": AGENT_PERSONALITY_TAGS,
        "instruction_rules": MANUAL_INSTRUCTION_RULES,
        "safety_rules": SAFETY_RULES,
        "blocked_patterns": MANUAL_WRITE_BLOCKED_CONTENT_PATTERNS,
    },
}


def get_category_manual_instruction_config(category_name: str) -> dict[str, object]:
    config = CATEGORY_MANUAL_INSTRUCTION_CONFIG.get(category_name)
    if config is None:
        raise ValueError(
            f"Instruction writes are not supported for category '{category_name}'",
        )
    return config


AGENT_PERSONALITY_DESCRIPTION = f"""
    You extract stable AGENT personality / behavior settings from claims.
    Input is a claim (a stated agent setting or preference), not a chat conversation.
    These memories configure how the agent should sound, who it is, how it responds,
    and what it should or should not do. They are NOT user profile facts.

    ## YOUR ROLE

    - Store reusable agent settings that persist across sessions
    - Keep one clean feature per topic: extend/update/replace via delete+add
    - Ignore user personal facts and one-off task requests

    ALWAYS compare with existing features before writing.

    ## TAG RULES

    Use only the tags listed below (tone, persona, style, boundaries).
    - DO NOT create new tags — pick the closest match
    - Tags MUST be lowercase
    - If unsure: identity/role → persona; sound → tone; format/habits → style;
      scope/filters/refusals → boundaries

    ## WHAT TO EXTRACT

    Stable agent settings only. Tag meanings are in the tag list below.
    Feature names: concise UPPERCASE with spaces.
    Values: short stable preference/rule text.

    ## WHAT NOT TO EXTRACT

    - User personal facts
    - Claims unrelated to tone / persona / style / boundaries
    - One-off tasks with no lasting setting
    - Content blocked by SAFETY below

    {SAME_TOPIC_WRITE_RULES}

    Required command shape for same-topic extend/update/replace:
    {{
        "0": {{"command": "delete", "tag": "<tag>", "feature": "<EXISTING FEATURE>"}},
        "1": {{
            "command": "add",
            "tag": "<tag>",
            "feature": "<EXISTING FEATURE>",
            "value": "<result per decision rules>"
        }}
    }}
    Never emit only the add for a same-topic change.

    {SAFETY_RULES}
"""

agent_personality_consolidation_prompt = (
    build_consolidation_prompt()
    + f"""

    ## AGENT PERSONALITY CONSOLIDATION

    All inputs share one tag; outputs must keep that tag.
    Allowed tags: tone, persona, style, boundaries (lowercase).

    {SAME_TOPIC_WRITE_RULES}

    Consolidation means: for each topic, at most one surviving memory.
    - Same topic → one consolidated memory; do NOT keep old ids in keep_memories
    - Exclusive "only" / clear conflict → one winning memory; drop the rest
    - Different topics → may keep both
    - Never keep multiple memories for the same topic
    """
    + SAFETY_RULES
)

AgentPersonalitySemanticCategory = SemanticCategory(
    name="agent_personality",
    prompt=RawSemanticPrompt(
        update_prompt=build_update_prompt(
            tags=AGENT_PERSONALITY_TAGS,
            description=AGENT_PERSONALITY_DESCRIPTION,
        ),
        consolidation_prompt=agent_personality_consolidation_prompt,
    ),
)

SEMANTIC_TYPE = AgentPersonalitySemanticCategory
