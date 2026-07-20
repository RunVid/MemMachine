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

SAFETY_SKIP_ITEMS = """
    - Sexual or explicit adult content — skip, do not write
    - Violence, harm, or instructions to hurt people — skip, do not write
    - Illegal activity or instructions to break the law — skip, do not write
"""

# Used by manual-write prompt assembly and consolidation appendix.
SAFETY_RULES = f"""
    ## SAFETY (MANDATORY)

    Skip these; do not extract, store, update, or consolidate:
    {SAFETY_SKIP_ITEMS}
    If the input is only unsafe content, return no write / no add commands.
"""

MANUAL_WRITE_BLOCKED_CONTENT_PATTERNS: tuple[str, ...] = (
    r"\b(porn|pornography|sexual|explicit|nsfw)\b",
    r"\b(murder|kill people|violent|gore|torture)\b",
    r"\b(illegal|hack into|steal credentials|make bomb|child abuse)\b",
)

# Ingest + consolidation only (manual write stays append-only).
SAME_TOPIC_WRITE_RULES = """
    ## WRITE CHOICE (CRITICAL — OVERRIDES GENERIC GUIDELINES)

    For every claim, choose exactly one path:

    A) ADD NEW — different topic from all existing features
       → emit only `add` with a new feature_name

    B) MERGE — same topic, and both old and new can coexist
       (new extends / refines the old; e.g. old said "only X", new adds another
       criterion that can still hold together)
       → Merge means: combine old criteria + new criteria into ONE value, then
         replace the old row (`delete` old feature_name, `add` same feature_name
         with that combined value). Result is a single feature covering both.
       → Do not keep two features. Do not keep only the new clause.

    C) CONFLICT — same topic, new claim conflicts with / cancels the old rule
       → `delete` the old feature, then `add` the new rule alone
       → do not try to keep incompatible old criteria

    Also:
    - Exact duplicate of an existing value → no write
    - For B and C: never leave the old same-topic feature behind; never emit only `add`
    - Ignore any generic guideline that says not to delete; B/C require delete+add
"""

MANUAL_INSTRUCTION_RULES = """
    ## INSTRUCTION WRITE RULES

    - Manual writes are append-only: never update, merge, or overwrite existing features
    - Map the instruction to exactly one best matching tag
    - Always pick a NEW concise UPPERCASE feature name (do not reuse an existing name;
      if the natural name is taken, choose a distinct variant)
    - Store a short stable value describing the preference or rule, not the raw instruction
    - Compare carefully with existing features before deciding
    - Accept tone, persona, style, and boundaries; boundaries need not sound like
      personality traits

    ## BOUNDARIES TAG

    - Scope limits, filters, task/topic focus, notify/ignore rules, refusal patterns
    - Do NOT reject for lacking a personality trait

    ## CONFLICT AND DUPLICATE HANDLING

    - Reject only when the new instruction semantically conflicts with an existing rule,
      or exactly duplicates an existing value
    - Reusing / colliding with an existing feature_name is NOT a conflict — pick a new name
      and append instead
    - Do NOT merge or overwrite existing features on manual write
    - Set accepted=false with a clear rejection_reason for true conflicts / duplicates

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
    - For each claim: ADD NEW, MERGE (compatible same topic), or CONFLICT-replace
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

    Skip these; do not write any commands for them:
    - User personal facts
    - Claims unrelated to tone / persona / style / boundaries
    - One-off tasks with no lasting setting
    {SAFETY_SKIP_ITEMS}

    {SAME_TOPIC_WRITE_RULES}

    ADD NEW:
    {{"0": {{"command": "add", "tag": "<tag>", "feature": "<NEW>", "value": "<new>"}}}}

    MERGE (compatible; keep old + new):
    {{
        "0": {{"command": "delete", "tag": "<tag>", "feature": "<EXISTING>"}},
        "1": {{
            "command": "add", "tag": "<tag>", "feature": "<EXISTING>",
            "value": "<prior criteria + new criteria>"
        }}
    }}

    CONFLICT (delete old, keep only new):
    {{
        "0": {{"command": "delete", "tag": "<tag>", "feature": "<EXISTING>"}},
        "1": {{
            "command": "add", "tag": "<tag>", "feature": "<EXISTING>",
            "value": "<new rule only>"
        }}
    }}
"""

agent_personality_consolidation_prompt = (
    build_consolidation_prompt()
    + f"""

    ## AGENT PERSONALITY CONSOLIDATION

    All inputs share one tag; outputs must keep that tag.
    Allowed tags: tone, persona, style, boundaries (lowercase).

    {SAME_TOPIC_WRITE_RULES}

    Consolidation: compatible same-topic → merge into one; conflicting → keep winner.
    Different topics → may keep both. Never keep multiple memories for one topic.

    Skip / drop unsafe content (do not write):
    {SAFETY_SKIP_ITEMS}
    """
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
