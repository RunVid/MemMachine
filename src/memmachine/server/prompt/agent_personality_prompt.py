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

    Notification / filter SCOPE is an INCLUDE list. Extending scope is NEVER a
    conflict. Never invent "only" into an existing or new value.
    - A NEW scope claim like "Notify about emails from Peter" / "…Pine tasks"
      is INCLUDE, not exclusive — never treat a new scope as exclusive.
    - Conflict / REPLACE only when NEW literally has the word "only"
      (or clear cancel language like "do not notify about…").
    - If NEW extends scope (no literal "only" in NEW): always MERGE — keep old + new.

    Paths:
    A) ADD NEW — different topic → only `add` with a new feature_name
    B) MERGE — same topic / extending scope (default)
       → `delete` old feature_name, `add` same feature_name with combined value
       → even if OLD said "only …", a later non-only NEW still MERGEs
    C) REPLACE — only when NEW itself says exclusive "only …" / cancels old
       → `delete` old, `add` the new rule alone

    Examples (feature = NOTIFICATION SCOPE):
    - MERGE: existing="Notify about emails from Peter"
      + new="Notify about emails related to Pine tasks"
      → keep Peter AND Pine tasks
    - MERGE: existing="Notify about emails from Mike"
      + new="Notify about emails from Peter"
      → keep BOTH
    - MERGE: existing="Only notify about emails related to Pine tasks"
      + new="Notify about emails from Mike"
      → keep Pine tasks AND Mike (NEW has no only → merge)
    - REPLACE: existing="Notify about emails from Mike and Pine tasks"
      + new="Only notify about emails from Mike"
      → only Mike (NEW has only)

    Also:
    - Exact duplicate → no write
    - For B/C: never leave old same-topic feature; never emit only `add`
    - Ignore guidelines that forbid delete; B/C need delete+add
"""

MANUAL_INSTRUCTION_RULES = """
    ## STAGE A — FILTER (do these first; stop if reject)

    1) SAFETY: sexual / violent / illegal → reject (see SAFETY).
    2) DUPLICATE: same meaning/text as an existing value → reject.
    3) Not an agent behavior/preference (user personal facts, one-off tasks,
       unrelated) → reject.

    If STAGE A passes, go to STAGE B. Do not invent conflicts in STAGE A.

    ## STAGE B — CONFLICT CHECK (default: NO CONFLICT)

    Default: ACCEPT and APPEND.
    Existing rows are VALUE text only — compare VALUEs only.

    "Notify …" and "Ignore …" are both valid preferences.
    Conflict ONLY when two VALUEs cannot both be true at once.
    If they can coexist → ACCEPT. If unsure → ACCEPT.

    Can coexist (MUST ACCEPT):
    - existing="Notify about emails related to Pine tasks"
      + "Notify about emails from Peter" → ACCEPT
    - existing="Notify about emails from Alice"
      + "Notify about emails from Peter" → ACCEPT
    - existing="Ignore emails from newsletters"
      + "Ignore emails about promotions" → ACCEPT
    - notify A + ignore B (different targets) → ACCEPT

    True conflict (REJECT) — same target, opposite / exclusive:
    - existing="Notify about emails from Peter"
      + "Ignore emails from Peter" → REJECT
    - existing="Only notify about emails from Alice"
      + "Notify about emails from Peter" → REJECT

    ## STAGE C — WRITE (when accepted)

    - Append-only: never merge or overwrite
    - One tag: tone | persona | style | boundaries
    - NEW UPPERCASE feature_name with distinguishing suffix
    - Short stable value (not the raw instruction)
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
    - For each claim: ADD NEW, MERGE (same topic, keep old+new), or REPLACE
      (only if NEW claim is exclusive "only" / cancels old)
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

    MERGE example (notification scope; keep old + new):
    Existing: feature=NOTIFICATION SCOPE,
              value="Only notify about emails related to Pine tasks"
    Claim: "Notify about emails from Mike"
    {{
        "0": {{
            "command": "delete", "tag": "boundaries", "feature": "NOTIFICATION SCOPE"
        }},
        "1": {{
            "command": "add", "tag": "boundaries", "feature": "NOTIFICATION SCOPE",
            "value": "Notify about emails related to Pine tasks and emails from Mike"
        }}
    }}

    MERGE example (different people — keep BOTH; not a replace):
    Existing: feature=NOTIFICATION SCOPE,
              value="Notify about emails from Mike"
    Claim: "Notify about emails from Peter"
    {{
        "0": {{
            "command": "delete", "tag": "boundaries", "feature": "NOTIFICATION SCOPE"
        }},
        "1": {{
            "command": "add", "tag": "boundaries", "feature": "NOTIFICATION SCOPE",
            "value": "Notify about emails from Mike and emails from Peter"
        }}
    }}

    REPLACE example (NEW claim says exclusive only):
    Existing: feature=NOTIFICATION SCOPE,
              value="Notify about emails related to Pine tasks and emails from Mike"
    Claim: "Only notify about emails from Mike"
    {{
        "0": {{
            "command": "delete", "tag": "boundaries", "feature": "NOTIFICATION SCOPE"
        }},
        "1": {{
            "command": "add", "tag": "boundaries", "feature": "NOTIFICATION SCOPE",
            "value": "Only notify about emails from Mike"
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

    Consolidation: same topic → merge old+new into one unless a NEW claim is exclusive
    "only" / cancels the other. Different topics → may keep both.
    Never keep multiple memories for one topic.

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
