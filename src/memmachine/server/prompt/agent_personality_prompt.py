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

    ### Scope language (read literally)

    A NOTIFICATION SCOPE value that lists who/what to notify about is an INCLUDE
    list, not an exclusive filter — unless the value itself uses exclusive "only"
    (or clear cancel language).
    - "Notify about emails from Peter" = include Peter. Do NOT treat as only-Peter.
    - Never invent "only" into an existing scope when deciding MERGE vs REPLACE.

    For every claim, choose exactly one path:

    A) ADD NEW — different topic from all existing features
       → emit only `add` with a new feature_name

    B) MERGE — default for same-topic writes (NOTIFICATION SCOPE, etc.)
       → Combine old criteria + new criteria into ONE value (both kept)
       → Then `delete` the old feature_name and `add` the same feature_name with
         that combined value
       → Even when the OLD value said "only …", a later non-exclusive claim still
         MERGEs: expand the scope (keep old criteria + new criteria). This is NOT
         a conflict and NOT a replace-with-new-only.
       → Person A then person B (Mike then Peter) → always MERGE both

    C) REPLACE — only when the NEW claim itself is exclusive "only …", or clearly
       cancels / mutually excludes the old rule
       → `delete` the old feature, then `add` the new rule alone
       → Do not use C just because topics are related, or because people differ

    Notification-scope examples (feature = NOTIFICATION SCOPE):
    - MERGE (even old "only"): existing value="Only notify about emails related
      to Pine tasks" + new="Notify about emails from Mike"
      → keep Pine-task emails AND emails from Mike
    - MERGE: existing value="Notify about emails from Mike"
      + new="Notify about emails from Peter"
      → keep BOTH Mike and Peter (do NOT drop Mike)
    - MERGE: existing value="Notify about emails from Peter"
      + new="Notify about emails from Mike"
      → keep BOTH (Peter scope was include, not only-Peter)
    - REPLACE (NEW says exclusive only): existing value="Notify about emails
      from Mike and Pine tasks" + new="Only notify about emails from Mike"
      → value is only Mike

    Also:
    - Exact duplicate of an existing value → no write
    - For B and C: never leave the old same-topic feature behind; never emit only `add`
    - Ignore any generic guideline that says not to delete; B/C require delete+add
"""

MANUAL_INSTRUCTION_RULES = """
    ## DECISION ORDER (follow in order)

    1) SAFETY — if the instruction is sexual / violent / illegal (see SAFETY),
       reject (accepted=false). Stop.

    2) DUPLICATE — if the instruction matches an existing value (same meaning /
       same text), reject as duplicate. Stop.

    3) CLEAR LOGICAL CONFLICT — reject ONLY for a true cannot-both-be-true clash
       (literal exclusive "only" / cancel language in a VALUE).
       Scope is NEVER "only" and NEVER a conflict by itself:
       NOTIFICATION_SCOPE / "Notify about emails from …" means INCLUDE, not
       exclusive-only. Extending scope is always OK — ACCEPT and add a NEW
       feature_name (do not reject, do not invent "only").
       Example (must ACCEPT, new feature to extend scope):
       - existing NOTIFICATION_SCOPE="Notify about emails from Peter"
         + instruction="Notify about emails related to Pine tasks"
         → ACCEPT; e.g. feature_name="NOTIFICATION SCOPE PINE TASKS",
           value="Notify about emails related to Pine tasks"
       Same for Peter then Mike, or any extra include criterion.

    4) ACCEPT and APPEND — map to one tag, pick a NEW feature_name, write a
       short stable value. Never merge or overwrite existing rows.

    Reject if the instruction is not an agent behavior / preference setting
    (e.g. user personal facts, one-off tasks, unrelated content).

    ## OUTPUT SHAPE

    - One best tag: tone | persona | style | boundaries
    - NEW concise UPPERCASE feature_name (do not reuse an existing name)
    - Short stable value (not the raw instruction)
    - boundaries may hold many independent criteria, each its own feature_name
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
