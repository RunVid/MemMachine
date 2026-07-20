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

# Shared merge rules for ingest + manual write.
COMPLEMENTARY_MERGE_RULES = """
    ## SAME-TOPIC WRITE (CRITICAL)

    Same topic → one feature. Prefer extend or update the existing value; do not
    create a sibling feature, and do not wipe prior criteria by default.

    - Extend / update: new claim adds or refines the same topic → keep prior
      criteria and incorporate the new ones into one value (reuse feature_name)
    - An existing value that already says "only …" does NOT block later extend:
      if a later claim adds another criterion on the same topic, MERGE both
      (keep the old criterion and the new one)
    - Replace only when the NEW claim itself is exclusive "only …" that clearly
      drops prior criteria, or clearly conflicts with / cancels the previous rule
    - Unrelated topics (different axes) may be separate features
    - Exact duplicate value → skip / reject
"""MANUAL_INSTRUCTION_RULES = f"""
    ## INSTRUCTION WRITE RULES

    - Map the instruction to exactly one best matching tag
    - Prefer stable UPPERCASE feature names
    - Store a short stable value describing the preference or rule, not the raw instruction
    - Compare carefully with existing features before deciding
    - Accept tone, persona, style, and boundaries instructions; boundaries do not need to
      sound like a personality trait

    ## BOUNDARIES TAG

    - Use `boundaries` for scope limits, filters, task/topic focus, notify/ignore rules,
      and refusal patterns
    - Do NOT reject a boundaries instruction for lacking a personality trait

    {COMPLEMENTARY_MERGE_RULES}

    When extending / updating the same topic:
    - accepted=true
    - reuse the existing feature_name
    - value keeps prior criteria and includes the new ones

    ## CONFLICT AND DUPLICATE HANDLING

    - Reject (accepted=false) only for:
      - exact duplicate of an existing value
      - clear conflicts that cannot be reconciled
      - safety / category violations
    - Same-topic extend/update → accept into the existing feature
    - Do NOT invent a second feature name for the same topic
    - Do NOT replace with the new clause alone unless the NEW claim is exclusive
      "only" or conflicts with the previous rule
    - Prior text containing "only" is still extendable by later same-topic claims

    ## CATEGORY REJECTION

    - Accept any instruction about how the agent should behave, respond, communicate, or
      scope its actions (tone, persona, style, boundaries)
    - Reject only content that is unrelated to agent behavior, or that involves sexual,
      violent, or illegal content per the SAFETY rules
    - Do not reject valid agent-behavior instructions for being too specific, operational,
      or not sounding like a personality trait
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

    - Store reusable agent settings that should persist across sessions
    - Prefer a clean, non-duplicative profile: update or delete before adding
    - Ignore claims that are user personal facts or not about agent behavior

    ALWAYS compare with existing features before creating new ones.

    ## TAG RULES

    Use only the tags listed below in this prompt (tone, persona, style, boundaries).
    - DO NOT create new tags — pick the closest match
    - Tags MUST be lowercase
    - If unsure between tags, prefer:
      - identity / character / role → persona
      - how it sounds → tone
      - response format / length / habits → style
      - scope, filters, notify/ignore, refusals → boundaries

    ## WHAT TO EXTRACT

    Stable, reusable agent settings from the claim only. Tag meanings are defined
    in the tag list below — do not invent other categories.

    Feature names: concise UPPERCASE with spaces.
    Values: short stable preference/rule text — not a verbatim dump of the claim
    when a clearer paraphrase exists.

    ## WHAT NOT TO EXTRACT

    - User personal facts (name, job, preferences about the user themselves)
    - Claims that are not about agent tone, persona, style, or boundaries
    - One-off task requests with no lasting agent setting
    - Content blocked by SAFETY rules below

    {COMPLEMENTARY_MERGE_RULES}

    ## UPDATE WORKFLOW

    1. Read existing features for the relevant tag
    2. Exact duplicate → do nothing
    3. Same topic → extend/update: delete old, add one feature with same feature_name
       whose value keeps prior criteria and includes the new ones (even if the old
       value said "only …")
    4. NEW claim is exclusive "only …" that drops prior criteria, or clear conflict
       → replace with the new rule
    5. Unrelated topic → add with a new feature name
    6. Never leave two features for the same topic side by side

    Same-topic extend/update shape:
    {{
        "0": {{"command": "delete", "tag": "<tag>", "feature": "<EXISTING FEATURE>"}},
        "1": {{
            "command": "add",
            "tag": "<tag>",
            "feature": "<EXISTING FEATURE>",
            "value": "<prior criteria kept + new criteria>"
        }}
    }}

    {SAFETY_RULES}
"""

agent_personality_consolidation_prompt = (
    build_consolidation_prompt()
    + f"""

    ## AGENT PERSONALITY CONSOLIDATION

    All input memories share the same tag. Outputs MUST keep that same tag.
    Allowed tags only: tone, persona, style, boundaries (lowercase).

    ### Goal
    Keep a small, clear set of stable agent settings. Remove redundancy and
    extend/update same-topic rules into one feature.

    {COMPLEMENTARY_MERGE_RULES}

    ### Workflow

    Step 1: DELETE first (exclude id from keep_memories)
    - Exact or near-duplicate values → keep one, drop rest
    - Vague / unusable entries → delete
    - Unsafe content → delete

    Step 2: Merge within the same tag
    - Same topic → one memory; value extends/updates prior criteria (keep old + new),
      including when an older memory said "only …"
    - NEW exclusive "only" or clear conflict → keep the winning rule only
    - Unrelated topics → keep both

    Step 3: Do NOT
    - Create new tags
    - Merge across different tags
    - Invent user-profile facts
    - Keep multiple memories for the same topic side by side
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
