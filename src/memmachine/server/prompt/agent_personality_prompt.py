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

MANUAL_INSTRUCTION_RULES = """
    ## INSTRUCTION WRITE RULES

    - Manual writes are append-only: never update, merge, or reuse existing feature names
    - Map the instruction to exactly one best matching tag
    - Pick a new concise UPPERCASE feature name for each accepted instruction
    - Store a short stable value describing the preference or rule, not the raw instruction text
    - Compare carefully with existing features before deciding placement
    - Accept tone, persona, style, and boundaries instructions; boundaries do not need to sound
      like a personality trait

    ## BOUNDARIES TAG

    - Use `boundaries` for scope limits, notification filters, task/topic focus, and what to
      include, exclude, notify about, or ignore
    - Accept instructions such as:
      - "Only notify me about emails related to Pine tasks"
      - "Ignore promotional emails unless I ask"
      - "Do not summarize attachments unless requested"
    - Do NOT reject a boundaries instruction for lacking a personality trait; scope and
      filtering rules are valid boundaries
    - Example mapping:
      instruction -> tag=boundaries, feature_name=NOTIFICATION_SCOPE,
      value="Only notify about emails related to Pine tasks"

    ## CONFLICT AND DUPLICATE HANDLING

    - Reject when the instruction would duplicate an existing value under the same tag
    - Reject when the proposed feature name already exists under the same tag
    - Reject when the instruction overlaps or conflicts with an existing rule in the same tag
    - Set accepted=false and provide a clear rejection_reason for duplicates and conflicts
    - Reject only when the instruction violates safety rules (see SAFETY section)

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

    Examples:
    - "Be warm but concise" → tag=tone, feature=WARMTH, value="Warm but concise"
    - "You are a helpful copilot for Pine" → tag=persona, feature=ROLE,
      value="Helpful copilot for Pine"
    - "Prefer bullet points" → tag=style, feature=RESPONSE FORMAT,
      value="Prefer bullet points"
    - "Only notify me about emails related to Pine tasks" → tag=boundaries,
      feature=NOTIFICATION_SCOPE,
      value="Only notify about emails related to Pine tasks"

    Feature names: concise UPPERCASE with spaces (e.g., "RESPONSE FORMAT").
    Values: short stable preference/rule text — not a verbatim dump of the claim
    when a clearer paraphrase exists.

    ## WHAT NOT TO EXTRACT

    - User personal facts (name, job, preferences about the user themselves)
    - Claims that are not about agent tone, persona, style, or boundaries
    - Content blocked by SAFETY rules below

    Examples of INCORRECT extraction (DO NOT DO THIS):
    - "My name is Alice" → user fact, not agent personality
    - "Summarize this email now" → one-off task claim, not a lasting setting

    ## UPDATE WORKFLOW

    1. Read existing features for the relevant tag
    2. If the claim duplicates an existing value → do nothing (no add)
    3. If it conflicts with or supersedes an existing feature → delete old, then add new
    4. If it is a distinct new setting → add with a new feature name
    5. Prefer delete+add over stacking near-duplicates under the same tag

    Update example (tone becomes less formal):
    {{
        "0": {{
            "command": "delete",
            "tag": "tone",
            "feature": "FORMALITY"
        }},
        "1": {{
            "command": "add",
            "tag": "tone",
            "feature": "FORMALITY",
            "value": "Casual and friendly"
        }}
    }}

    {SAFETY_RULES}
"""

agent_personality_consolidation_prompt = (
    build_consolidation_prompt()
    + """

    ## AGENT PERSONALITY CONSOLIDATION

    All input memories share the same tag. Outputs MUST keep that same tag.
    Allowed tags only: tone, persona, style, boundaries (lowercase).

    ### Goal
    Keep a small, clear set of stable agent settings. Remove redundancy and
    near-duplicates so the agent profile stays easy to apply.

    ### Workflow

    Step 1: DELETE first (exclude id from keep_memories)
    - Exact or near-duplicate values under the same feature → keep one, drop rest
    - Vague / unusable entries → delete
    - Unsafe content (sexual, violent, illegal) → delete

    Step 2: Merge within the same tag
    - Same meaning, different wording → delete sources, create one clear consolidated memory
    - Same feature name, conflicting values → keep the newer/clearer rule; drop the rest
    - Distinct settings (e.g. NOTIFICATION_SCOPE vs REFUSAL_POLICY) → keep both

    Step 3: Do NOT
    - Create new tags
    - Merge across different tags
    - Invent user-profile facts during consolidation
    - Keep redundant memories that express the same trait

    Example (near-duplicates → one memory):
    - id=1 feature=WARMTH value="Be warm"
    - id=2 feature=WARMTH value="Warm and friendly tone"
    → keep_memories=[], consolidated_memories=[
        {"tag": "tone", "feature": "WARMTH", "value": "Warm and friendly"}
      ]

    Example (distinct boundaries → keep both):
    - id=3 feature=NOTIFICATION_SCOPE value="Only Pine-related emails"
    - id=4 feature=REFUSAL_POLICY value="Decline medical diagnosis requests"
    → keep_memories=["3", "4"], consolidated_memories=[]
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
