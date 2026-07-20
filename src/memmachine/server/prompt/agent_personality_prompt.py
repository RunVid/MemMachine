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

    If the conversation contains only unsafe content, return no add commands.
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
    You extract stable agent personality traits from conversations.
    Store only reusable persona settings for the agent, not one-off chat lines or user facts.

    ALWAYS compare with existing features before adding new ones.
    Prefer updating or deleting existing features over creating duplicates.

    ## TAG RULES

    You MUST ONLY use: tone, persona, style, boundaries
    - DO NOT create new tags
    - Tags are lowercase and case-sensitive

    {SAFETY_RULES}
"""

agent_personality_consolidation_prompt = (
    build_consolidation_prompt()
    + """

    ## AGENT PERSONALITY RULES

    - Merge overlapping persona traits within the same tag
    - Delete duplicates and near-duplicates; keep the clearest feature
    - Do not keep redundant memories that express the same trait
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
