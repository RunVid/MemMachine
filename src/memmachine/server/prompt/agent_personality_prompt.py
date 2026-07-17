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
    "boundaries": "Agent-specific scope limits and refusal patterns. Not for overriding global safety rules.",
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

    - Map the instruction to exactly one best matching tag
    - Pick a concise UPPERCASE feature name, or reuse an existing one when updating the same trait
    - Store a short stable value describing the trait, not the raw instruction text
    - Compare carefully with existing features before deciding placement

    ## CONFLICT AND DUPLICATE HANDLING

    - If the instruction updates an existing trait, reuse that feature's exact feature_name and return the updated merged value
    - If the instruction overlaps or duplicates an existing feature (same meaning or same value under a different name), merge into the existing feature instead of creating a new one
    - Reject only when the instruction is unrelated to this category or violates safety rules
    - Do not reject merely because similar content already exists; merge instead

    ## CATEGORY REJECTION

    - Reject instructions unrelated to agent personality
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
