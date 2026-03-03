"""Prompt template for life-oriented personal context semantic memory extraction.

This prompt focuses on extracting personal characteristics, interests, lifestyle patterns,
goals, and life context that help provide personalized life guidance and advice.
This complements structured facts extraction by focusing on personal insights rather than concrete data.
"""

from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    SemanticCategory,
)
from memmachine.semantic_memory.util.semantic_prompt_template import (
    build_update_prompt,
)

# Life-oriented personal context tags
life_context_tags: dict[str, str] = {
    # === Life & Personal Summary ===
    "interests": "Interests and hobbies: what the user enjoys doing, passions, recreational activities, creative pursuits, learning interests, entertainment preferences, cultural interests, things the user likes to do in their free time.",
    "lifestyle": "Lifestyle patterns and habits: daily routines, sleep patterns, exercise habits, dietary habits, work-life balance, stress management, leisure activities, travel patterns, time management style, how the user lives their daily life.",
    "goals": "Goals and aspirations: short-term and long-term goals (career, personal development, health, financial, relationship, educational), life vision, desired achievements, long-term plans, what the user wants to become or accomplish.",
    "personality": "Personality traits and characteristics: communication style, decision-making style, introversion/extroversion, openness to new experiences, conscientiousness, emotional stability, how the user typically behaves and interacts.",
    "life_situation": "Current life circumstances and context: living situation, family structure, work situation, major life events, transitions, challenges, opportunities, current stage of life, what's happening in the user's life right now.",
}

# Optimized description for life-oriented personal context
life_context_description = """
    You are a PERSONAL INSIGHT EXTRACTOR for a life advisor assistant. Your job is to extract deep 
    personal understanding that enables the agent to provide meaningful life guidance, understand 
    the user's motivations, and offer personalized advice that resonates with the user's values and goals.
    
    YOUR ROLE AND DISTINCTION
    
    What You Extract:
    - WHY and HOW the user thinks, feels, and behaves (motivations, patterns, characteristics)
    - Personal insights, values, goals, and life context
    - Psychological profile and life understanding
    
    What You Do NOT Extract:
    - WHAT the user has (facts, data, contact info, account numbers)
    - Concrete structured facts (these are handled separately)
    - Historical events or temporary states (these belong in episodic memory)
    
    Think of yourself as: Building a psychological profile and life context understanding, not a contact book.
    
    Important Context:
    - Episodic memories already contain refined descriptions and atomic claims, including all historical events and temporary states
    - Semantic memory is for STABLE, REUSABLE user information that persists across sessions
    - ALWAYS compare with existing features before creating new ones
    
    CRITICAL TAG RULES:
    - You MUST ONLY use the tags defined in the tags list below
    - DO NOT create new tags - use one of the existing tags: interests, lifestyle, goals, personality, life_situation
    - If information doesn't fit perfectly, choose the closest matching tag:
      * Personal interests/hobbies → "interests"
      * Daily routines/habits/patterns → "lifestyle"
      * Aspirations/objectives/plans → "goals"
      * Character traits/behavioral patterns → "personality"
      * Current circumstances/context → "life_situation"
    
    WHAT TO EXTRACT
    
    Extract These Personal Insights and Characteristics:
    
    Interests and Passions:
    - What the user enjoys, what motivates them, what they find meaningful
    - Examples: hobbies, creative pursuits, learning interests, entertainment preferences
    
    Lifestyle Patterns:
    - How the user lives daily life, routines, habits, work-life balance approach
    - Examples: daily routines, sleep patterns, exercise habits, dietary habits, stress management
    
    Goals and Aspirations:
    - What the user wants to achieve, life vision, desired outcomes
    - Examples: career goals, personal development goals, health goals, financial goals, life vision
    
    Personality Traits:
    - How the user communicates, makes decisions, handles stress, interacts with others
    - Examples: communication style, decision-making style, introversion/extroversion, emotional stability
    
    Life Situation Context:
    - Current life stage, major transitions, challenges, opportunities, family/work context
    - Examples: living situation, family structure, work situation, current stage of life
    
    Values and Priorities:
    - What matters to the user, what drives their decisions
    - Examples: core values, priorities, what the user finds important
    
    Behavioral Patterns:
    - How the user typically responds to situations, manages time, handles relationships
    - Examples: response patterns, time management style, relationship handling
    
    Do NOT Extract:
    - Historical events or past actions (episodic memory)
    - Temporary states or pending actions (episodic memory)
    - Concrete facts like names, phone numbers, email addresses, account numbers (these are structured facts, not personal insights)
    - Identity documents or account identifiers (these are structured facts, not personal insights)
    - Service provider contact information (these are structured facts, not personal insights)
    - One-time preferences or context-dependent choices (episodic memory)
    
    FEATURE NAMING RULES
    
    Format Rules:
    - Use UPPERCASE letters with SPACES between words (e.g., "PRIMARY INTEREST", "CAREER GOAL")
    - Use full words, not abbreviations
    - Be descriptive and meaningful - capture the essence of the characteristic
    
    Standard Feature Names:
    
    Interests:
    - "PRIMARY INTEREST" (not "INTEREST", "LIKES")
    - "PASSION" (not "HOBBY" if already using "PRIMARY INTEREST")
    - "HOBBY" (for secondary interests)
    
    Lifestyle:
    - "WORK LIFE BALANCE STYLE" (not "BALANCE", "LIFESTYLE")
    - "EXERCISE HABIT" or "FITNESS ROUTINE" (not "EXERCISE", "ROUTINE")
    - "SLEEP PATTERN" (not "SLEEP", "PATTERN")
    
    Goals:
    - "CAREER GOAL" (not "GOAL", "ASPIRATION")
    - "LONG TERM GOAL" (for general long-term goals)
    - "HEALTH GOAL" (not "HEALTH")
    - "FINANCIAL GOAL"
    - "LIFE VISION"
    
    Personality:
    - "COMMUNICATION STYLE" (not "STYLE", "PERSONALITY")
    - "DECISION MAKING STYLE" (not "DECISION", "APPROACH")
    - "STRESS MANAGEMENT APPROACH" (not "STRESS", "MANAGEMENT")
    - "INTROVERSION LEVEL"
    - "STRESS RESPONSE PATTERN"
    - "PERSONALITY TYPE"
    
    Life Context:
    - "CURRENT LIFE STAGE" (not "STAGE", "SITUATION")
    - "CORE VALUE" or "PRIORITY" (not "VALUE", "IMPORTANT")
    
    General Rules:
    - Avoid generic names like "INFO", "DATA", "DETAIL" - be specific about the characteristic
    - If an existing feature name means the same thing, USE THAT EXACT NAME - do not create a synonym
    
    HANDLING DUPLICATES AND UPDATES
    
    Before Adding or Updating a Feature:
    1. ALWAYS compare with existing features to check for duplicates or updates
    2. Analyze the claims (content) to determine if it's the same information or different
    
    Decision Rules Based on Claims:
    - If the claim represents the SAME information (same value, same meaning): OVERWRITE the existing feature using UPDATE command
      * Example: Existing "PRIMARY INTEREST" with value "photography", new claim mentions "main hobby is photography" → UPDATE "PRIMARY INTEREST"
    - If the claim represents DIFFERENT information (different value, different aspect): Create a new feature with a different suffix or name
      * Example: Existing "CAREER GOAL" with value "become a manager", new claim mentions "also wants to start a side business" → Keep "CAREER GOAL" and ADD "SIDE BUSINESS GOAL" or "ENTREPRENEURIAL GOAL"
    
    Handling Multiple Aspects of the Same Type:
    - If existing features already have specific names (e.g., "CAREER GOAL"), and new information is about a different aspect, create a new feature with a different name (e.g., "PERSONAL DEVELOPMENT GOAL", "HEALTH GOAL")
    - If existing feature has a generic name (e.g., "GOAL") and new information is about a different aspect:
      * Determine which aspect based on claims (e.g., "career goal" vs "health goal")
      * UPDATE the existing one to be more specific (e.g., "CAREER GOAL")
      * ADD the new one with appropriate name (e.g., "HEALTH GOAL")
    
    Reusing Feature Names:
    - If an existing feature name means the same thing, USE THAT EXACT NAME
    - Do not create synonyms or variations
    - Compare with existing features first - reuse existing feature names when the information matches
    
    EXTRACTION PROCESS
    
    Step-by-Step Process:
    1. Compare with existing features to identify duplicates or updates
    2. Analyze claims (content) to determine if information is the same or different
    3. **CRITICAL: Select the correct tag from the defined list (interests, lifestyle, goals, personality, life_situation) - DO NOT create new tags**
    4. Use standard feature names (see FEATURE NAMING RULES above)
    5. **CRITICAL: For duplicate feature names, decide based on claims: if same information → OVERWRITE (UPDATE), if different information → create new with different name or suffix**
    6. Extract INSIGHTS and UNDERSTANDING, not just facts
    7. Extract patterns and recurring themes that reveal life context
    8. Look for underlying motivations, values, and personality traits
    9. Include information that helps the agent understand the user deeply
    
    Priority Order:
    1. Personal characteristics that affect life guidance (personality, life_situation) - most important for advice
    2. Goals and aspirations (goals) - essential for providing relevant guidance
    3. Lifestyle patterns and habits (lifestyle) - important for understanding daily life
    4. Interests and hobbies (interests) - helpful for personalization
    
    Remember: Extract stable, reusable personal insights that help the agent understand the user 
    deeply and provide meaningful life guidance. Focus on WHY and HOW, not WHAT. If it's a 
    one-time event or temporary state, it belongs in episodic memory, not semantic memory.
"""

# Custom consolidation prompt for life-oriented personal context
life_context_consolidation_prompt = """
    Your job is to perform memory consolidation for a life-oriented personal context memory system.
    Despite the name, consolidation is not solely about reducing the amount of memories, but rather, minimizing interference between personal insights while maintaining psychological profile integrity and life understanding.
    By consolidating memories, we remove unnecessary couplings of personal information from context, spurious correlations inherited from the circumstances of their acquisition.

    You will receive a set of life-oriented personal context memories which are semantically similar (same tag and feature name).
    Produce a new list of memories to keep.

    A memory is a json object with 4 fields:
    - tag: broad category of memory (interests, lifestyle, goals, personality, life_situation)
    - feature: feature name (e.g., "PRIMARY INTEREST", "CAREER GOAL", "COMMUNICATION STYLE")
    - value: detailed contents of the memory
    - metadata: object with 1 field
    -- id: integer

    You will output consolidated memories, which are json objects with 4 fields:
    - tag: string (must be one of: interests, lifestyle, goals, personality, life_situation)
    - feature: string (must follow FEATURE NAMING RULES below)
    - value: string (detailed contents)
    - metadata: object with 1 field
    -- citations: list of ids of old memories which influenced this one

    You will also output a list of old memories to keep (memories are deleted by default).

    CRITICAL TAG RULES:
    - You MUST ONLY use the tags defined in the tags list: interests, lifestyle, goals, personality, life_situation
    - DO NOT create new tags - if information doesn't fit perfectly, choose the closest matching tag
    - Tag selection guidance:
      * Personal interests/hobbies → "interests"
      * Daily routines/habits/patterns → "lifestyle"
      * Aspirations/objectives/plans → "goals"
      * Character traits/behavioral patterns → "personality"
      * Current circumstances/context → "life_situation"

    FEATURE NAMING RULES (MUST FOLLOW):
    - Use UPPERCASE letters with SPACES between words (e.g., "PRIMARY INTEREST", "CAREER GOAL")
    - Use full words, not abbreviations
    - Be descriptive and meaningful - capture the essence of the characteristic

    Standard Feature Names:
    - Interests: "PRIMARY INTEREST", "PASSION", "HOBBY"
    - Lifestyle: "WORK LIFE BALANCE STYLE", "EXERCISE HABIT", "FITNESS ROUTINE", "SLEEP PATTERN"
    - Goals: "CAREER GOAL", "LONG TERM GOAL", "HEALTH GOAL", "FINANCIAL GOAL", "LIFE VISION"
    - Personality: "COMMUNICATION STYLE", "DECISION MAKING STYLE", "STRESS MANAGEMENT APPROACH", "INTROVERSION LEVEL", "STRESS RESPONSE PATTERN", "PERSONALITY TYPE"
    - Life Context: "CURRENT LIFE STAGE", "CORE VALUE", "PRIORITY"

    CONSOLIDATION GUIDELINES:

    1. **Identical Information (Same Value & Meaning)**: 
       - If memories have identical values and meanings, DELETE duplicates and KEEP only one
       - Example: Multiple "PRIMARY INTEREST" features with value "photography" → Keep one, delete others

    2. **Different Information (Different Aspect or Evolution)**:
       - If memories have different values for the same feature name, they represent different aspects or evolution
       - For goals: Different goals should be kept separate (e.g., "CAREER GOAL" vs "HEALTH GOAL")
       - For personality traits: If values represent evolution or refinement, KEEP the most complete/current version
       - Example: "CAREER GOAL": "become a manager" and "CAREER GOAL": "become a senior manager" → 
         Keep "CAREER GOAL": "become a senior manager" (more complete), delete the older one

    3. **Synonym Consolidation**:
       - If memories have the same meaning but different feature names (synonyms), consolidate to use the standard feature name
       - Example: "MAIN HOBBY": "photography" and "PRIMARY INTEREST": "photography" → 
         Keep "PRIMARY INTEREST": "photography", delete "MAIN HOBBY"

    4. **Redundant Information**:
       - Memories containing only redundant information should be deleted entirely
       - If information has been processed into a more complete memory, delete the incomplete versions
       - Example: "GOAL": "career advancement" and "CAREER GOAL": "advance to senior management role" → 
         Keep "CAREER GOAL" (more specific), delete generic "GOAL"

    5. **Feature Name Synchronization**:
       - If memories are sufficiently similar but differ in key details, synchronize their feature names
       - Use consistent naming across similar memories
       - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field
       - Example: "GOAL": "health improvement" and "HEALTH GOAL": "lose weight and exercise more" → 
         Keep "HEALTH GOAL": "lose weight and exercise more" (more specific), delete generic "GOAL"

    6. **Multiple Aspects Handling**:
       - If memories represent different aspects of the same type, keep them separate with appropriate names
       - Example: "CAREER GOAL": "become a manager" and "ENTREPRENEURIAL GOAL": "start a side business" → 
         Keep both (different aspects)
       - Don't merge different aspects into a single memory unless they naturally belong together

    7. **Personality and Lifestyle Evolution**:
       - For personality traits and lifestyle patterns, if values represent evolution or refinement:
         * Keep the most complete/current version
         * If both are valuable, merge insights into a single comprehensive memory
       - Example: "COMMUNICATION STYLE": "direct" and "COMMUNICATION STYLE": "direct and concise" → 
         Keep "COMMUNICATION STYLE": "direct and concise" (more complete)

    Overall memory life-cycle:
    raw personal insights -> extracted characteristics -> characteristics sorted by tag -> consolidated life profiles

    The more memories you receive, the more interference there is in the memory system.
    This causes cognitive load. Cognitive load is bad.
    To minimize this, under such circumstances, you need to be more aggressive about deletion:
        - Be looser about what you consider to be similar. Some distinctions are not worth the energy to maintain.
        - Massage out the parts to keep and ruthlessly throw away the rest
        - There is no free lunch here! At least some information must be deleted!

    Do not create new tag names. Only use: interests, lifestyle, goals, personality, life_situation.

    Remember: Focus on consolidating personal insights that help understand the user deeply. 
    Preserve meaningful distinctions while removing redundant information.

    The proper noop syntax is:
    {
        "consolidate_memories": [],
        "keep_memories": []
    }

    The final output schema is:
    <think> insert your chain of thought here. </think>
    {
        "consolidate_memories": list of new memories to add,
        "keep_memories": list of ids of old memories to keep
    }
"""

LifeContextSemanticCategory = SemanticCategory(
    name="profile_life_context",
    prompt=RawSemanticPrompt(
        update_prompt=build_update_prompt(
        tags=life_context_tags,
        description=life_context_description,
        ),
        consolidation_prompt=life_context_consolidation_prompt,
    ),
)

SEMANTIC_TYPE = LifeContextSemanticCategory
