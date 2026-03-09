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
    "interests": "LONG-TERM interests and hobbies: what the user enjoys doing, passions, recreational activities, creative pursuits, learning interests, entertainment preferences, cultural interests, things the user likes to do in their free time. EXCLUDE: One-time activities, temporary interests, or things the user is doing just for this trip/event.",
    "lifestyle": "STABLE lifestyle patterns and habits: daily routines, sleep patterns, exercise habits, dietary habits, work-life balance, stress management, leisure activities, time management style, how the user lives their daily life. EXCLUDE: Temporary routines, vacation schedules, or travel-related patterns.",
    "goals": "LONG-TERM goals and aspirations: career goals, personal development goals, health goals, financial goals, relationship goals, educational goals, life vision, desired achievements, what the user wants to become or accomplish. EXCLUDE: Short-term tasks, current to-dos, or temporary objectives.",
    "personality": "STABLE personality traits and characteristics: communication style, decision-making style, introversion/extroversion, openness to new experiences, conscientiousness, emotional stability, how the user typically behaves and interacts. These are enduring traits, not temporary moods.",
    "life_situation": "STABLE life circumstances and context: living situation (permanent), family structure, work situation (job/career, not current projects), major life stage. EXCLUDE: Current location, temporary residence (hotels/Airbnbs), travel status, current trips, temporary accommodations, or any time-bound situations. ASK: 'Will this still be true in 6 months?' If NO, do not extract.",
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
    
    CRITICAL: Do NOT Extract (These belong in episodic memory):
    
    LOCATION & TRAVEL (NEVER EXTRACT):
    - Current location or whereabouts (e.g., "user is at the airport", "user is in Chicago")
    - Temporary residence (e.g., "staying at Airbnb at 123 Main St", "hotel room 405")
    - Travel destinations or itineraries (e.g., "flying to Paris tomorrow", "on vacation in Hawaii")
    - Vacation rentals, hotels, short-term accommodations
    - Current trips or travel status
    
    TEMPORAL STATES (NEVER EXTRACT):
    - Historical events or past actions (episodic memory)
    - Temporary states or pending actions (episodic memory)
    - Time-specific information that will become outdated
    - Current projects or tasks in progress
    - Appointments, meetings, or scheduled events
    
    STRUCTURED FACTS (EXTRACT IN task_assistant_prompt INSTEAD):
    - Concrete facts like names, phone numbers, email addresses, account numbers
    - Identity documents or account identifiers
    - Service provider contact information
    
    TRANSIENT INFORMATION:
    - One-time preferences or context-dependent choices
    - Temporary moods or situational feelings
    - Things the user is doing "right now" that won't persist
    
    ## CRITICAL: NEVER Extract Sensitive Information
    
    For security reasons, NEVER extract or store ANY of the following (not even partial):
    - Social security numbers, passport numbers, driver's license numbers
    - Credit card numbers, bank account numbers, routing numbers
    - CVV/security codes, expiration dates, bank PINs
    - Passwords, PINs, security questions, authentication credentials
    - Biometric data, medical records, detailed financial records
    - Any information that could enable identity theft or fraud
    
    If the user mentions sensitive information in passing, DO NOT extract it into semantic memory.
    Such information should never appear in life context features.
    
    Note: Life context is about personal insights (interests, goals, personality), NOT account/identity data.
    Non-sensitive IDs like employee ID or student ID belong in task_assistant_prompt, not here.
    
    ASK YOURSELF: "Will this information still be accurate in 6 months?"
    - If YES → Extract it
    - If NO → Do NOT extract (it belongs in episodic memory)
    
    FEATURE NAMING RULES
    
    Format Rules:
    - Use UPPERCASE letters with SPACES between words (e.g., "PRIMARY INTEREST", "CAREER GOAL")
    - Use full words, not abbreviations
    - Be descriptive and meaningful - capture the essence of the characteristic
    
    Standard Feature Names:
    
    Interests:
    - "PRIMARY INTEREST" (not "INTEREST", "LIKES", "HOBBY")
    - "SECONDARY INTEREST" (for additional interests beyond primary)
    - "PASSION" (for deep, meaningful interests)
    - Multiple interests: If the user has multiple distinct interests, use descriptive suffixes
      * Examples: "INTEREST PHOTOGRAPHY", "INTEREST COOKING", "INTEREST MUSIC"
      * Or use numbered format: "INTEREST 1", "INTEREST 2" (less preferred)
    
    Lifestyle:
    - "WORK LIFE BALANCE STYLE" (not "BALANCE", "LIFESTYLE")
    - "EXERCISE HABIT" or "FITNESS ROUTINE" (not "EXERCISE", "ROUTINE")
    - "SLEEP PATTERN" (not "SLEEP", "PATTERN")
    - "DIETARY HABIT" (not "DIET", "FOOD HABIT")
    - "DAILY ROUTINE" (for general daily patterns)
    - Multiple patterns: Use descriptive suffixes
      * Examples: "ROUTINE MORNING", "ROUTINE EVENING", "ROUTINE WEEKEND"
    
    Goals:
    - "CAREER GOAL" (not "GOAL", "ASPIRATION", "WORK GOAL")
    - "HEALTH GOAL" (not "HEALTH", "FITNESS GOAL")
    - "FINANCIAL GOAL" (not "MONEY GOAL", "FINANCE")
    - "PERSONAL DEVELOPMENT GOAL" (for self-improvement goals)
    - "RELATIONSHIP GOAL" (for social/family goals)
    - "LIFE VISION" (for overarching life direction)
    - "LONG TERM GOAL" (for general long-term goals that don't fit above)
    - "SHORT TERM GOAL" (for near-term objectives)
    - Multiple goals of the same type: Use descriptive suffixes
      * Examples: "CAREER GOAL PRIMARY", "CAREER GOAL SECONDARY"
      * Or: "HEALTH GOAL FITNESS", "HEALTH GOAL DIET"
    
    Personality:
    - "COMMUNICATION STYLE" (not "STYLE", "PERSONALITY", "HOW THEY TALK")
    - "DECISION MAKING STYLE" (not "DECISION", "APPROACH")
    - "STRESS MANAGEMENT APPROACH" (not "STRESS", "MANAGEMENT")
    - "SOCIAL PREFERENCE" (introversion/extroversion, not "INTROVERSION LEVEL")
    - "CONFLICT RESOLUTION STYLE" (not "CONFLICT", "ARGUMENT STYLE")
    - "EMOTIONAL PATTERN" (not "EMOTIONS", "MOOD")
    - "PERSONALITY TYPE" (for formal personality assessments like MBTI)
    
    Life Situation:
    - "CURRENT LIFE STAGE" (not "STAGE", "SITUATION", "LIFE PHASE")
    - "CORE VALUE" (not "VALUE", "IMPORTANT", "BELIEF")
    - "PRIORITY" (for what matters most right now)
    - "LIFE CHALLENGE" (for current difficulties)
    - "LIFE OPPORTUNITY" (for current opportunities)
    - "FAMILY SITUATION" (for family context)
    - "WORK SITUATION" (for career/job context)
    - Multiple values: Use descriptive suffixes
      * Examples: "CORE VALUE FAMILY", "CORE VALUE CAREER", "CORE VALUE HEALTH"
    
    General Rules:
    - Avoid generic names like "INFO", "DATA", "DETAIL", "THING" - be specific about the characteristic
    - If an existing feature name means the same thing, USE THAT EXACT NAME - do not create a synonym
    - Use consistent naming: don't mix "GOAL" and "ASPIRATION" for the same concept
    - Prefer specific names over generic ones: "CAREER GOAL" is better than "GOAL"
    
    HANDLING DUPLICATES AND UPDATES
    
    Before Adding or Updating a Feature:
    1. ALWAYS compare with existing features to check for duplicates or updates
    2. Analyze the claims (content) to determine if it's the same information or different
    
    Decision Rules Based on Claims:
    - If the claim represents the SAME information (same value, same meaning): OVERWRITE the existing feature using UPDATE command
      * Example: Existing "PRIMARY INTEREST" with value "photography", new claim mentions "main hobby is photography" → UPDATE "PRIMARY INTEREST"
      * Example: Existing "CAREER GOAL" with value "become a manager", new claim mentions "wants to be promoted to manager" → UPDATE "CAREER GOAL" (same goal, different wording)
    - If the claim represents DIFFERENT information (different value, different aspect): Create a new feature with a different suffix or name
      * Example: Existing "CAREER GOAL" with value "become a manager", new claim mentions "also wants to start a side business" → Keep "CAREER GOAL" and ADD "CAREER GOAL ENTREPRENEURIAL" or "ENTREPRENEURIAL GOAL"
      * Example: Existing "PRIMARY INTEREST" with value "photography", new claim mentions "also loves cooking" → Keep "PRIMARY INTEREST" and ADD "INTEREST COOKING" or "SECONDARY INTEREST"
    
    Handling Multiple Aspects of the Same Type:
    - If existing features already have specific names (e.g., "CAREER GOAL"), and new information is about a different aspect, create a new feature with a different name or suffix (e.g., "PERSONAL DEVELOPMENT GOAL", "HEALTH GOAL", or "CAREER GOAL SECONDARY")
    - If existing feature has a generic name (e.g., "GOAL") and new information is about a different aspect:
      * Determine which aspect based on claims (e.g., "career goal" vs "health goal")
      * UPDATE the existing one to be more specific (e.g., "CAREER GOAL")
      * ADD the new one with appropriate name (e.g., "HEALTH GOAL")
    - If existing feature has no suffix (e.g., "CORE VALUE") and new information is about a different value of the same type:
      * Determine which is which based on claims (e.g., "family values" vs "career values")
      * UPDATE the existing one to add appropriate suffix (e.g., "CORE VALUE FAMILY")
      * ADD the new one with different suffix (e.g., "CORE VALUE CAREER")
    
    Reusing Feature Names:
    - If an existing feature name means the same thing, USE THAT EXACT NAME
    - Do not create synonyms or variations (e.g., don't use "HOBBY" if "PRIMARY INTEREST" already exists for the same concept)
    - Compare with existing features first - reuse existing feature names when the information matches
    - For multiple items: Check if a suffix already exists, and use consistent suffix naming
      * Example: If "INTEREST PHOTOGRAPHY" exists, use "INTEREST COOKING" not "HOBBY COOKING"
      * Example: If "CORE VALUE FAMILY" exists, use "CORE VALUE CAREER" not "CAREER PRIORITY"
    
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
    - Use consistent naming: don't mix "GOAL" and "ASPIRATION" for the same concept
    - Prefer specific names over generic ones: "CAREER GOAL" is better than "GOAL"

    Standard Feature Names:
    - Interests: "PRIMARY INTEREST", "SECONDARY INTEREST", "PASSION", or use suffixes like "INTEREST PHOTOGRAPHY", "INTEREST COOKING"
    - Lifestyle: "WORK LIFE BALANCE STYLE", "EXERCISE HABIT", "FITNESS ROUTINE", "SLEEP PATTERN", "DIETARY HABIT", "DAILY ROUTINE", or use suffixes like "ROUTINE MORNING", "ROUTINE EVENING"
    - Goals: "CAREER GOAL", "HEALTH GOAL", "FINANCIAL GOAL", "PERSONAL DEVELOPMENT GOAL", "RELATIONSHIP GOAL", "LIFE VISION", "LONG TERM GOAL", "SHORT TERM GOAL", or use suffixes like "CAREER GOAL PRIMARY", "HEALTH GOAL FITNESS"
    - Personality: "COMMUNICATION STYLE", "DECISION MAKING STYLE", "STRESS MANAGEMENT APPROACH", "SOCIAL PREFERENCE", "CONFLICT RESOLUTION STYLE", "EMOTIONAL PATTERN", "PERSONALITY TYPE"
    - Life Situation: "CURRENT LIFE STAGE", "CORE VALUE", "PRIORITY", "LIFE CHALLENGE", "LIFE OPPORTUNITY", "FAMILY SITUATION", "WORK SITUATION", or use suffixes like "CORE VALUE FAMILY", "CORE VALUE CAREER"

    CONSOLIDATION GUIDELINES:

    0. **DELETE SENSITIVE AND TEMPORARY INFORMATION (HIGHEST PRIORITY)**:
    
       SENSITIVE INFORMATION - DELETE IMMEDIATELY (no partial storage allowed):
       - DELETE: Any social security numbers (full or partial)
       - DELETE: Any passport numbers, driver's license numbers
       - DELETE: Any credit card numbers (full or partial), bank account numbers, routing numbers
       - DELETE: Passwords, PINs, security questions, authentication credentials, CVV codes
       - DELETE: Biometric data, detailed medical records, detailed financial records
       - DELETE: Any information that could enable identity theft or fraud
       - Life context should NEVER contain sensitive PII - delete any that appears
       
       TEMPORARY/TRANSIENT INFORMATION - DELETE:
       - DELETE: "CURRENT LOCATION", "USER LOCATION", "STAYING AT", "TEMPORARY RESIDENCE", or similar
       - DELETE: Airbnb addresses, hotel rooms, vacation rentals, current trips
       - DELETE: Any information with specific dates that indicate it's time-bound
       - DELETE: Current projects, tasks in progress, temporary situations
       - DELETE: Travel itineraries, current trips, "currently at" information
       - Example: "LIFE SITUATION": "staying at Airbnb in Chicago" → DELETE entirely
       - Example: "USER LOCATION": "currently traveling in Europe" → DELETE entirely
       - ASK: "Will this still be true in 6 months?" If NO → DELETE

    1. **Identical Information (Same Value & Meaning)**: 
       - If memories have identical values and meanings, DELETE duplicates and KEEP only one
       - Example: Multiple "PRIMARY INTEREST" features with value "photography" → Keep one, delete others

    2. **Different Information (Different Aspect or Evolution)**:
       - If memories have different values for the same feature name, determine if they represent:
         * Evolution/refinement of the same thing → KEEP the most complete/current version
         * Different distinct items → UPDATE feature names to use appropriate suffixes
       - For goals: Different goals should be kept separate with different names or suffixes
       - For personality traits: If values represent evolution or refinement, KEEP the most complete/current version
       - Example (Evolution): "CAREER GOAL": "become a manager" and "CAREER GOAL": "become a senior manager" → 
         Keep "CAREER GOAL": "become a senior manager" (more complete), delete the older one
       - Example (Different items): "PRIMARY INTEREST": "photography" and "PRIMARY INTEREST": "cooking" → 
         * Keep "INTEREST PHOTOGRAPHY": "photography"
         * Keep "INTEREST COOKING": "cooking"
         * Delete original "PRIMARY INTEREST" entries

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
       - If memories represent different aspects of the same type, keep them separate with appropriate names or suffixes
       - UPDATE feature names to use appropriate suffixes when needed
       - Example: "CAREER GOAL": "become a manager" and "CAREER GOAL": "start a side business" → 
         * Keep "CAREER GOAL PRIMARY": "become a manager"
         * Keep "CAREER GOAL SECONDARY": "start a side business"
         * Delete original "CAREER GOAL" entries
       - Or use descriptive suffixes: "CAREER GOAL MANAGEMENT" and "CAREER GOAL ENTREPRENEURIAL"
       - Don't merge different aspects into a single memory unless they naturally belong together
       - Don't create suffixes too early. Have at least two distinct items first
       - Use consistent suffix naming across similar types (e.g., "INTEREST PHOTOGRAPHY", "INTEREST COOKING", not "HOBBY PHOTOGRAPHY", "PASSION COOKING")

    7. **Handling Multiple Items of the Same Type**:
       - If existing features already have suffixes (e.g., "INTEREST PHOTOGRAPHY"), and new information is about a different item, create a new feature with a different suffix (e.g., "INTEREST COOKING")
       - If existing feature has no suffix (e.g., "PRIMARY INTEREST") and new information is about a different item of the same type:
         * Determine which is which based on content
         * UPDATE the existing one to add appropriate suffix if needed (e.g., "INTEREST PHOTOGRAPHY")
         * ADD the new one with different suffix (e.g., "INTEREST COOKING")
       - Use consistent suffix naming (e.g., "CORE VALUE FAMILY", "CORE VALUE CAREER", not "FAMILY VALUE", "CAREER PRIORITY")

    8. **Personality and Lifestyle Evolution**:
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
