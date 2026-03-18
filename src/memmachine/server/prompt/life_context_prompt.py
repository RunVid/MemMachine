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
    "interests": "Long-term interests and hobbies: what the user enjoys doing, passions, recreational activities, creative pursuits, learning interests, entertainment preferences, cultural interests, and things the user likes to do in their free time.",
    "lifestyle": "Stable lifestyle patterns and habits: daily routines, sleep patterns, exercise habits, dietary habits, work-life balance approach, stress management techniques, leisure activities, and time management style.",
    "goals": "Long-term goals and aspirations: career goals, personal development goals, health and fitness goals, financial goals, relationship goals, educational goals, life vision, and desired achievements.",
    "personality": "Stable personality traits and characteristics: communication style, decision-making style, introversion/extroversion, openness to new experiences, conscientiousness, emotional stability, and how the user typically interacts with others.",
    "life_situation": "Stable life circumstances and context: permanent living situation (city/region), family structure, work situation (job/career), major life stage, and long-term commitments.",
}

# Optimized description for life-oriented personal context
life_context_description = """
    You are extracting life-oriented personal context from conversations with a life advisor assistant.
    This information is used to provide personalized life guidance and understand the user's psychological profile.
    
    ## YOUR ROLE
    
    - Extract personal insights: WHY and HOW the user thinks, feels, and behaves
    - Store interests, lifestyle patterns, goals, personality traits, and life circumstances
    - Build a psychological profile that enables meaningful, personalized advice
    
    Do NOT Extract:
    - WHAT the user has (contact info, account numbers) → task_assistant_prompt extracts those
    - Historical events or temporary states → episodic memory stores those
    
    Important: Semantic memory is for STABLE, REUSABLE information.
    ALWAYS compare with existing features before creating new ones.
    
    ## TAG RULES
    
    You MUST ONLY use: interests, lifestyle, goals, personality, life_situation
    - interests: hobbies, passions, entertainment preferences
    - lifestyle: routines, habits, work-life balance
    - goals: aspirations, objectives, plans
    - personality: character traits, behavioral patterns
    - life_situation: stable circumstances, context
    - DO NOT create new tags - choose the closest matching tag
    - CRITICAL: Tags are case-sensitive and MUST be lowercase (e.g., "interests" NOT "INTERESTS" or "Interests")
    
    ## WHAT TO EXTRACT
    
    **ONLY extract STATIC, FACTUAL DATA VALUES - NOT actions or events.**
    
    Stable, long-term information:
    - Interests: hobbies, passions, creative pursuits, learning interests, entertainment preferences
    - Lifestyle: daily routines, sleep patterns, exercise habits, dietary habits, work-life balance
    - Goals: career aspirations, personal development goals, health goals, life vision, educational goals
    - Personality: communication style, decision-making style, introversion/extroversion, emotional patterns
    - Life Situation: living situation (city/region), family structure, work situation, major life stage
    - Values: core values, priorities, what drives decisions
    
    **CRITICAL: Extract ONLY the personal characteristic itself, NOT the action or event.**
    
    Examples of CORRECT extraction:
    - "I exercise 3 times a week" → feature="EXERCISE HABIT", value="exercises 3 times a week" ✓
    - "My goal is to become a senior manager" → feature="CAREER GOAL", value="become a senior manager" ✓
    - "I prefer working alone" → feature="SOCIAL PREFERENCE", value="prefers working alone" ✓
    
    Key Questions:
    1. "Is this a stable personal characteristic (interest, habit, goal, trait)?" If YES → extract. If NO → skip.
    2. "Does this describe an action or event?" If YES → skip (episodic memory handles this).
    3. "Will this still be accurate in 6 months?" If YES → extract. If NO → skip.
    
    ## WHAT NOT TO EXTRACT
    
    ### Temporary/Transient Information (belongs in episodic memory)
    - Current location, travel status, temporary residence (hotels, Airbnbs)
    - Historical events, past actions, current projects or specific tasks
    - Temporary moods, situational feelings, context-dependent choices
    - Time-bound information (e.g., "going to gym today" vs stable "exercises regularly")
    - Actions or events: "User started X", "User decided Y", "User completed Z"
    - Timestamps of actions: "started on March 12", "decided yesterday"
    
    Examples of INCORRECT extraction (DO NOT DO THIS):
    - "User started exercising on March 12" → DO NOT EXTRACT (this is an action/event)
    - "User decided to learn photography" → DO NOT EXTRACT (this is an action/event)
    - "User completed career goal yesterday" → DO NOT EXTRACT (this is an action/event)
    
    ### Structured Facts (belongs in task_assistant_prompt)
    - Contact info (phone, email), IDs, account numbers
    - Service provider contact details
    - Specific addresses, appointments, schedules
    
    ### Highly Sensitive PII (never store for security)
    - Government IDs: SSN, passport numbers, driver's license numbers
    - Financial: credit card numbers, bank account numbers
    - Security: passwords, PINs, authentication credentials
    - Private records: medical records, financial records, legal documents
    
    Examples:
    - "I'm currently at the gym" → DO NOT EXTRACT (temporary state)
    - "I go to the gym 3 times a week" → EXTRACT as EXERCISE HABIT (stable pattern)
    - "My SSN is 123-45-6789" → DO NOT EXTRACT (sensitive PII)
    
    ## FEATURE NAMING RULES
    
    ### Format
    - Use UPPERCASE letters with SPACES between words (e.g., "EXERCISE HABIT", "CAREER GOAL")
    - Use full words, not abbreviations
    - Be specific and descriptive
    
    ### Standard Feature Names by Tag
    
    **Interests:** Use descriptive suffixes for multiple interests
    - For specific interests: "INTEREST PHOTOGRAPHY", "INTEREST COOKING", "INTEREST GAMING"
    - Avoid vague names like "PRIMARY INTEREST" or "SECONDARY INTEREST"
    
    **Lifestyle:** 
    - "EXERCISE HABIT", "SLEEP PATTERN", "DIETARY HABIT", "WORK LIFE BALANCE STYLE"
    - Multiple routines: "ROUTINE MORNING", "ROUTINE EVENING"
    
    **Goals:**
    - "CAREER GOAL", "HEALTH GOAL", "FINANCIAL GOAL", "PERSONAL DEVELOPMENT GOAL", "LIFE VISION"
    - Multiple goals of same type: "CAREER GOAL PRIMARY", "CAREER GOAL ENTREPRENEURIAL"
    
    **Personality:**
    - "COMMUNICATION STYLE", "DECISION MAKING STYLE", "SOCIAL PREFERENCE", "EMOTIONAL PATTERN"
    
    **Life Situation:**
    - "CURRENT LIFE STAGE", "FAMILY SITUATION", "WORK SITUATION"
    - Core values: "CORE VALUE FAMILY", "CORE VALUE CAREER"
    
    ### Multiple Items
    Use SUFFIXES to distinguish similar items:
    - Multiple interests: "INTEREST PHOTOGRAPHY", "INTEREST COOKING"
    - Multiple values: "CORE VALUE FAMILY", "CORE VALUE CAREER"
    
    ## HANDLING DUPLICATES AND UPDATES
    
    Before adding new features:
    1. Compare with existing features to check for duplicates
    2. Analyze if it's the same or different information
    
    ### Decision Rules
    - SAME information (same meaning): Do NOT add duplicate - skip it
    - UPDATED information (evolution of same characteristic): DELETE old, ADD new
    - DIFFERENT information (different characteristic): ADD new with descriptive suffixes
    
    ### Examples
    
    **Example 1: Exact Duplicate (Skip)**
    - Existing: feature="INTEREST PHOTOGRAPHY", value="User likes photography"
    - New claim: "I enjoy photography"
    → Skip (same interest, duplicate)
    
    **Example 2: Evolution of Same Characteristic (DELETE + ADD)**
    - Existing: feature="CAREER GOAL", value="become a manager"
    - New claim: "My goal is to become a senior manager"
    → DELETE "CAREER GOAL" (old), ADD "CAREER GOAL" with value="become a senior manager"
    
    **Example 3: Multiple Different Items (ADD with descriptive suffix)**
    - Existing: feature="PRIMARY INTEREST", value="photography"
    - New claim: "I also like cooking"
    → ADD "INTEREST COOKING" (cooking)
    
    **Example 4: Discovering Multiple Goals (ADD with descriptive suffix)**
    - Existing: feature="CAREER GOAL", value="become a manager"
    - New claim: "I also want to start a side business"
    → ADD "CAREER GOAL ENTREPRENEURIAL" (start a side business)
    
    **REMEMBER: NEVER have two features with the SAME feature name. Update (Delete old + Add new) or Use suffixes to distinguish them.**
    **REMEMBER: Avoid vague names like "PRIMARY INTEREST" or "SECONDARY INTEREST". Use specific names like "INTEREST PHOTOGRAPHY".**
    
    ## EXTRACTION PROCESS
    
    **REMEMBER: Extract ONLY static personal characteristics, NEVER actions or events.**
    
    1. Compare with existing features to identify duplicates or updates
    2. Select the correct tag (DO NOT create new tags)
    3. Use specific, descriptive feature names (avoid vague names like "PRIMARY INTEREST")
    4. For duplicates: same info → skip, evolution → DELETE old + ADD new, different items → ADD new with descriptive suffix
    5. Extract the characteristic itself (e.g., "exercises 3 times a week"), NOT the action (e.g., "User started exercising")
    6. Look for underlying motivations, values, and personality traits
    7. Avoid extracting sensitive PII or temporary states
    8. **NEVER create two features with the same feature name** - use descriptive suffixes to distinguish
    
    Priority: personality/life_situation > goals > lifestyle > interests
"""

# Custom consolidation prompt for life-oriented personal context
life_context_consolidation_prompt = """
    You are performing memory consolidation for a life-oriented personal context memory system.
    Consolidation minimizes interference between personal insights while maintaining psychological profile integrity.

    ## INPUT/OUTPUT FORMAT

    ### Input Memory
    ```json
    {"tag": "string", "feature": "string", "value": "string", "metadata": {"id": integer}}
    ```

    ### Output Memory
    ```json
    {"tag": "string", "feature": "string", "value": "string", "metadata": {"citations": [list of ids]}}
    ```

    ## RULES

    ### Tags
    
    - All input memories have the SAME tag. Your output memories MUST use the SAME tag as the input.
    - DO NOT change tags during consolidation. If input tag is "interests", output tag for all consolidation memory MUST be "interests".
    - CRITICAL: Tags MUST be lowercase (e.g., "interests" NOT "INTERESTS" or "Interests")

    ### Feature Names
    - UPPERCASE with SPACES (e.g., "PRIMARY INTEREST", "CAREER GOAL")
    - No underscores or other special characters in feature names
    - MUST NOT use underscores or other special characters in feature names
    - For multiple items of same type, use descriptive suffixes: "INTEREST PHOTOGRAPHY", "INTEREST COOKING"
    - Standard names:
      - Interests: "INTEREST [NAME]" for specific interests (e.g., "INTEREST PHOTOGRAPHY")
      - Lifestyle: "EXERCISE HABIT", "SLEEP PATTERN", "DIETARY HABIT"
      - Goals: "CAREER GOAL", "HEALTH GOAL", "FINANCIAL GOAL", "LIFE VISION"
      - Personality: "COMMUNICATION STYLE", "DECISION MAKING STYLE", "SOCIAL PREFERENCE"
      - Life Situation: "CURRENT LIFE STAGE", "CORE VALUE", "FAMILY SITUATION"

    ## CONSOLIDATION GUIDELINES

    ### 0. DELETE FIRST (Highest Priority)

    **Actions/Events - DELETE:**
    - "User started X on [date]"
    - "User decided Y"
    - "User completed Z yesterday"
    - Any value describing an action or event instead of a static characteristic
    - ASK: "Is this a static characteristic or an action?" If ACTION → DELETE

    **Highly Sensitive PII - DELETE:**
    - SSN, passport numbers, driver's license numbers
    - Credit card/bank account numbers
    - Passwords, PINs, credentials
    - Medical records, financial records, legal documents

    **Temporary Information - DELETE:**
    - "CURRENT LOCATION", "USER LOCATION", "STAYING AT"
    - Airbnb addresses, hotel rooms, current trips
    - Time-bound information, current projects
    - ASK: "Will this still be true in 6 months?" If NO → DELETE

    **OK to Keep:**
    - Long-term interests, stable lifestyle patterns
    - Personality traits, life goals
    - Core values, stable life circumstances
    - ONLY if the value is the actual characteristic (e.g., "exercises regularly"), NOT an action description

    ### 1. Identical or Nearly Identical Information
    
    **Same tag + same feature name + same/similar value:**
    - DELETE all duplicates, KEEP only one (the most complete version)
    - Or DELETE all, CREATE one consolidated version
    
    Examples:
    - tag="interests", feature="INTEREST PHOTOGRAPHY", value="User likes photography"
    - tag="interests", feature="INTEREST PHOTOGRAPHY", value="User enjoys photography"
    → DELETE both, CREATE: {"tag": "interests", "feature": "INTEREST PHOTOGRAPHY", "value": "User enjoys photography as a hobby"}
    
    - tag="interests", feature="INTEREST BIKING", value="User likes biking"
    - tag="interests", feature="INTEREST BIKING", value="User likes biking."  (with period)
    → DELETE duplicate, KEEP one or CREATE consolidated version
    
    **Same tag + same feature name + different value:**
    - This usually means a mistake or evolution
    - If evolution: DELETE old, KEEP new
    - If conflicting: merge into one comprehensive value
    
    Example:
    - tag="goals", feature="CAREER GOAL", value="become a manager"
    - tag="goals", feature="CAREER GOAL", value="become a senior manager"
    → DELETE old, KEEP: {"tag": "goals", "feature": "CAREER GOAL", "value": "become a senior manager"}

    ### 2. Multiple Items of Same Type
    Use descriptive feature names with the item name as suffix
    
    **Note: "UPDATE" or "Rename" means DELETE old feature(s) and CREATE new feature(s) with better names.**
    
    Examples:
    - Multiple "PRIMARY INTEREST" entries → DELETE all, CREATE "INTEREST [NAME]" for each distinct interest
    - Input: "PRIMARY INTEREST": "photography", "SECONDARY INTEREST": "cooking"
    - Output: DELETE both, CREATE "INTEREST PHOTOGRAPHY" and "INTEREST COOKING"

    ### 3. Ownership and Relationships
    When consolidating related interests (e.g., "dogs", "cats", "pets"):
    - Merge into broader category if appropriate: DELETE specific ones, CREATE "INTEREST PETS" with value "User enjoys dogs and cats"
    - OR keep separate if distinct: CREATE "INTEREST DOGS" and "INTEREST CATS"

    ### 4. Evolution vs Different Items
    - Evolution/refinement: DELETE old, KEEP most complete/current version
    - Different items: DELETE old vague names, CREATE separate features with descriptive names
    
    **Note: Any "UPDATE" operation means DELETE old + CREATE new. There is no direct UPDATE command.**

    Example (Evolution): "CAREER GOAL": "become a manager" → "become a senior manager"
    → DELETE old, KEEP "become a senior manager"

    Example (Different): Multiple hobbies
    → DELETE vague "PRIMARY INTEREST" entries, CREATE specific "INTEREST [NAME]" for each

    ### 5. Redundant Information
    DELETE incomplete versions, KEEP complete ones

    ### 6. Multiple Items
    Use consistent, descriptive suffixes. Don't create suffixes until you have 2+ distinct items.

    ## AGGRESSIVE DELETION

    **REMEMBER: This is PROFILE data storage, NOT event logging.**
    
    More memories = more interference = more cognitive load.
    Be aggressive: some distinctions aren't worth maintaining. Delete ruthlessly.
    
    DELETE any memory that describes an action or event rather than a static personal characteristic.
    
    **IMPORTANT:** Vague feature names like "PRIMARY INTEREST" and "SECONDARY INTEREST" should be replaced with specific names like "INTEREST PHOTOGRAPHY", "INTEREST COOKING".

    ## OUTPUT FORMAT

    CRITICAL: Both fields MUST be arrays. NEVER use null/None for any field.

    ### Output Schema
    ```
    <think> your reasoning </think>
    {"consolidated_memories": [...], "keep_memories": [...]}
    ```

    ### Field Descriptions

    **keep_memories** (REQUIRED - must be an array, never null):
    - List of metadata.id values (as strings) for memories to KEEP unchanged
    - Use empty array [] to delete ALL input memories
    - Example: ["123", "456"] keeps memories with those IDs

    **consolidated_memories** (REQUIRED - must be an array, never null):
    - List of NEW memories to create after consolidation
    - Each memory has: {"tag": "...", "feature": "...", "value": "..."}
    - Use empty array [] if no new memories needed

    ### Examples

    Keep one, delete duplicates:
    {"consolidated_memories": [], "keep_memories": ["1"]}

    Delete temporary/sensitive data (delete all):
    {"consolidated_memories": [], "keep_memories": []}

    Keep all unchanged (rare - only if truly distinct):
    {"consolidated_memories": [], "keep_memories": ["1", "2", "3"]}

    Consolidate multiple interests with better naming:
    {"consolidated_memories": [
        {"tag": "interests", "feature": "INTEREST PHOTOGRAPHY", "value": "User enjoys photography as a hobby"},
        {"tag": "interests", "feature": "INTEREST BIKING", "value": "User likes biking"}
    ], "keep_memories": []}
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
