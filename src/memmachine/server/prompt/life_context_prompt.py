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
    "general_preference": "Broad, cross-cutting tastes and defaults: general likes and dislikes, environment and pacing preferences (quiet vs busy, urban vs nature), how they prefer advice or explanations (brief vs detailed, direct vs gentle), aesthetic or stylistic taste, learning or information consumption preferences when none of these are a specific hobby (interests) or habit (lifestyle). Use for softer, general preferences—not structured contact/account facts (those belong in task-oriented memory).",
}

# Optimized description for life-oriented personal context
life_context_description = """
    You are extracting life-oriented personal context from conversations with a life advisor assistant.
    This information is used to provide personalized life guidance and understand the user's psychological profile.
    
    ## YOUR ROLE
    
    - Extract personal insights: WHY and HOW the user thinks, feels, and behaves
    - Store interests, lifestyle patterns, goals, personality traits, life circumstances, and general_preference items
    - Build a psychological profile that enables meaningful, personalized advice
    
    Do NOT Extract:
    - WHAT the user has (contact info, account numbers) → task_assistant_prompt extracts those
    - Historical events or temporary states → episodic memory stores those
    
    Important: Semantic memory is for STABLE, REUSABLE information.
    ALWAYS compare with existing features before creating new ones.
    
    ## TAG RULES
    
    You MUST ONLY use: interests, lifestyle, goals, personality, life_situation, general_preference
    - interests: hobbies, passions, concrete activities they enjoy
    - lifestyle: routines, habits, work-life balance
    - goals: aspirations, objectives, plans
    - personality: character traits, behavioral patterns
    - life_situation: stable circumstances, context
    - general_preference: broad tastes and defaults (how they like advice, environments, pacing, aesthetics)—not specific hobbies or daily habits
    - DO NOT create new tags - choose the closest matching tag
    - CRITICAL: Tags are case-sensitive and MUST be lowercase (e.g., "interests" NOT "INTERESTS" or "Interests")
    
    ## WHAT TO EXTRACT
    
    **ONLY extract STATIC, FACTUAL DATA VALUES - NOT actions or events.**
    
    Stable, long-term information:
    - Interests: hobbies, passions, creative pursuits, learning interests, entertainment tied to specific activities
    - Lifestyle: daily routines, sleep patterns, exercise habits, dietary habits, work-life balance
    - Goals: career aspirations, personal development goals, health goals, life vision, educational goals
    - Personality: communication style, decision-making style, introversion/extroversion, emotional patterns
    - Life Situation: living situation (city/region), family structure, work situation, major life stage
    - General preference (tag general_preference): general tastes and defaults (e.g., prefers minimal explanations, likes quiet spaces, prefers gentler feedback)
    - Values: core values, priorities, what drives decisions
    
    **CRITICAL: Extract ONLY the personal characteristic itself, NOT the action or event.**
    
    Examples of CORRECT extraction:
    - "I exercise 3 times a week" → feature="EXERCISE HABIT", value="exercises 3 times a week" ✓
    - "My goal is to become a senior manager" → feature="CAREER GOAL", value="become a senior manager" ✓
    - "I prefer working alone" → feature="SOCIAL PREFERENCE", value="prefers working alone" ✓ (personality)
    - "I like short bullet answers, not long essays" → tag=general_preference, feature="PREFERENCE EXPLANATION DEPTH", value="prefers concise bullet-style answers" ✓
    
    Key Questions:
    1. "Is this a stable personal characteristic (interest, habit, goal, trait, preference)?" If YES → extract. If NO → skip.
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
    
    **general_preference:** (broad tastes—use when it is a default or style, not a hobby or routine)
    - "PREFERENCE ADVICE STYLE", "PREFERENCE ENVIRONMENT", "PREFERENCE EXPLANATION DEPTH", "PREFERENCE AESTHETIC"
    - Multiple distinct preferences: suffix descriptively, e.g. "PREFERENCE INFORMATION DENSITY", "PREFERENCE FEEDBACK TONE"
    
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
    8. Always Check before returning the output: 
        a. MUST NOT add two features with the same feature name under the same tag - only keep one if they are the same, use suffixes to distinguish if they are different. For update operation, do not forget to delete the old feature.
        b. MUST NOT add temporary information (belongs in episodic memory).
        c. MUST NOT add sensitive Personal Identifiable Information (PII).
"""

# Custom consolidation prompt for life-oriented personal context
life_context_consolidation_prompt = """
    You are performing memory consolidation for a life-oriented personal context memory system.
    Consolidation minimizes interference between personal insights while maintaining psychological profile integrity.
    
    ## INPUT/OUTPUT FORMAT
    
    All input memories have the SAME tag. All outputs MUST use this SAME tag.

    ## CONSOLIDATION RULES

    ### Feature Names Format
    - UPPERCASE with SPACES (e.g., "INTEREST PHOTOGRAPHY", "CAREER GOAL")
    - Use descriptive suffixes for multiple items: "INTEREST PHOTOGRAPHY", "INTEREST COOKING"
    - Standard patterns: "INTEREST [NAME]", "EXERCISE HABIT", "CAREER GOAL", "COMMUNICATION STYLE", "PREFERENCE [AREA]"
    - Avoid vague names: use "INTEREST PHOTOGRAPHY" not "PRIMARY INTEREST"

    ### Step 1: DELETE First (Highest Priority)
    
    **IMPORTANT: DELETE means NOT putting the ID in keep_memories array**
    
    **Actions/Events - Must DELETE (don't put in keep_memories):**
    - "User started X on [date]", "User decided Y", "User completed Z"
    - ASK: "Is this a static characteristic or an action?" If ACTION → DELETE (exclude from keep_memories)
    
    **Sensitive PII - Must DELETE (don't put in keep_memories):**
    - SSN, passport numbers, driver's license, credit cards
    - Passwords, PINs, medical records, financial records
    
    **Temporary Information - Must DELETE (don't put in keep_memories):**
    - Current location, hotel stays, Airbnb addresses
    - Time-bound information, current projects
    
    **OK to Keep (can put in keep_memories):**
    - Long-term interests, stable lifestyle patterns
    - Personality traits, life goals, core values, general_preference memories
    - Stable life circumstances
    - ONLY if the value is actual characteristic (e.g., "exercises regularly"), NOT action description

    ### Step 2: Group and Consolidate
    
    **CRITICAL: Output MUST NOT have duplicate feature names**
    
    **Same feature name + same/similar value:**
    - Exact duplicates → DELETE duplicates (don't put duplicate IDs in keep_memories), KEEP only one ID
    - Similar meaning → DELETE all (exclude all IDs from keep_memories), CREATE one consolidated version
    
    Example (Exact duplicate):
    - Input: id='1' feature="INTEREST PHOTOGRAPHY", value="User likes photography" + id='2' same value
    → keep_memories=['1'], consolidated_memories=[]
    
    Example (Similar meaning):
    - Input: id='1' feature="INTEREST PHOTOGRAPHY", value="User likes photography" + id='2' value="User enjoys photography"
    → keep_memories=[], consolidated_memories=[{"tag": "interests", "feature": "INTEREST PHOTOGRAPHY", "value": "enjoys photography as a hobby"}]
    
    **Same feature name + different value:**
    - If evolution → DELETE old (exclude old ID from keep_memories), KEEP new ID or CREATE new
    - If conflicting → DELETE all (exclude all IDs from keep_memories), CREATE merged comprehensive value
    
    Example (Evolution):
    - Input: id='3' feature="CAREER GOAL", value="become a manager" + id='7' value="become a senior manager" (newer)
    → keep_memories=['7'], consolidated_memories=[]
    
    **Vague feature names → Rename with specifics:**
    - "PRIMARY INTEREST", "SECONDARY INTEREST" → DELETE all (exclude from keep_memories), CREATE specific names
    
    Example (Rename vague):
    - Input: id='5' feature="PRIMARY INTEREST", value="photography" + id='8' feature="SECONDARY INTEREST", value="cooking"
    → keep_memories=[], consolidated_memories=[
        {"tag": "interests", "feature": "INTEREST PHOTOGRAPHY", "value": "photography"},
        {"tag": "interests", "feature": "INTEREST COOKING", "value": "cooking"}
      ]
    
    **Multiple items of same type:**
    - Use descriptive suffixes with item names
    - Don't create suffixes until you have 2+ distinct items
    
    **KEY RULE: When you CREATE consolidated_memories, the source IDs MUST be deleted (NOT in keep_memories)!**

    ### Step 3: Handle Related Items
    - Related interests (e.g., "dogs", "cats") → Merge into broader OR keep separate
    - Example: DELETE specific, CREATE "INTEREST PETS" with "enjoys dogs and cats"
    - OR: Keep as "INTEREST DOGS", "INTEREST CATS" if distinct enough

    ## PROCESSING WORKFLOW
    
    Use your <think> section to follow these steps systematically:
    
    **Step 1: List all inputs**
    Write: "Input: id=X, feature=Z, value=W" for each memory
    
    **Step 2: Identify DELETE candidates**
    List IDs to DELETE (actions, PII, temporary, vague names) with reason
    Example: "DELETE id=3 (action: 'User started exercising')"
    
    **Step 3: Group remaining by feature or similarity**
    Group memories with same/similar feature names
    Example: "Group INTERESTS: id=1 (photography), id=2 (cooking)"
    
    **Step 4: Decide action per group**
    - Same meaning → keep one or merge
    - Evolution → keep most complete
    - Vague names → DELETE all, CREATE with specific names
    
    **Step 5: Generate output**
    - keep_memories: IDs to keep unchanged
    - consolidated_memories: New memories

    ## OUTPUT FORMAT

    Both fields MUST be arrays. NEVER use null.
    
    Return ONLY valid JSON:
    ```json
    {"consolidated_memories": [...], "keep_memories": [...]}
    ```

    **keep_memories**: Array of ID strings to keep unchanged. Use [] to delete all.
    
    **consolidated_memories**: Array of new memories. Each must include:
    - tag: same as input (lowercase string)
    - feature: UPPERCASE with SPACES, specific names (string)
    - value: ONLY the characteristic itself (string)
    
    The value field must contain ONLY the characteristic, never include reasoning or explanations.

    ### Output Examples

    Example 1 - Keep one:
    {"consolidated_memories": [], "keep_memories": ["1"]}

    Example 2 - Delete all:
    {"consolidated_memories": [], "keep_memories": []}

    Example 3 - Merge similar:
    {"consolidated_memories": [
        {"tag": "interests", "feature": "INTEREST PHOTOGRAPHY", "value": "enjoys photography as a hobby"}
    ], "keep_memories": []}

    Example 4 - Rename vague to specific:
    {"consolidated_memories": [
        {"tag": "interests", "feature": "INTEREST PHOTOGRAPHY", "value": "photography"},
        {"tag": "interests", "feature": "INTEREST COOKING", "value": "cooking"},
        {"tag": "interests", "feature": "INTEREST HIKING", "value": "hiking"}
    ], "keep_memories": []}
    
    ## CONSOLIDATION CONTRACT (MANDATORY)
    
    **IF** you return consolidated_memories with N items:
    **THEN** you MUST exclude AT LEAST N source memory IDs from keep_memories
    
    **WHY**: Consolidated memories are created BY MERGING existing memories. If you keep the source memories 
    AND add consolidated memories, you create DUPLICATES. This corrupts the system.
    
    **RULE**: consolidated_memories.length > 0  =>  (total_memories - keep_memories.length) >= consolidated_memories.length
    
    **If you cannot determine which memories to consolidate, return empty consolidated_memories array.**

    ## FINAL VALIDATION CHECKLIST
    
    Before returning, verify:
    
    1. All value fields contain ONLY characteristics (no reasoning notes)
    2. **NO duplicate feature names** (CRITICAL! Each feature name must be unique in consolidated_memories)
    3. Feature names are specific, not vague (INTEREST PHOTOGRAPHY not PRIMARY INTEREST)
    4. Suffixes are descriptive when needed
    5. **CRITICAL DATA INTEGRITY CHECK**: 
       - If consolidated_memories is NOT EMPTY, you MUST have deleted some source memories
       - Those deleted source memory IDs MUST NOT appear in keep_memories
       - If you cannot identify which source memories to delete, DO NOT create consolidated_memories
       - Example: If consolidating memory IDs [1,2,3] into a new memory, keep_memories MUST NOT contain [1,2,3]
       - Violation = DATA DUPLICATION = SYSTEM CORRUPTION
       - You must check this following CONSOLIDATION CONTRACT before returning the output.
    6. Output is valid JSON with both arrays present
    
    ## REMEMBER
    
    - Be aggressive with deletion: More memories = more interference
    - Profile data storage, NOT event logging
    - Use specific feature names, not vague ones
    - Keep value fields CLEAN - only characteristics
    - Delete ruthlessly when in doubt
    - **NEVER consolidate without deleting source memories**
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
