"""Prompt template for task-oriented structured facts semantic memory extraction.

This prompt focuses on extracting stable, structured facts needed for task completion:
contact information, account details, identities, preferences, relationships, and services.
These facts are typically listed at the beginning of a session for quick reference.
"""

from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    SemanticCategory,
)
from memmachine.semantic_memory.util.semantic_prompt_template import (
    build_update_prompt,
)

# Task-oriented structured facts tags
task_assistant_tags: dict[str, str] = {
    "basics": "Basic personal information about the user: full name, date of birth, gender, age, marital status, education level, occupation, and other demographic information.",
    "contacts": "Permanent contact information: phone numbers, email addresses, home address, work address, mailing address, and emergency contacts.",
    "identities": "Personal identification documents and IDs: employee ID, student ID, insurance member ID, professional license numbers, and other non-government issued identifiers.",
    "accounts": "Service accounts and memberships: customer IDs, subscription IDs, membership numbers, loyalty program numbers, utility account numbers, and library card numbers.",
    "preferences": "Long-term user preferences: preferred contact methods, communication style preferences, dietary preferences, accessibility needs, language preferences, timezone, and notification settings.",
    "relationships": "Personal relationships and family: family members (spouse, children, parents), close friends, business contacts, and authorized representatives with their contact information.",
    "services": "Service providers and professionals: doctor, lawyer, accountant, dentist, insurance agent, financial advisor, and other professional contacts with their specialties and contact information.",
    "others": "Other stable structured facts that don't fit the above categories but are still permanent, reusable information needed for task completion.",
}

# Optimized description for task-oriented structured facts
task_assistant_description = """
    You are extracting structured, factual user information from conversations with a task-oriented assistant.
    This information is used to efficiently complete user requests and is typically referenced at the start of sessions.
    
    ## YOUR ROLE
    
    - Extract structured facts that help the agent quickly access information needed for task completion
    - Store contact information, account details, identities, preferences, relationships, and service providers
    - Enable the agent to complete tasks efficiently without asking for repeated information
    
    Important: Semantic memory is for STABLE, REUSABLE information that persists across sessions.
    ALWAYS compare with existing features before creating new ones.
    
    ## TAG RULES
    
    You MUST ONLY use: basics, contacts, identities, accounts, preferences, relationships, services, others
    - DO NOT create new tags - choose the closest matching tag
    - Use "others" only when information clearly doesn't belong elsewhere
    - CRITICAL: Tags are case-sensitive and MUST be lowercase (e.g., "accounts" NOT "ACCOUNTS" or "Accounts")
    
    ## WHAT TO EXTRACT
    
    **ONLY extract STATIC, FACTUAL DATA VALUES - NOT actions or events.**
    
    Stable, long-term information:
    - Permanent contact info: phone numbers, email addresses, home/work addresses
    - Personal IDs: employee ID, student ID, insurance member ID, professional licenses
    - Service accounts: customer IDs, subscription IDs, membership numbers, loyalty programs
    - Long-term preferences: communication methods, dietary needs, accessibility, timezone
    - Relationships: family members, close friends, business contacts with their info
    - Service providers: doctor, lawyer, accountant with their contact details
    
    **CRITICAL: Extract ONLY the data value itself, NOT the action or event.**
    
    Examples of CORRECT extraction:
    - "My email is john@example.com" → feature="EMAIL", value="john@example.com" ✓
    - "My employee ID is E12345" → feature="EMPLOYEE ID", value="E12345" ✓
    - "I prefer email for communication" → feature="PREFERRED CONTACT METHOD", value="email" ✓
    
    ## WHAT NOT TO EXTRACT
    
    ### Temporary/Transient Information (belongs in episodic memory)
    - Current location, travel status, temporary residence (hotels, Airbnbs)
    - Historical events, past actions, one-time transactions
    - Pending states, appointments, scheduled events
    - Context-dependent choices (e.g., "wants pizza today" vs stable "prefers Italian food")
    - Actions or events: "User confirmed X", "User updated Y", "User verified Z"
    - Timestamps of actions: "confirmed on March 12", "updated at 3pm"
    
    Examples of INCORRECT extraction (DO NOT DO THIS):
    - "User confirmed email on March 12" → DO NOT EXTRACT (this is an action/event)
    - "User updated phone number" → DO NOT EXTRACT (this is an action/event)
    - "User verified address today" → DO NOT EXTRACT (this is an action/event)
    
    ### Highly Sensitive PII (never store for security)
    - Government IDs: SSN, passport numbers, driver's license numbers
    - Financial: credit card numbers, bank account numbers, routing numbers
    - Security: passwords, PINs, security questions, authentication credentials
    - Private records: medical records, financial records, legal documents
    - Biometric data, complete addresses with unit/apartment numbers
    
    Examples:
    - "My SSN is 123-45-6789" → DO NOT EXTRACT (sensitive)
    - "Staying at Airbnb at 123 Main St" → DO NOT EXTRACT (temporary)
    - "I live at 123 Main St, Apt 4B" → Extract "123 Main St" as HOME ADDRESS (no unit)
    - "User confirmed email on March 12" → DO NOT EXTRACT (action/event)
    - "User verified phone number today" → DO NOT EXTRACT (action/event)
    
    ## FEATURE NAMING RULES
    
    ### Format
    - MUST Use UPPERCASE letters with SPACES between words (e.g., "PHONE NUMBER", "EMAIL")
    - MUST NOT use underscores or other special characters in feature names
    - Use full words, not abbreviations
    - Be specific and descriptive
    
    ### Standard Feature Names
    - "FULL NAME", "EMAIL", "PHONE NUMBER", "DATE OF BIRTH", "HOME ADDRESS", "TIMEZONE"
    - "EMPLOYEE ID", "STUDENT ID", "MEMBER ID" (non-sensitive IDs - store fully)
    - "PREFERRED PAYMENT METHOD"
    
    ### Multiple Accounts
    Use SUFFIXES to distinguish:
    - Emails: "EMAIL WORK", "EMAIL PERSONAL"
    - Phones: "PHONE NUMBER WORK", "PHONE NUMBER PERSONAL"
    - Addresses: "HOME ADDRESS", "WORK ADDRESS", "MAILING ADDRESS"
    
    ### Ownership (Information Belonging to Others)
    Format: "[OWNER] [INFORMATION TYPE]"
    
    Priority: ALWAYS use person's name if available, otherwise use relationship type.
    - Name known: "ALICE PHONE NUMBER", "SARAH EMAIL", "DR SMITH PHONE"
    - Name unknown: "SPOUSE EMAIL", "CHILD PHONE NUMBER", "DOCTOR PHONE"
    
    ## HANDLING DUPLICATES AND UPDATES
    
    Before adding new features:
    1. Compare with existing features to check for duplicates
    2. Analyze claims to determine if it's the same or different information
    
    ### Decision Rules
    - SAME information (same value/meaning): Do NOT add duplicate - skip it
    - UPDATED information (same feature, new value): DELETE old, ADD new
    - DIFFERENT information (different account/value): ADD new with different suffix
    
    ### Examples
    
    **Example 1: Exact Duplicate (Skip)**
    - Existing: feature="EMAIL", value="user@example.com"
    - New claim: "My email is user@example.com"
    → Skip (exact duplicate)
    
    **Example 2: Same Feature, Updated Value (DELETE + ADD)**
    - Existing: feature="PHONE NUMBER", value="555-1234"
    - New claim: "My phone number changed to 555-5678"
    → DELETE "PHONE NUMBER" (old), ADD "PHONE NUMBER" with value="555-5678"
    
    **Example 3: Multiple Accounts (ADD new with suffix)**
    - Existing: feature="EMAIL", value="user@example.com"
    - New claim: "My work email is user@work.com"
    → ADD "EMAIL WORK" (user@work.com)
    
    **Example 4: Discovering Multiple Accounts (ADD new with suffixes)**
    - Existing: feature="PHONE NUMBER", value="555-1234"
    - New claim: "That's my home number. My work number is 555-5678"
    → ADD "PHONE NUMBER WORK" (555-5678)
    
    **REMEMBER: NEVER have two features with the SAME feature name. Update (Delete old + Add new) or Use suffixes to distinguish them.**
    
    ## EXTRACTION PROCESS
    
    **REMEMBER: Extract ONLY static data values, NEVER actions or events.**
    
    1. Compare with existing features to identify duplicates or updates
    2. Select the correct tag (DO NOT create new tags)
    3. Use standard feature names
    4. Include ownership prefix if information belongs to someone else (use name if available)
    5. For duplicates: same info → skip, updated info → DELETE old + ADD new, different accounts → ADD new with suffix
    6. Extract ONLY the factual data value (e.g., "john@example.com"), NOT the action (e.g., "User confirmed email")
    7. For non-sensitive IDs, store; for sensitive info, DO NOT store
    8. Always Check before returning the output: 
        a. MUST NOT add two features with the same feature name under the same tag - only keep one if they are the same, use suffixes to distinguish if they are different. For update operation, do not forget to delete the old feature.
        b. MUST NOT add temporary information (belongs in episodic memory).
        c. MUST NOT add sensitive Personal Identifiable Information (PII).
"""

# Custom consolidation prompt for task-oriented structured facts
task_assistant_consolidation_prompt = """
    You are performing memory consolidation for a task-oriented structured facts memory system.
    Consolidation minimizes interference between structured facts while maintaining data integrity.
    
    ## INPUT/OUTPUT FORMAT
    
    All input memories have the SAME tag. All outputs MUST use this SAME tag.**

    ## CONSOLIDATION RULES

    ### Feature Names Format
    - UPPERCASE with SPACES (e.g., "PHONE NUMBER", "EMAIL")
    - Use suffixes for multiple accounts: "EMAIL WORK", "EMAIL PERSONAL"
    - For ownership: prefer name over relationship ("ALICE PHONE NUMBER" > "SPOUSE EMAIL")

    ### Step 1: DELETE First (Highest Priority)
    
    **Actions/Events - Must DELETE:**
    - "User confirmed X on [date]", "User verified Y", "User updated Z"
    - ASK: "Is this a static data value or an action?" If ACTION → DELETE
    
    **Sensitive PII - Must DELETE:**
    - SSN, passport numbers, driver's license, credit cards, bank accounts
    - Passwords, PINs, medical records, financial records
    
    **Temporary Information - Must DELETE:**
    - Current location, Airbnb/hotel addresses, vacation rentals
    - Travel itineraries, time-bound information
    - ASK: "Will this still be true in 6 months?" If NO → DELETE
    
    **OK to Keep:**
    - Contact info: names, emails, phones, general addresses (no unit numbers)
    - Non-sensitive IDs: employee ID, student ID, member ID, customer ID
    - Long-term preferences, birthdate, occupation
    - ONLY if the value is actual data (e.g., "john@example.com"), NOT action description

    ### Step 2: Group and Consolidate
    
    **Same feature name + same/similar value:**
    - Exact duplicates → DELETE duplicates, KEEP only one
    - Nearly identical → DELETE all, CREATE one consolidated version
    
    Example:
    - feature="EMAIL", value="user@example.com" (appears twice)
    → DELETE duplicate, KEEP: {"tag": "contacts", "feature": "EMAIL", "value": "user@example.com"}
    
    - feature="FULL NAME", value="John Smith" / "John D. Smith"
    → DELETE both, CREATE: {"tag": "basics", "feature": "FULL NAME", "value": "John D. Smith"}
    
    **Same feature name + different value:**
    - If different accounts → DELETE old, CREATE new with suffixes
    - If evolution → DELETE old, KEEP new (most complete/current)
    
    Example (Different accounts):
    - feature="EMAIL", value="personal@email.com" / "work@company.com"
    → DELETE both, CREATE:
      {"tag": "contacts", "feature": "EMAIL PERSONAL", "value": "personal@email.com"}
      {"tag": "contacts", "feature": "EMAIL WORK", "value": "work@company.com"}
    
    Example (Evolution):
    - feature="PHONE NUMBER", value="555-1234" (old) / "555-5678" (new)
    → DELETE old, KEEP: {"tag": "contacts", "feature": "PHONE NUMBER", "value": "555-5678"}
    
    **Multiple Accounts:**
    - Don't create suffixes until you have 2+ distinct accounts
    - Use consistent suffixes: WORK, PERSONAL, HOME

    ### Step 3: Apply Ownership Rules
    - Prefer name-based over relationship-based
    - "ALICE PHONE NUMBER" > "FRIEND PHONE NUMBER"
    - "DR SMITH EMAIL" > "DOCTOR EMAIL"

    ## PROCESSING WORKFLOW
    
    Use your <think> section to follow these steps systematically:
    
    **Step 1: List all inputs**
    Write: "Input: id=X, feature=Z, value=W" for each memory
    
    **Step 2: Identify DELETE candidates**
    List IDs to DELETE (actions, sensitive PII, temporary info) with reason
    Example: "DELETE id=3 (action: 'User confirmed email')"
    
    **Step 3: Group remaining by feature name**
    Group memories with same feature name
    Example: "Group EMAIL: id=1 (personal@email.com), id=2 (work@company.com)"
    
    **Step 4: Decide action per group**
    - Same values → keep one or merge
    - Different values → check if accounts or evolution
    - Apply feature name rules (suffixes, ownership)
    
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
    - feature: UPPERCASE with SPACES (string)
    - value: ONLY the raw data (string) 
    
    The value field must contain ONLY clean data, never include reasoning or explanations.

    ### Output Examples

    Example 1 - Keep one:
    {"consolidated_memories": [], "keep_memories": ["1"]}

    Example 2 - Delete all:
    {"consolidated_memories": [], "keep_memories": []}

    Example 3 - Merge:
    {"consolidated_memories": [
        {"tag": "basics", "feature": "FULL NAME", "value": "John D. Smith"}
    ], "keep_memories": []}

    Example 4 - Multiple accounts with descriptive suffixes:
    {"consolidated_memories": [
        {"tag": "contacts", "feature": "EMAIL PERSONAL", "value": "personal@email.com"},
        {"tag": "contacts", "feature": "EMAIL WORK", "value": "work@company.com"},
        {"tag": "contacts", "feature": "PHONE NUMBER WORK", "value": "+1 510-987-6035"},
        {"tag": "contacts", "feature": "PHONE NUMBER HOME", "value": "+86 21 5067 7716"}
    ], "keep_memories": []}
    
    ## FINAL VALIDATION CHECKLIST
    
    Before returning, verify:
    
    1. All value fields contain ONLY raw data (no reasoning notes)
    2. No duplicate feature names under same tag
    3. Suffixes are descriptive (WORK, HOME, MOBILE) not vague (SECONDARY, TERTIARY)
    4. Ownership naming is consistent
    5. Output is valid JSON with both arrays present
    
    ## REMEMBER
    
    - Be aggressive with deletion: More memories = more interference
    - Profile data storage, NOT event logging
    - Keep value fields CLEAN - only raw data
    - Delete ruthlessly when in doubt
"""

TaskAssistantSemanticCategory = SemanticCategory(
    name="profile",
    prompt=RawSemanticPrompt(
        update_prompt=build_update_prompt(
            tags=task_assistant_tags,
            description=task_assistant_description,
        ),
        consolidation_prompt=task_assistant_consolidation_prompt,
    ),
)

SEMANTIC_TYPE = TaskAssistantSemanticCategory
