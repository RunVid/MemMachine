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
    
    Key Questions:
    1. "Is this a static data value (name, email, ID, preference)?" If YES → extract. If NO → skip.
    2. "Does this describe an action or event?" If YES → skip (episodic memory handles this).
    
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
    7. For non-sensitive IDs, store fully; for sensitive info, DO NOT store
    8. **NEVER have two features with the same feature name** - use suffixes to distinguish
    
    Priority: contacts/basics > accounts/identities > preferences > relationships/services
"""

# Custom consolidation prompt for task-oriented structured facts
task_assistant_consolidation_prompt = """
    You are performing memory consolidation for a task-oriented structured facts memory system.
    Consolidation minimizes interference between structured facts while maintaining data integrity.

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
    - DO NOT change tags during consolidation. If input tag is "accounts", output tag for all consolidation memory MUST be "accounts".   
    - CRITICAL: Tags MUST be lowercase (e.g., "accounts" NOT "ACCOUNTS" or "Accounts")

    ### Feature Names

    - UPPERCASE with SPACES (e.g., "PHONE NUMBER", "EMAIL")
    - No underscores or other special characters in feature names
    - Use suffixes for multiple accounts: "EMAIL WORK", "EMAIL PERSONAL"
    - For ownership: use name if available ("ALICE PHONE NUMBER"), otherwise relationship ("SPOUSE EMAIL")

    ## CONSOLIDATION GUIDELINES

    ### 0. DELETE FIRST (Highest Priority)

    **Actions/Events - DELETE:**
    - "User confirmed X on [date]"
    - "User verified Y today"
    - "User updated Z"
    - Any value describing an action or event instead of static data
    - ASK: "Is this a static data value or an action?" If ACTION → DELETE

    **Highly Sensitive PII - DELETE:**
    - SSN, passport numbers, driver's license numbers
    - Credit card/bank account numbers, routing numbers
    - Passwords, PINs, security questions, credentials
    - Medical records, financial records, legal documents
    - Complete addresses with unit/apartment numbers

    **Temporary Information - DELETE:**
    - "USER LOCATION", "CURRENT ADDRESS", "STAYING AT"
    - Airbnb addresses, hotel rooms, vacation rentals
    - Travel itineraries, time-bound information
    - ASK: "Will this still be true in 6 months?" If NO → DELETE

    **OK to Keep:**
    - Names, emails, phone numbers, general addresses (no unit)
    - Employee ID, student ID, member ID, customer ID
    - Birthdate, occupation, preferences
    - ONLY if the value is the actual data (e.g., "john@example.com"), NOT an action description

    ### 1. Identical or Nearly Identical Information
    
    **Same tag + same feature name + same/similar value:**
    - DELETE all duplicates, KEEP only one (the most complete version)
    - Or DELETE all, CREATE one consolidated version
    
    Examples:
    - tag="contacts", feature="EMAIL", value="user@example.com"
    - tag="contacts", feature="EMAIL", value="user@example.com"
    → DELETE duplicate, KEEP one
    
    - tag="basics", feature="FULL NAME", value="John Smith"
    - tag="basics", feature="FULL NAME", value="John D. Smith"
    → DELETE incomplete, KEEP: {"tag": "basics", "feature": "FULL NAME", "value": "John D. Smith"}
    
    **Same tag + same feature name + different value:**
    - This usually means different accounts or evolution
    - If different accounts: "UPDATE" = DELETE both old + CREATE new with suffixes
    - If evolution: DELETE old, KEEP new
    
    Examples:
    - tag="contacts", feature="EMAIL", value="personal@email.com"
    - tag="contacts", feature="EMAIL", value="work@company.com"
    → DELETE both, CREATE: {"tag": "contacts", "feature": "EMAIL PERSONAL", "value": "personal@email.com"}
                          {"tag": "contacts", "feature": "EMAIL WORK", "value": "work@company.com"}

    ### 2. Different Accounts
    "UPDATE" feature names with suffixes = DELETE old + CREATE new with suffixes (e.g., "EMAIL WORK", "EMAIL PERSONAL")
    
    **Note: "UPDATE" means DELETE the old feature(s) and CREATE new feature(s) with better names.**

    ### 3. Ownership
    Prefer name-based over relationship-based ("ALICE PHONE NUMBER" > "FRIEND PHONE NUMBER")

    ### 4. Redundant Information
    DELETE incomplete versions, KEEP complete ones

    ### 5. Multiple Accounts
    Use consistent suffixes. Don't create suffixes until you have 2+ distinct accounts.

    ## AGGRESSIVE DELETION

    **REMEMBER: This is PROFILE data storage, NOT event logging.**
    
    More memories = more interference = more cognitive load.
    Be aggressive: some distinctions aren't worth maintaining. Delete ruthlessly.
    
    DELETE any memory that describes an action, event, or confirmation rather than a static data value.

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

    Delete sensitive data (delete all):
    {"consolidated_memories": [], "keep_memories": []}

    Keep all unchanged:
    {"consolidated_memories": [], "keep_memories": ["1", "2", "3"]}

    Merge into new memory:
    {"consolidated_memories": [{"tag": "contacts", "feature": "EMAIL PRIMARY", "value": "user@example.com"}], "keep_memories": []}
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
