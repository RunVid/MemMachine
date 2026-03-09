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
    # === Core Identity & Contact ===
    "basics": "Basic personal information: full name, date of birth, gender, age, marital status, education level, occupation, and other basic demographic information. IMPORTANT: If information belongs to someone other than the user (e.g., spouse's name, child's date of birth), include ownership in the feature name (e.g., 'SPOUSE FULL NAME', 'CHILD DATE OF BIRTH'). EXCLUDE: Current location, temporary residence, travel status, or any time-bound information.",
    "contacts": "PERMANENT contact information and addresses ONLY: phone numbers, email addresses, permanent home addresses, permanent work addresses, mailing addresses, emergency contacts. IMPORTANT: Always include ownership/relationship in feature names when the contact belongs to someone other than the user (e.g., 'SPOUSE EMAIL', 'CHILD PHONE NUMBER', 'EMERGENCY CONTACT PHONE'). For the user's own contacts, use standard names like 'EMAIL', 'PHONE NUMBER'. EXCLUDE: Temporary addresses (hotels, Airbnbs, vacation rentals), current location, travel destinations.",
    "identities": "Personal identification documents and IDs that identify the USER: employee ID, student ID, insurance member ID, professional license numbers, military ID. These are IDs issued to the person. NEVER STORE: SSN (full or partial), passport numbers, driver's license numbers, or any government-issued ID numbers that could enable identity theft.",
    
    # === Accounts & Memberships ===
    "accounts": "Service accounts and memberships that identify the user's ACCOUNTS with organizations: customer IDs, subscription IDs, membership numbers, loyalty program numbers, utility account numbers, library card numbers. These are account identifiers issued by services/companies. NEVER STORE: Credit card numbers, bank account numbers, routing numbers, CVV codes, PINs, passwords, or any financial credentials.",
    
    # === Preferences & Settings ===
    "preferences": "LONG-TERM user preferences ONLY: preferred contact methods (phone, email, text), communication style preferences, service preferences (appointment times, service providers), payment methods, dietary preferences, accessibility needs, language preferences, notification preferences, time zone, preferred meeting formats. EXCLUDE: Temporary preferences, situational choices, or context-dependent decisions.",
    
    # === Relationships & Network ===
    "relationships": "Personal relationships and family contacts: family members (spouse, children, parents, siblings), close friends, business contacts, authorized representatives, people the user frequently interacts with or makes decisions on behalf of. Include relationship context and relevant contact information or identifiers. IMPORTANT: When storing contact information, ALWAYS use the person's name if available (e.g., 'SARAH EMAIL', 'ALICE PHONE NUMBER'). Only use relationship type (e.g., 'SPOUSE EMAIL', 'FRIEND PHONE NUMBER') if the name is unknown.",
    "services": "Service providers and professional contacts: doctor, lawyer, accountant, dentist, insurance agent, financial advisor, therapist, personal trainer, and other professional service providers. Include contact information, specialties, and relevant details. IMPORTANT: ALWAYS use the provider's name if available (e.g., 'DR SMITH PHONE', 'JOHN LAWYER EMAIL'). Only use service type (e.g., 'DOCTOR PHONE', 'LAWYER EMAIL') if the name is unknown.",
    
    # === Other Information ===
    "others": "Other structured facts that don't fit into the above categories but are still STABLE, PERMANENT, REUSABLE information needed for task completion. Use this tag only when the information clearly doesn't belong to any of the other defined tags (basics, contacts, identities, accounts, preferences, relationships, services). EXCLUDE: Temporary states, current situations, travel information, or any time-bound data.",
}

# Optimized description for task-oriented structured facts
task_assistant_description = """
    You are extracting structured, factual user information from conversations with a task-oriented assistant.
    This information is used to efficiently complete user requests and is typically referenced at the start of sessions.
    
    YOUR ROLE AND CONTEXT
    
    Your Job:
    - Extract structured facts that help the agent quickly access information needed for task completion
    - Store contact information, account details, identities, preferences, relationships, and service providers
    - Enable the agent to complete tasks efficiently without asking for repeated information
    
    Important Context:
    - Episodic memories already contain refined descriptions and atomic claims, including all historical events and temporary states
    - Semantic memory is for STABLE, REUSABLE user information that persists across sessions
    - ALWAYS compare with existing features before creating new ones
    
    CRITICAL TAG RULES:
    - You MUST ONLY use the tags defined in the tags list: basics, contacts, identities, accounts, preferences, relationships, services, others
    - DO NOT create new tags - if information doesn't fit perfectly, choose the closest matching tag
    - Financial-related information should use "accounts" (for account details) or "preferences" (for financial preferences)
    - Use "others" tag only when the information clearly doesn't belong to any of the other defined tags
    
    WHAT TO EXTRACT
    
    Extract These Stable Structured Facts:
    - PERMANENT contact information (phone, email, permanent home/work addresses)
    - Non-sensitive account information (membership IDs, customer IDs, subscription IDs)
    - Non-sensitive identity documents (employee ID, student ID, insurance member ID)
    - LONG-TERM preferences (communication methods, service preferences, payment methods)
    - Relationship information that remains stable (family members, close contacts)
    - Service provider relationships with contact information
    
    CRITICAL: Do NOT Extract These (They belong in episodic memory):
    
    LOCATION & TRAVEL (NEVER EXTRACT):
    - Current location or whereabouts (e.g., "user is at the airport", "user is in Chicago")
    - Temporary residence (e.g., "staying at Airbnb at 123 Main St", "hotel room 405")
    - Travel destinations or itineraries (e.g., "flying to Paris tomorrow")
    - Vacation rentals, hotels, short-term accommodations
    - "Current address" if it's temporary (use only PERMANENT addresses like "HOME ADDRESS")
    
    TEMPORAL STATES (NEVER EXTRACT):
    - Historical events or past actions (e.g., "booked a flight on 2026-01-23")
    - Temporary states or pending actions (e.g., "flight_booking_pending", "waiting for delivery")
    - One-time transactions or specific occurrences (e.g., "made a purchase", "ordered food")
    - Time-specific information that will become outdated (e.g., "currently traveling", "on vacation")
    - Any information with dates that indicate it's temporary (e.g., "staying until March 15")
    
    TRANSIENT INFORMATION (NEVER EXTRACT):
    - Travel history, booking history, transaction history, or any event-based information
    - Temporary preferences or context-dependent choices (e.g., "wants pizza today" vs. stable "prefers Italian food")
    - Current projects or tasks in progress
    - Appointments, meetings, or scheduled events (these are calendar items, not profile data)
    
    ASK YOURSELF: "Will this information still be accurate in 6 months?"
    - If YES → Extract it
    - If NO → Do NOT extract (it belongs in episodic memory)
    
    ## CRITICAL: NEVER Extract Sensitive Information
    
    For security reasons, NEVER extract or store ANY of the following (not even partial):
    - Social security numbers (full or partial)
    - Passport numbers, driver's license numbers
    - Credit card numbers (full or partial)
    - Bank account numbers, routing numbers
    - CVV/security codes, expiration dates
    - Passwords, PINs, security questions, authentication credentials
    - Biometric data, medical records, detailed financial records
    - Any information that could enable identity theft or fraud
    
    If the user provides sensitive information, DO NOT EXTRACT IT AT ALL:
    - "My SSN is 123-45-6789" → DO NOT EXTRACT
    - "My credit card is 4111-1111-1111-1234" → DO NOT EXTRACT
    - "My password is secret123" → DO NOT EXTRACT
    
    OK to store (non-sensitive IDs):
    - Employee ID, student ID, insurance member ID, loyalty program number
    - Subscription IDs, customer IDs, membership numbers
    
    FEATURE NAMING RULES
    
    Format Rules:
    - Use UPPERCASE letters with SPACES between words (e.g., "PHONE NUMBER", "EMAIL")
    - Use full words, not abbreviations
    - Be specific and descriptive
    
    Standard Feature Names (User's Own Information):
    - "FULL NAME" (not "NAME", "USER NAME", "USERNAME")
    - "EMAIL" (not "EMAIL ADDRESS", "CONTACT EMAIL", "E-MAIL")
    - "PHONE NUMBER" (not "PHONE", "MOBILE", "TELEPHONE")
    - "EMPLOYEE ID", "STUDENT ID", "MEMBER ID" (non-sensitive IDs - store fully)
    - "PREFERRED PAYMENT METHOD" (not "PAYMENT", "PREFERENCE")
    - "TIMEZONE" (not "TZ", "TIME ZONE")
    - "DATE OF BIRTH" (not "DOB", "BIRTHDATE", "BIRTH DATE")
    - "HOME ADDRESS" (not "ADDRESS", "HOME", "RESIDENCE")
    
    Multiple Accounts of the Same Type:
    - If the user has multiple accounts of the same type, use SUFFIXES to distinguish them
    - Examples:
      * Multiple emails: "EMAIL WORK", "EMAIL PERSONAL"
      * Multiple phones: "PHONE NUMBER WORK", "PHONE NUMBER PERSONAL", "MOBILE PHONE", "HOME PHONE"
      * Multiple addresses: "HOME ADDRESS", "WORK ADDRESS", "MAILING ADDRESS"
      * Multiple IDs: "EMPLOYEE ID", "STUDENT ID", "INSURANCE MEMBER ID"
    
    Feature Names with Ownership (Information Belonging to Others):
    Format: "[OWNER] [INFORMATION TYPE]"
    
    Priority Rule: ALWAYS use the person's name if available, instead of relationship type.
    - If name is known: Use "[NAME] [INFORMATION TYPE]" (e.g., "ALICE PHONE NUMBER", "SARAH EMAIL")
    - If name is unknown: Use relationship type as fallback (e.g., "FRIEND PHONE NUMBER", "COLLEAGUE EMAIL")
    
    Examples by Category:
    - Family: "SARAH EMAIL" (spouse, name known) OR "SPOUSE EMAIL" (name unknown)
    - Children: "JOHN PHONE NUMBER" (name known) OR "CHILD PHONE NUMBER" (name unknown)
      * Multiple children: "JOHN PHONE NUMBER", "MARY PHONE NUMBER" (names known) OR "CHILD 1 PHONE NUMBER", "CHILD 2 PHONE NUMBER" (names unknown)
    - Parents: "MOTHER SARAH PHONE" (name known) OR "MOTHER PHONE NUMBER" (name unknown)
    - Siblings: "ALICE PHONE NUMBER" (name known) OR "SIBLING PHONE NUMBER" (name unknown)
    - Emergency contacts: "ALICE PHONE NUMBER" (name known) OR "EMERGENCY CONTACT PHONE NUMBER" (name unknown)
    - Friends: "ALICE PHONE NUMBER" (name known) OR "FRIEND PHONE NUMBER" (name unknown)
    - Service providers: "DR SMITH PHONE" (name known) OR "DOCTOR PHONE" (name unknown)
    - Work contacts: "JOHN MANAGER EMAIL" (name known) OR "MANAGER EMAIL" (name unknown)
    
    HANDLING DUPLICATES AND UPDATES
    
    Before Adding or Updating a Feature:
    1. ALWAYS compare with existing features to check for duplicates or updates
    2. Analyze the claims (content) to determine if it's the same information or different
    
    Decision Rules Based on Claims:
    - If the claim represents the SAME information (same value, same meaning): OVERWRITE the existing feature using UPDATE command
      * Example: Existing "EMAIL" with value "user@example.com", new claim mentions "email address user@example.com" → UPDATE "EMAIL"
    - If the claim represents DIFFERENT information (different value, different account): Create a new feature with a different suffix
      * Example: Existing "EMAIL" with value "user@example.com", new claim mentions "work email user@work.com" → UPDATE "EMAIL" to "EMAIL PERSONAL" and ADD "EMAIL WORK" with value "user@work.com"
    
    Handling Multiple Accounts of the Same Type:
    - If existing features already have suffixes (e.g., "EMAIL WORK"), and new information is about a different account, create a new feature with a different suffix (e.g., "EMAIL PERSONAL")
    - If existing feature has no suffix (e.g., "EMAIL") and new information is about a different account of the same type:
      * Determine which is which based on claims (e.g., "work email" vs "personal email")
      * UPDATE the existing one to add appropriate suffix (e.g., "EMAIL PERSONAL")
      * ADD the new one with different suffix (e.g., "EMAIL WORK")
    
    Reusing Feature Names:
    - If an existing feature name means the same thing, USE THAT EXACT NAME
    - Do not create synonyms or variations
    - Compare with existing features first - reuse existing feature names when the information matches
    - For multiple accounts: Check if a suffix already exists, and use consistent suffix naming
    
    EXTRACTION PROCESS
    
    Step-by-Step Process:
    1. Compare with existing features to identify duplicates or updates
    2. Analyze claims (content) to determine if information is the same or different
    3. **CRITICAL: Select the correct tag from the defined list (basics, contacts, identities, accounts, preferences, relationships, services, others) - DO NOT create new tags**
    4. Use standard feature names (see FEATURE NAMING RULES above)
    5. Include ownership prefix if information belongs to someone else
    6. **CRITICAL: For ownership, ALWAYS use the person's name if available (e.g., "ALICE PHONE NUMBER") instead of relationship type (e.g., "FRIEND PHONE NUMBER"). Only use relationship type if the name is unknown.**
    7. **CRITICAL: For duplicate feature names, decide based on claims: if same information → OVERWRITE (UPDATE), if different information → create new with suffix (e.g., "EMAIL WORK" vs "EMAIL PERSONAL")**
    8. Extract ALL structured facts, even basic ones like names and contact details
    9. For non-sensitive IDs (employee ID, student ID, etc.), store fully; for sensitive info, DO NOT store at all
    10. Include relationship context when extracting family/contact information
    11. Extract service provider information with contact details and specialties
    
    Priority Order:
    1. Contact information needed for task completion (contacts, basics)
    2. Account and identity information (accounts, identities)
    3. User preferences that affect task execution (preferences)
    4. Relationship and service provider information (relationships, services)
    
    Remember: Extract stable, reusable facts that can be quickly referenced at the start 
    of sessions to complete tasks efficiently. If it's a one-time event or temporary 
    state, it belongs in episodic memory, not semantic memory.
"""

# Custom consolidation prompt for task-oriented structured facts
task_assistant_consolidation_prompt = """
    Your job is to perform memory consolidation for a task-oriented structured facts memory system.
    Despite the name, consolidation is not solely about reducing the amount of memories, but rather, minimizing interference between structured facts while maintaining data integrity and usability.
    By consolidating memories, we remove unnecessary couplings of information from context, spurious correlations inherited from the circumstances of their acquisition.

    You will receive a set of task-oriented structured facts memories which are semantically similar (same tag and feature name).
    Produce a new list of memories to keep.

    A memory is a json object with 4 fields:
    - tag: broad category of memory (basics, contacts, identities, accounts, preferences, relationships, services, others)
    - feature: feature name (e.g., "EMAIL", "PHONE NUMBER", "FULL NAME")
    - value: detailed contents of the memory
    - metadata: object with 1 field
    -- id: integer

    You will output consolidated memories, which are json objects with 4 fields:
    - tag: string (must be one of: basics, contacts, identities, accounts, preferences, relationships, services, others)
    - feature: string (must follow FEATURE NAMING RULES below)
    - value: string (detailed contents)
    - metadata: object with 1 field
    -- citations: list of ids of old memories which influenced this one

    You will also output a list of old memories to keep (memories are deleted by default).

    CRITICAL TAG RULES:
    - You MUST ONLY use the tags defined in the tags list: basics, contacts, identities, accounts, preferences, relationships, services, others
    - DO NOT create new tags - if information doesn't fit perfectly, choose the closest matching tag
    - Financial-related information should use "accounts" (for account details) or "preferences" (for financial preferences)
    - Use "others" tag only when the information clearly doesn't belong to any of the other defined tags

    FEATURE NAMING RULES (MUST FOLLOW):
    - Use UPPERCASE letters with SPACES between words (e.g., "PHONE NUMBER", "EMAIL")
    - Use full words, not abbreviations
    - Be specific and descriptive

    Standard Feature Names (User's Own Information):
    - "FULL NAME" (not "NAME", "USER NAME", "USERNAME")
    - "EMAIL" or "EMAIL WORK", "EMAIL PERSONAL" (for multiple emails)
    - "PHONE NUMBER" or "PHONE NUMBER WORK", "PHONE NUMBER PERSONAL" (for multiple phones)
    - "EMPLOYEE ID", "STUDENT ID", "MEMBER ID" (non-sensitive IDs - store fully)

    Feature Names with Ownership (Information Belonging to Others):
    - Priority Rule: ALWAYS use the person's name if available (e.g., "ALICE PHONE NUMBER") instead of relationship type (e.g., "FRIEND PHONE NUMBER")
    - If name is unknown: Use relationship type as fallback (e.g., "SPOUSE EMAIL", "CHILD PHONE NUMBER")
    - For service providers: Use provider's name if available (e.g., "DR SMITH PHONE") instead of service type (e.g., "DOCTOR PHONE")

    CONSOLIDATION GUIDELINES:

    0. **DELETE SENSITIVE AND TEMPORARY INFORMATION (HIGHEST PRIORITY)**:
    
       SENSITIVE INFORMATION - DELETE IMMEDIATELY (no partial storage allowed):
       - DELETE: Any social security numbers (full or partial)
       - DELETE: Any passport numbers, driver's license numbers
       - DELETE: Any credit card numbers (full or partial), bank account numbers, routing numbers
       - DELETE: Passwords, PINs, security questions, authentication credentials, CVV codes
       - DELETE: Biometric data, detailed medical records, detailed financial records
       - KEEP: Non-sensitive IDs like employee ID, student ID, member ID, customer ID
       
       TEMPORARY/TRANSIENT INFORMATION - DELETE:
       - DELETE: "USER LOCATION", "CURRENT ADDRESS", "CURRENT LOCATION", "STAYING AT", or similar
       - DELETE: Airbnb addresses, hotel rooms, vacation rentals, temporary residences
       - DELETE: Any information with specific dates that indicate it's time-bound
       - DELETE: Travel itineraries, current trips, "currently at" information
       - KEEP ONLY: Permanent addresses like "HOME ADDRESS", "WORK ADDRESS"
       - Example: "USER LOCATION": "staying at Airbnb at 123 Main St" → DELETE entirely
       - Example: "USER LOCATION_2": "User's current residence is the Airbnb..." → DELETE entirely
       - ASK: "Will this still be true in 6 months?" If NO → DELETE

    1. **Identical Information (Same Value & Meaning)**: 
       - If memories have identical values and meanings, DELETE duplicates and KEEP only one
       - Example: Multiple "EMAIL" features with value "user@example.com" → Keep one, delete others

    2. **Different Information (Different Value or Distinct Account)**:
       - If memories have different values for the same feature name, they represent different accounts
       - UPDATE feature names to use appropriate suffixes (e.g., "EMAIL WORK", "EMAIL PERSONAL")
       - Example: "EMAIL": "user@example.com" and "EMAIL": "work@example.com" → 
         * Keep "EMAIL PERSONAL": "user@example.com"
         * Keep "EMAIL WORK": "work@example.com"
         * Delete original "EMAIL" entries

    3. **Ownership Consolidation**:
       - If memories have the same information but different ownership representations:
         * Prefer name-based features over relationship-based features
         * Example: "FRIEND PHONE NUMBER": "123-456-7890" and "ALICE PHONE NUMBER": "123-456-7890" → 
           Keep "ALICE PHONE NUMBER", delete "FRIEND PHONE NUMBER"

    4. **Redundant Information**:
       - Memories containing only redundant information should be deleted entirely
       - If information has been processed into a more complete memory, delete the incomplete versions

    5. **Feature Name Synchronization**:
       - If memories are sufficiently similar but differ in key details, synchronize their feature names
       - Use consistent naming across similar memories
       - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field

    6. **Multiple Accounts Handling**:
       - If enough memories share similar features but represent different accounts, use suffixes to distinguish them
       - Don't create suffixes too early. Have at least two distinct accounts first
       - Use consistent suffix naming (e.g., "EMAIL WORK", "EMAIL PERSONAL", not "EMAIL OFFICE", "EMAIL HOME")

    Overall memory life-cycle:
    raw structured facts -> extracted features -> features sorted by tag -> consolidated structured profiles

    The more memories you receive, the more interference there is in the memory system.
    This causes cognitive load. Cognitive load is bad.
    To minimize this, under such circumstances, you need to be more aggressive about deletion:
        - Be looser about what you consider to be similar. Some distinctions are not worth the energy to maintain.
        - Massage out the parts to keep and ruthlessly throw away the rest
        - There is no free lunch here! At least some information must be deleted!

    Do not create new tag names. Only use: basics, contacts, identities, accounts, preferences, relationships, services, others.

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
