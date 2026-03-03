"""Prompt template for task-oriented structured facts semantic memory extraction.

This prompt focuses on extracting stable, structured facts needed for task completion:
contact information, account details, identities, preferences, relationships, and services.
These facts are typically listed at the beginning of a session for quick reference.
"""

from memmachine.semantic_memory.semantic_model import (
    SemanticCategory,
    StructuredSemanticPrompt,
)

# Task-oriented structured facts tags
task_assistant_tags: dict[str, str] = {
    # === Core Identity & Contact ===
    "basics": "Basic personal information: full name, date of birth, gender, age, marital status, education level, occupation, and other basic demographic information. IMPORTANT: If information belongs to someone other than the user (e.g., spouse's name, child's date of birth), include ownership in the feature name (e.g., 'SPOUSE FULL NAME', 'CHILD DATE OF BIRTH').",
    "contacts": "Contact information and addresses: phone numbers, email addresses, permanent addresses, mailing addresses, work addresses, emergency contacts. IMPORTANT: Always include ownership/relationship in feature names when the contact belongs to someone other than the user (e.g., 'SPOUSE EMAIL', 'CHILD PHONE NUMBER', 'EMERGENCY CONTACT PHONE'). For the user's own contacts, use standard names like 'EMAIL', 'PHONE NUMBER'.",
    "identities": "Stable identification numbers and documents: social security numbers (last 4 digits), driver's license numbers, passport numbers, tax IDs, employee IDs, student IDs, insurance member IDs.",
    
    # === Financial & Accounts ===
    "accounts": "Account information: account numbers, account holder names, account types, bank account details, credit card information (last 4 digits, card type), subscription account IDs, service account numbers, membership numbers, customer IDs, loyalty program numbers.",
    
    # === Preferences & Settings ===
    "preferences": "User preferences: preferred contact methods (phone, email, text), communication style preferences, service preferences (appointment times, service providers), payment methods, dietary preferences, accessibility needs, language preferences, notification preferences, time zone, preferred meeting formats.",
    
    # === Relationships & Network ===
    "relationships": "Personal relationships and family contacts: family members (spouse, children, parents, siblings), close friends, business contacts, authorized representatives, people the user frequently interacts with or makes decisions on behalf of. Include relationship context and relevant contact information or identifiers. IMPORTANT: When storing contact information, ALWAYS use the person's name if available (e.g., 'SARAH EMAIL', 'ALICE PHONE NUMBER'). Only use relationship type (e.g., 'SPOUSE EMAIL', 'FRIEND PHONE NUMBER') if the name is unknown.",
    "services": "Service providers and professional contacts: doctor, lawyer, accountant, dentist, insurance agent, financial advisor, therapist, personal trainer, and other professional service providers. Include contact information, specialties, and relevant details. IMPORTANT: ALWAYS use the provider's name if available (e.g., 'DR SMITH PHONE', 'JOHN LAWYER EMAIL'). Only use service type (e.g., 'DOCTOR PHONE', 'LAWYER EMAIL') if the name is unknown.",
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
    - You MUST ONLY use the tags defined in the tags list: basics, contacts, identities, accounts, preferences, relationships, services
    - DO NOT create new tags - if information doesn't fit perfectly, choose the closest matching tag
    - Financial-related information should use "accounts" (for account details) or "preferences" (for financial preferences)
    
    WHAT TO EXTRACT
    
    Extract These Stable Structured Facts:
    - Permanent contact information (phone, email, addresses)
    - Stable account information (account numbers, IDs - last 4 digits only)
    - Long-term preferences (communication methods, service preferences, payment methods)
    - Relationship information that remains stable (family members, close contacts)
    - Service provider relationships with contact information
    
    Do NOT Extract (These belong in episodic memory):
    - Historical events or past actions (e.g., "booked a flight on 2026-01-23")
    - Temporary states or pending actions (e.g., "flight_booking_pending")
    - One-time transactions or specific occurrences (e.g., "made a purchase")
    - Time-specific information that will become outdated (e.g., "currently traveling")
    - Travel history, booking history, transaction history, or any event-based information
    - Temporary preferences or context-dependent choices (e.g., "wants pizza today" vs. stable "prefers Italian food")
    
    FEATURE NAMING RULES
    
    Format Rules:
    - Use UPPERCASE letters with SPACES between words (e.g., "PHONE NUMBER", "EMAIL")
    - Use full words, not abbreviations
    - Be specific and descriptive
    
    Standard Feature Names (User's Own Information):
    - "FULL NAME" (not "NAME", "USER NAME", "USERNAME")
    - "EMAIL" (not "EMAIL ADDRESS", "CONTACT EMAIL", "E-MAIL")
    - "PHONE NUMBER" (not "PHONE", "MOBILE", "TELEPHONE")
    - "BANK ACCOUNT LAST4" (not "ACCOUNT", "BANK", "ACCOUNT NUMBER")
    - "CREDIT CARD LAST4" (not "CARD", "CARD NUMBER")
    - "PREFERRED PAYMENT METHOD" (not "PAYMENT", "PREFERENCE")
    - "TIMEZONE" (not "TZ", "TIME ZONE")
    - "DATE OF BIRTH" (not "DOB", "BIRTHDATE", "BIRTH DATE")
    - "HOME ADDRESS" (not "ADDRESS", "HOME", "RESIDENCE")
    
    Multiple Accounts of the Same Type:
    - If the user has multiple accounts of the same type, use SUFFIXES to distinguish them
    - Examples:
      * Multiple emails: "EMAIL WORK", "EMAIL PERSONAL" (or "EMAIL WORKING", "EMAIL PRIVATE")
      * Multiple phones: "PHONE NUMBER WORK", "PHONE NUMBER PERSONAL", "MOBILE PHONE", "HOME PHONE"
      * Multiple bank accounts: "BANK ACCOUNT LAST4 CHECKING", "BANK ACCOUNT LAST4 SAVINGS"
      * Multiple credit cards: "CREDIT CARD LAST4 VISA", "CREDIT CARD LAST4 AMEX"
      * Multiple addresses: "HOME ADDRESS", "WORK ADDRESS", "MAILING ADDRESS"
    
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
    3. **CRITICAL: Select the correct tag from the defined list (basics, contacts, identities, accounts, preferences, relationships, services) - DO NOT create new tags**
    4. Use standard feature names (see FEATURE NAMING RULES above)
    5. Include ownership prefix if information belongs to someone else
    6. **CRITICAL: For ownership, ALWAYS use the person's name if available (e.g., "ALICE PHONE NUMBER") instead of relationship type (e.g., "FRIEND PHONE NUMBER"). Only use relationship type if the name is unknown.**
    7. **CRITICAL: For duplicate feature names, decide based on claims: if same information → OVERWRITE (UPDATE), if different information → create new with suffix (e.g., "EMAIL WORK" vs "EMAIL PERSONAL")**
    8. Extract ALL structured facts, even basic ones like names and contact details
    9. For account numbers and IDs, store only the last 4 digits
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

TaskAssistantSemanticCategory = SemanticCategory(
    name="profile",
    prompt=StructuredSemanticPrompt(
        tags=task_assistant_tags,
        description=task_assistant_description,
    ),
)

SEMANTIC_TYPE = TaskAssistantSemanticCategory
