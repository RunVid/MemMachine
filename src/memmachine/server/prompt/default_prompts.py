"""Default prompt templates for included sample domains."""

from memmachine.semantic_memory.semantic_model import SemanticCategory
from memmachine.server.prompt.agent_personality_prompt import (
    AgentPersonalitySemanticCategory,
)
from memmachine.server.prompt.coding_style_prompt import CodingStyleSemanticCategory
from memmachine.server.prompt.crm_prompt import CrmSemanticCategory
from memmachine.server.prompt.financial_analyst_prompt import (
    FinancialAnalystSemanticCategory,
)
from memmachine.server.prompt.health_assistant_prompt import (
    HealthAssistantSemanticCategory,
)
from memmachine.server.prompt.profile_prompt import UserProfileSemanticCategory
from memmachine.server.prompt.task_assistant_prompt import (
    TaskAssistantSemanticCategory,
)
from memmachine.server.prompt.life_context_prompt import (
    LifeContextSemanticCategory,
)
from memmachine.server.prompt.writing_assistant_prompt import (
    WritingAssistantSemanticCategory,
)

PREDEFINED_SEMANTIC_CATEGORIES: dict[str, SemanticCategory] = {
    "profile_prompt": UserProfileSemanticCategory,
    "task_assistant_prompt": TaskAssistantSemanticCategory,
    "life_context_prompt": LifeContextSemanticCategory,
    "agent_personality_prompt": AgentPersonalitySemanticCategory,
    "coding_prompt": CodingStyleSemanticCategory,
    "writing_assistant_prompt": WritingAssistantSemanticCategory,
    "financial_analyst_prompt": FinancialAnalystSemanticCategory,
    "health_assistant_prompt": HealthAssistantSemanticCategory,
    "crm_prompt": CrmSemanticCategory,
}
