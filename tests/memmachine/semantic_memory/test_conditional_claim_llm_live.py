"""
Live LLM test: a call-screening claim must keep that the assistant
handles the call when refined into semantic memory.

Case from Agent1#1421: "I don't pick up unknown numbers on my own phone;
anything to discuss should come as a call arranged by my assistant"
was stored as an unconditional "does not answer unrecognized numbers" rule.

Requires OPENAI_API_KEY. Optional:
  OPENAI_BASE_URL (default https://api.openai.com/v1)
  OPENAI_MODEL (default gpt-4o-mini)

Run:
  pytest tests/memmachine/semantic_memory/test_conditional_claim_llm_live.py --integration -v -s
"""

from __future__ import annotations

import os
from pprint import pprint

import openai
import pytest

from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine.semantic_memory.semantic_llm import llm_feature_update
from memmachine.semantic_memory.semantic_model import (
    SemanticCategory,
    SemanticCommandType,
)
from memmachine.server.prompt.life_context_prompt import LifeContextSemanticCategory
from memmachine.server.prompt.task_assistant_prompt import TaskAssistantSemanticCategory

pytestmark = pytest.mark.integration

_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_requires_openai = pytest.mark.skipif(
    not _API_KEY,
    reason="OPENAI_API_KEY not set",
)

# Clerk-style claim: unknown calls on the user's phone, assistant handles discussion.
_CONDITIONAL_CLAIM = (
    "user does not answer calls from unrecognized numbers on their own phone; "
    "anything to discuss should come as a call arranged by the user's assistant"
)


@pytest.fixture(scope="module")
def live_llm():
    api_key = os.environ["OPENAI_API_KEY"].strip()
    base_url = (
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        or "https://api.openai.com/v1"
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(client=client, model=model),
    )


def _added_text(commands) -> str:
    return " ".join(
        f"{command.feature} {command.value}"
        for command in commands
        if command.command == SemanticCommandType.ADD
    ).casefold()


@_requires_openai
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category",
    [TaskAssistantSemanticCategory, LifeContextSemanticCategory],
    ids=["task_assistant", "life_context"],
)
async def test_conditional_call_claim_keeps_assistant(
    live_llm,
    category: SemanticCategory,
):
    """Refine must keep that the assistant arranges the call."""
    commands = await llm_feature_update(
        features=[],
        message_contents=[_CONDITIONAL_CLAIM],
        model=live_llm,
        update_prompt=category.prompt.update_prompt,
    )

    print(f"\n=== {category.name} refine of conditional claim ===")
    pprint([command.model_dump() for command in commands])

    added = _added_text(commands)
    assert added, f"{category.name} dropped the claim entirely: {commands!r}"
    assert "assistant" in added, (
        f"{category.name} dropped that the assistant handles the call: {added!r}"
    )
