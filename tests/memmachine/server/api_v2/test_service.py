import pytest

from memmachine.semantic_memory.semantic_session_manager import IsolationType
from memmachine.server.api_v2.service import (
    _infer_semantic_isolation,
    _extract_raw_ids_from_messages,
)
from memmachine.common.api.spec import MemoryMessage


def test_infer_semantic_isolation_role_only():
    assert _infer_semantic_isolation(role_id="agent-42", session_id="chat-1") == [
        IsolationType.ROLE
    ]


def test_infer_semantic_isolation_user_only():
    assert _infer_semantic_isolation(role_id=None, session_id=None) == [
        IsolationType.USER
    ]


def test_infer_semantic_isolation_user_and_session():
    assert _infer_semantic_isolation(role_id=None, session_id="chat-1") == [
        IsolationType.USER,
        IsolationType.SESSION,
    ]


def test_extract_raw_ids_from_messages():
    message = MemoryMessage(
        content="hello",
        metadata={
            "user_id": "alice",
            "role_id": "agent-42",
            "session_id": "chat-1",
        },
    )
    assert _extract_raw_ids_from_messages([message]) == ("alice", "agent-42", "chat-1")

    empty_metadata = MemoryMessage(content="hello", metadata={})
    assert _extract_raw_ids_from_messages([empty_metadata]) == (None, None, None)
