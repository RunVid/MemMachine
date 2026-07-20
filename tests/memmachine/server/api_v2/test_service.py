import pytest

from memmachine.semantic_memory.semantic_session_manager import IsolationType
from memmachine.server.api_v2.service import (
    _infer_semantic_isolation,
    _extract_raw_ids_from_messages,
    _resolve_list_semantic_scope,
)
from memmachine.common.api.spec import ListMemoriesSpec, MemoryMessage


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


def test_resolve_list_semantic_scope_role_only():
    spec = ListMemoriesSpec(
        org_id="agent1",
        project_id="user_123",
        role_id="copilot",
    )
    user_id, role_id, session_id, isolation = _resolve_list_semantic_scope(spec)
    assert user_id is None
    assert role_id == "copilot"
    assert session_id is None
    assert isolation == [IsolationType.ROLE]


def test_session_data_role_profile_id_is_project_scoped():
    from memmachine.server.api_v2.service import _SessionData

    a = _SessionData(
        org_id="agent1",
        project_id="user_a",
        role_id="copilot",
    )
    b = _SessionData(
        org_id="agent1",
        project_id="user_b",
        role_id="copilot",
    )
    assert a.role_profile_id == "agent1/user_a/copilot"
    assert b.role_profile_id == "agent1/user_b/copilot"
    assert a.role_profile_id != b.role_profile_id


def test_resolve_list_semantic_scope_default_user_and_session():
    spec = ListMemoriesSpec(
        org_id="agent1",
        project_id="user_123",
    )
    user_id, role_id, session_id, isolation = _resolve_list_semantic_scope(spec)
    assert user_id == "user_123"
    assert role_id is None
    assert session_id is None
    assert isolation == [IsolationType.USER, IsolationType.SESSION]
