from datetime import UTC, datetime, timedelta

from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Episode,
)
from memmachine.episodic_memory.declarative_memory.declarative_memory import (
    DeclarativeMemory,
)


def _episode(uid: str, content: str, minutes: int = 0) -> Episode:
    return Episode(
        uid=uid,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes),
        source="user",
        content_type=ContentType.TEXT,
        content=content,
    )


def _unify(contexts: list[tuple[float, Episode, list[Episode]]], limit: int):
    return DeclarativeMemory._unify_scored_anchored_episode_contexts(
        contexts,
        max_num_episodes=limit,
    )


def test_unify_skips_exact_duplicate_content_and_backfills():
    unique_a = _episode("a", "I like pizza", minutes=1)
    duplicate_a = _episode("a2", "I like pizza", minutes=2)
    unique_b = _episode("b", "I like pasta", minutes=3)
    unique_c = _episode("c", "I like salad", minutes=4)
    unique_d = _episode("d", "I like soup", minutes=5)
    unique_e = _episode("e", "I like bread", minutes=6)

    ranked = [
        (0.9, unique_a, [unique_a]),
        (0.8, duplicate_a, [duplicate_a]),
        (0.7, unique_b, [unique_b]),
        (0.6, unique_c, [unique_c]),
        (0.5, unique_d, [unique_d]),
        (0.4, unique_e, [unique_e]),
    ]

    result = _unify(ranked, limit=5)
    contents = [episode.content for _, episode in result]
    uids = [episode.uid for _, episode in result]

    assert contents == [
        "I like pizza",
        "I like pasta",
        "I like salad",
        "I like soup",
        "I like bread",
    ]
    assert "a2" not in uids
    assert len(result) == 5


def test_unify_returns_single_episode_when_all_content_matches():
    first = _episode("1", "same", minutes=1)
    second = _episode("2", "same", minutes=2)
    third = _episode("3", "same", minutes=3)

    result = _unify(
        [
            (0.9, first, [first]),
            (0.8, second, [second]),
            (0.7, third, [third]),
        ],
        limit=5,
    )

    assert len(result) == 1
    assert result[0][1].uid == "1"


def test_unify_keeps_distinct_content_up_to_limit():
    episodes = [_episode(str(i), f"content-{i}", minutes=i) for i in range(8)]
    ranked = [(1.0 - i * 0.1, ep, [ep]) for i, ep in enumerate(episodes)]

    result = _unify(ranked, limit=5)

    assert [episode.uid for _, episode in result] == ["0", "1", "2", "3", "4"]
