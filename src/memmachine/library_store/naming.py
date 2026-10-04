"""Sample a markdown body and turn it into a free title before anything is stored."""

import asyncio
from collections.abc import Awaitable, Callable
from time import monotonic

from memmachine.common.api.spec import LIBRARY_NAME_MAX_LENGTH
from memmachine.common.errors import LibraryTimeoutError

_FALLBACK_NAME = "未命名"
_SAMPLE_CHARS = 1200
_HEADING_LIMIT = 20
# One server process. These are concurrent waits on that process, not extra workers.
LIBRARY_TITLE_MAX_CONCURRENT = 20
_title_slots = asyncio.Semaphore(LIBRARY_TITLE_MAX_CONCURRENT)


def sample_for_title(content: str) -> str:
    """Keep the start, middle, end, and markdown headings for title generation."""
    headings = _headings(content)
    parts = [f"Headings:\n{headings}" if headings else "Headings:\n(none)"]
    parts.append(f"Excerpt:\n{_excerpt(content)}")
    return "\n\n".join(parts)


def choose_available_name(raw: str, existing: set[str]) -> str:
    """Pick one title, then add a numeric suffix when that title is taken."""
    name = _first_line(raw)
    if name not in existing:
        return name

    number = 2
    while True:
        suffix = f" {number}"
        room = LIBRARY_NAME_MAX_LENGTH - len(suffix)
        base = name[:room].rstrip() or _FALLBACK_NAME
        candidate = f"{base}{suffix}"
        if candidate not in existing:
            return candidate
        number += 1


async def resolve_title(
    *,
    name: str | None,
    content: str,
    existing: set[str],
    seconds: float,
    suggest: Callable[[str, set[str]], Awaitable[str]],
) -> tuple[str, bool]:
    """
    Return a title and whether it was generated.

    Nothing is written here. A timeout is raised before a generated title is
    accepted, so the caller can skip the insert.
    """
    deadline = monotonic() + seconds
    if name is not None:
        _ensure_time(deadline)
        return name, False

    remaining = deadline - monotonic()
    if remaining <= 0:
        raise LibraryTimeoutError
    try:
        suggested = await asyncio.wait_for(
            _suggest_within_limit(suggest, content, existing),
            timeout=remaining,
        )
    except TimeoutError as error:
        raise LibraryTimeoutError from error
    _ensure_time(deadline)
    return choose_available_name(suggested, existing), True


async def _suggest_within_limit(
    suggest: Callable[[str, set[str]], Awaitable[str]],
    content: str,
    existing: set[str],
) -> str:
    """Run one title suggestion. Extra requests wait for a free slot."""
    async with _title_slots:
        return await suggest(content, existing)


def _ensure_time(deadline: float) -> None:
    if monotonic() >= deadline:
        raise LibraryTimeoutError


def _excerpt(content: str) -> str:
    length = len(content)
    if length <= _SAMPLE_CHARS * 3:
        return content
    middle = length // 2
    half = _SAMPLE_CHARS // 2
    return "\n...\n".join(
        (
            content[:_SAMPLE_CHARS],
            content[middle - half : middle + half],
            content[-_SAMPLE_CHARS:],
        )
    )


def _headings(content: str) -> str:
    found = [
        line.strip() for line in content.splitlines() if line.lstrip().startswith("#")
    ]
    return "\n".join(found[:_HEADING_LIMIT])


def _first_line(raw: str) -> str:
    text = raw.strip()
    line = text.splitlines()[0].strip() if text else ""
    line = line.strip("`\"'“”").strip()
    if line.startswith("#"):
        line = line.lstrip("#").strip()
    if not line:
        line = _FALLBACK_NAME
    if len(line) > LIBRARY_NAME_MAX_LENGTH:
        line = line[:LIBRARY_NAME_MAX_LENGTH].rstrip()
    return line or _FALLBACK_NAME
