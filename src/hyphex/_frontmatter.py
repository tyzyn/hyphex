"""Shared frontmatter split, parse, and serialization utilities.

Used by both the parser and the wiki store. Has no dependency on Lark.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml
from pydantic import ValidationError

from hyphex.exceptions import FrontmatterError
from hyphex.models import Source

logger = logging.getLogger(__name__)

_FENCE = "---"


def split_frontmatter(text: str) -> tuple[str, str]:
    """Split a page into frontmatter YAML and body content.

    Args:
        text: Raw page text, optionally starting with ``---`` fenced YAML.

    Returns:
        A tuple of ``(frontmatter_yaml, body)``. If no frontmatter is
        present, ``frontmatter_yaml`` is an empty string.

    Raises:
        FrontmatterError: If a frontmatter opening fence is found but
            the closing fence is missing.

    Example:
        >>> split_frontmatter("---\\ntype: person\\n---\\nHello")
        ('type: person', 'Hello')
    """
    stripped = text.lstrip("\n")
    if not stripped.startswith(_FENCE):
        return ("", text)

    after_first_fence = stripped[len(_FENCE) :]
    end_idx = after_first_fence.find(f"\n{_FENCE}")
    if end_idx == -1:
        raise FrontmatterError("Unterminated frontmatter: missing closing '---'")

    fm_yaml = after_first_fence[:end_idx].strip()
    body_start = end_idx + len(f"\n{_FENCE}")
    body = after_first_fence[body_start:].lstrip("\n")
    return (fm_yaml, body)


def parse_frontmatter(fm_yaml: str) -> dict[str, Any]:
    """Parse a YAML string into a frontmatter dictionary.

    Args:
        fm_yaml: Raw YAML content from between ``---`` fences.

    Returns:
        Parsed dictionary. Returns an empty dict for empty input.

    Raises:
        FrontmatterError: If the YAML is malformed or not a mapping.

    Example:
        >>> parse_frontmatter("type: person")
        {'type': 'person'}
    """
    if not fm_yaml:
        return {}

    try:
        data = yaml.safe_load(fm_yaml)
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"Invalid YAML in frontmatter: {exc}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise FrontmatterError(f"Frontmatter must be a YAML mapping, got {type(data).__name__}")

    return data


def serialize_frontmatter(frontmatter: dict[str, Any], body: str) -> str:
    """Serialize frontmatter and body into a complete markdown page.

    Args:
        frontmatter: Metadata dictionary to render as YAML.
        body: Markdown body content.

    Returns:
        A string with ``---``-fenced YAML followed by the body.

    Example:
        >>> serialize_frontmatter({"type": "person"}, "Hello world.")
        '---\\ntype: person\\n---\\nHello world.'
    """
    fm_yaml = yaml.dump(
        frontmatter,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip("\n")
    return f"{_FENCE}\n{fm_yaml}\n{_FENCE}\n{body}"


def parse_sources(frontmatter: dict[str, Any]) -> list[Source]:
    """Extract Source models from a frontmatter ``sources`` list.

    Args:
        frontmatter: Parsed frontmatter dictionary.

    Returns:
        List of validated Source models.

    Raises:
        FrontmatterError: If a source entry fails validation.

    Example:
        >>> parse_sources({"sources": [{"id": 1, "title": "Doc", "url": "https://x.com"}]})
        [Source(id=1, title='Doc', url='https://x.com', ...)]
    """
    raw_sources = frontmatter.get("sources")
    if not raw_sources:
        return []

    sources: list[Source] = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            logger.warning("Skipping non-mapping source entry: %s", raw)
            continue
        try:
            sources.append(Source(**raw))
        except ValidationError as exc:
            raise FrontmatterError(f"Invalid source entry: {exc}") from exc
    return sources
