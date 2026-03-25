"""Parser for Hyphex-flavoured markdown.

Uses Lark for inline element extraction (entity links and citations)
and PyYAML for frontmatter parsing. A Lark Transformer converts the
parse tree into Pydantic models.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, exceptions as lark_exceptions

from hyphex._frontmatter import parse_frontmatter, parse_sources, split_frontmatter
from hyphex.exceptions import ParseError
from hyphex.models import Citation, Direction, EntityLink, Page

logger = logging.getLogger(__name__)

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


@functools.cache
def _build_lark() -> Lark:
    """Compile the Lark grammar once per process."""
    return Lark(_GRAMMAR_PATH.read_text(), parser="earley")


class _HyphexTransformer(Transformer[Token, list[EntityLink | Citation]]):
    """Lark Transformer that converts parse-tree nodes into Pydantic models."""

    def start(self, items: list[Any]) -> list[EntityLink | Citation]:
        return [item for item in items if isinstance(item, (EntityLink, Citation))]

    def entity_link(self, items: list[Any]) -> EntityLink:
        target: str = items[0]
        direction: Direction | None = None
        relationship: str | None = None
        properties: dict[str, str] = {}

        for item in items[1:]:
            if isinstance(item, Direction):
                direction = item
            elif isinstance(item, str):
                relationship = item
            elif isinstance(item, dict):
                properties = item

        return EntityLink(
            target=target,
            direction=direction,
            relationship=relationship,
            properties=properties,
        )

    def target(self, items: list[Token]) -> str:
        return str(items[0]).strip()

    def DIRECTION(self, token: Token) -> Direction:  # noqa: N802
        return Direction(str(token))

    def relationship(self, items: list[Token]) -> str:
        return str(items[0])

    def properties(self, items: list[tuple[str, str]]) -> dict[str, str]:
        return dict(items)

    def prop(self, items: list[str]) -> tuple[str, str]:
        return (items[0], items[1])

    def key(self, items: list[Token]) -> str:
        return str(items[0])

    def value(self, items: list[Token]) -> str:
        return str(items[0]).strip()

    def citation(self, items: list[int]) -> Citation:
        return Citation(source_id=items[0])

    def source_id(self, items: list[Token]) -> int:
        return int(items[0])

    def TEXT(self, token: Token) -> Token:  # noqa: N802
        return token


class Parser:
    """Parser for Hyphex-flavoured markdown pages.

    Splits pages into YAML frontmatter and body content, then extracts
    entity links and citations from the body using a Lark grammar.

    Example:
        >>> parser = Parser()
        >>> page = parser.parse("---\\ntype: person\\n---\\n[[Alice>knows]] said hello. {{1}}")
        >>> page.entity_links[0].target
        'Alice'
        >>> page.citations[0].source_id
        1
    """

    def __init__(self) -> None:
        self._lark = _build_lark()
        self._transformer = _HyphexTransformer()

    def parse(self, text: str) -> Page:
        """Parse a full Hyphex-flavoured markdown page.

        Args:
            text: Raw page content, optionally with ``---``-fenced YAML
                frontmatter followed by markdown body.

        Returns:
            A ``Page`` with extracted frontmatter, body, entity links,
            citations, and sources.

        Raises:
            ParseError: If inline element syntax is malformed.
            FrontmatterError: If the YAML frontmatter is invalid.

        Example:
            >>> parser = Parser()
            >>> page = parser.parse("[[Machine Learning]]")
            >>> page.entity_links[0].target
            'Machine Learning'
        """
        fm_yaml, body = split_frontmatter(text)
        frontmatter = parse_frontmatter(fm_yaml)
        sources = parse_sources(frontmatter)
        entity_links, citations = self.parse_inline(body)

        return Page(
            frontmatter=frontmatter,
            body=body,
            entity_links=entity_links,
            citations=citations,
            sources=sources,
        )

    def parse_inline(self, text: str) -> tuple[list[EntityLink], list[Citation]]:
        """Parse inline elements from a text fragment (no frontmatter).

        Args:
            text: Markdown body text containing ``[[...]]`` and ``{{...}}``.

        Returns:
            A tuple of ``(entity_links, citations)``.

        Raises:
            ParseError: If inline syntax is malformed.

        Example:
            >>> parser = Parser()
            >>> links, cites = parser.parse_inline("See [[Python>uses]] here. {{1}}")
            >>> links[0].relationship
            'uses'
        """
        if not text or ("[[" not in text and "{{" not in text):
            return ([], [])

        try:
            tree = self._lark.parse(text)
        except lark_exceptions.LarkError as exc:
            raise ParseError(f"Failed to parse inline elements: {exc}") from exc

        items = self._transformer.transform(tree)

        entity_links: list[EntityLink] = []
        citations: list[Citation] = []
        for item in items:
            if isinstance(item, EntityLink):
                entity_links.append(item)
            elif isinstance(item, Citation):
                citations.append(item)
        return (entity_links, citations)
