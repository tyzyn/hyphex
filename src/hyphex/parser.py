"""Parser for Hyphex-flavoured markdown.

Uses Lark for inline element extraction (entity links, citations, and
evidence blocks) and PyYAML for frontmatter parsing. A Lark Transformer
converts the parse tree into Pydantic models.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, exceptions as lark_exceptions

from hyphex._frontmatter import parse_frontmatter, parse_sources, split_frontmatter
from hyphex.exceptions import ParseError
from hyphex.models import Citation, EntityLink, EvidenceBlock, Page

logger = logging.getLogger(__name__)

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


@functools.cache
def _build_lark() -> Lark:
    """Compile the Lark grammar once per process."""
    return Lark(_GRAMMAR_PATH.read_text(), parser="earley")


class _HyphexTransformer(Transformer[Token, list[EntityLink | Citation | EvidenceBlock]]):
    """Lark Transformer that converts parse-tree nodes into Pydantic models."""

    def start(self, items: list[Any]) -> list[EntityLink | Citation | EvidenceBlock]:
        """Collect all parsed elements, discarding plain text.

        Args:
            items: Mixed list of models and tokens from the parse tree.

        Returns:
            List of parsed elements.

        Example:
            >>> transformer.start([EntityLink(...), Token(...)])
            [EntityLink(...)]
        """
        return [
            item
            for item in items
            if isinstance(item, (EntityLink, Citation, EvidenceBlock))
        ]

    def entity_link(self, items: list[Any]) -> EntityLink:
        """Build an EntityLink from parse tree tokens.

        Args:
            items: ``[LINK_TEXT, RELATIONSHIP]`` tokens.

        Returns:
            An ``EntityLink`` with target and relationship.

        Example:
            >>> transformer.entity_link([Token('LINK_TEXT', 'Python'), Token('RELATIONSHIP', 'has_skill')])
            EntityLink(target='Python', relationship='has_skill')
        """
        target = str(items[0]).strip()
        relationship = str(items[1]).strip()
        return EntityLink(target=target, relationship=relationship)

    def citation(self, items: list[Any]) -> Citation:
        """Build a Citation from parse tree tokens.

        Args:
            items: ``[SRC_ID]`` or ``[SRC_ID, PAGE_REF]`` tokens.

        Returns:
            A ``Citation`` with source_id and optional page_ref.

        Example:
            >>> transformer.citation([Token('SRC_ID', 'src_001'), Token('PAGE_REF', ', p. 5')])
            Citation(source_id='src_001', page_ref='p. 5')
        """
        source_id = str(items[0]).strip()
        page_ref: str | None = None
        if len(items) > 1:
            raw = str(items[1]).strip()
            # Strip leading comma: ", p. 5" → "p. 5"
            page_ref = raw.lstrip(",").strip()
        return Citation(source_id=source_id, page_ref=page_ref)

    def evidence_block(self, items: list[Any]) -> EvidenceBlock:
        """Build an EvidenceBlock from parse tree tokens.

        Args:
            items: Tokens including the evidence content.

        Returns:
            An ``EvidenceBlock`` with the verbatim content.

        Example:
            >>> transformer.evidence_block([..., Token('EVIDENCE_CONTENT', 'text'), ...])
            EvidenceBlock(content='text')
        """
        for item in items:
            if isinstance(item, Token) and item.type == "EVIDENCE_CONTENT":
                return EvidenceBlock(content=str(item).strip())
        return EvidenceBlock(content="")

    def TEXT(self, token: Token) -> Token:  # noqa: N802
        """Pass through plain text tokens.

        Args:
            token: A TEXT token.

        Returns:
            The token unchanged.

        Example:
            >>> transformer.TEXT(Token('TEXT', 'hello'))
            Token('TEXT', 'hello')
        """
        return token


class Parser:
    """Parser for Hyphex-flavoured markdown pages.

    Splits pages into YAML frontmatter and body content, then extracts
    entity links, citations, and evidence blocks from the body using a
    Lark grammar.

    Example:
        >>> parser = Parser()
        >>> page = parser.parse("---\\ntype: person\\n---\\n[Alice](rel:knows) said hello. [@src_001]")
        >>> page.entity_links[0].target
        'Alice'
        >>> page.citations[0].source_id
        'src_001'
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
            citations, evidence blocks, and sources.

        Raises:
            ParseError: If inline element syntax is malformed.
            FrontmatterError: If the YAML frontmatter is invalid.

        Example:
            >>> parser = Parser()
            >>> page = parser.parse("[Machine Learning](rel:relates_to)")
            >>> page.entity_links[0].target
            'Machine Learning'
        """
        fm_yaml, body = split_frontmatter(text)
        frontmatter = parse_frontmatter(fm_yaml)
        sources = parse_sources(frontmatter)
        entity_links, citations, evidence_blocks = self.parse_inline(body)

        return Page(
            frontmatter=frontmatter,
            body=body,
            entity_links=entity_links,
            citations=citations,
            evidence_blocks=evidence_blocks,
            sources=sources,
        )

    def parse_inline(
        self, text: str
    ) -> tuple[list[EntityLink], list[Citation], list[EvidenceBlock]]:
        """Parse inline elements from a text fragment (no frontmatter).

        Args:
            text: Markdown body text containing entity links, citations,
                and evidence blocks.

        Returns:
            A tuple of ``(entity_links, citations, evidence_blocks)``.

        Raises:
            ParseError: If inline syntax is malformed.

        Example:
            >>> parser = Parser()
            >>> links, cites, evidence = parser.parse_inline("[Python](rel:uses) here. [@src_001]")
            >>> links[0].relationship
            'uses'
        """
        if not text or ("[" not in text and ":::" not in text):
            return ([], [], [])

        try:
            tree = self._lark.parse(text)
        except lark_exceptions.LarkError as exc:
            raise ParseError(f"Failed to parse inline elements: {exc}") from exc

        items = self._transformer.transform(tree)

        entity_links: list[EntityLink] = []
        citations: list[Citation] = []
        evidence_blocks: list[EvidenceBlock] = []
        for item in items:
            if isinstance(item, EntityLink):
                entity_links.append(item)
            elif isinstance(item, Citation):
                citations.append(item)
            elif isinstance(item, EvidenceBlock):
                evidence_blocks.append(item)
        return (entity_links, citations, evidence_blocks)
