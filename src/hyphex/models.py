"""Pydantic models for Hyphex-flavoured markdown elements."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EntityLink(BaseModel):
    """An entity link parsed from Hyphex markdown.

    Uses markdown link syntax with a ``rel:`` prefix:
    ``[Target](rel:relationship)``.

    Args:
        target: Title of the linked wiki page (the display text).
        relationship: Relationship type in ``snake_case``. Always required.

    Example:
        >>> link = EntityLink(target="Python", relationship="has_skill")
        >>> link.relationship
        'has_skill'
    """

    target: str
    relationship: str


class Citation(BaseModel):
    """A pandoc-style inline citation referencing a source.

    Format: ``[@src_001]`` or ``[@src_001, p. 5]`` or ``[@src_001, pp. 12-15]``.

    Args:
        source_id: String identifier matching a source in frontmatter (e.g. ``"src_001"``).
        page_ref: Optional page reference string (e.g. ``"p. 5"``, ``"pp. 12-15"``).

    Example:
        >>> cite = Citation(source_id="src_001", page_ref="p. 5")
        >>> cite.source_id
        'src_001'
    """

    source_id: str
    page_ref: str | None = None


class EvidenceBlock(BaseModel):
    """A verbatim evidence block from a source document.

    Fenced with ``::: evidence`` and ``:::``. Contains text copied
    directly from the source for grounding verification.

    Args:
        content: The verbatim text from the source.

    Example:
        >>> ev = EvidenceBlock(content="Gabriel has 5+ years of experience.")
        >>> ev.content
        'Gabriel has 5+ years of experience.'
    """

    content: str


class Source(BaseModel):
    """A source document declared in page frontmatter.

    Args:
        id: String identifier used in ``[@id]`` citations (e.g. ``"src_001"``).
        title: Human-readable source document title.
        url: URL or path to the source document.
        section: Optional section heading within the document.
        hash: Optional content hash for change detection.

    Example:
        >>> src = Source(id="src_001", title="Report", url="https://example.com/report.pdf")
        >>> src.id
        'src_001'
    """

    id: str
    title: str
    url: str
    section: str | None = None
    hash: str | None = None


class Page(BaseModel):
    """A parsed Hyphex wiki page.

    Args:
        frontmatter: Raw YAML frontmatter as a dictionary.
        body: The markdown body content (everything after the frontmatter).
        entity_links: Entity links extracted from the body.
        citations: Citations extracted from the body.
        evidence_blocks: Verbatim evidence blocks extracted from the body.
        sources: Source documents parsed from frontmatter.

    Example:
        >>> page = Page(frontmatter={"type": "person"}, body="See [Alice](rel:knows).")
        >>> page.frontmatter["type"]
        'person'
    """

    frontmatter: dict[str, Any] = Field(default_factory=dict)
    body: str = ""
    entity_links: list[EntityLink] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_blocks: list[EvidenceBlock] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
