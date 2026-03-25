"""Pydantic models for Hyphex-flavoured markdown elements."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Direction(str, Enum):
    """Edge direction relative to the current page.

    Args:
        OUTGOING: Current page points to target (``>``).
        INCOMING: Target points to current page (``<``).

    Example:
        >>> Direction.OUTGOING.value
        '>'
    """

    OUTGOING = ">"
    INCOMING = "<"


class EntityLink(BaseModel):
    """A wiki-style entity link parsed from Hyphex markdown.

    Args:
        target: Title of the linked wiki page.
        direction: Edge direction, or ``None`` for bare ``[[Target]]`` links.
        relationship: Relationship type in ``snake_case``, or ``None`` before
            default resolution.
        properties: Optional key-value edge properties.
        is_default: ``True`` when the relationship was inferred from the
            schema's ``default_relationship`` rather than written explicitly.

    Example:
        >>> link = EntityLink(target="Python", direction=Direction.OUTGOING, relationship="implemented_in")
        >>> link.is_default
        False
    """

    target: str
    direction: Direction | None = None
    relationship: str | None = None
    properties: dict[str, str] = Field(default_factory=dict)
    is_default: bool = False


class Citation(BaseModel):
    """An inline citation referencing a source declared in frontmatter.

    Args:
        source_id: Integer ID matching a source in the page's frontmatter ``sources`` list.

    Example:
        >>> cite = Citation(source_id=1)
        >>> cite.source_id
        1
    """

    source_id: int


class Source(BaseModel):
    """A source document declared in page frontmatter.

    Args:
        id: Unique integer identifier used in ``{{id}}`` citations.
        title: Human-readable source document title.
        url: URL to the source document.
        page: Optional page number or range.
        section: Optional section heading within the document.
        hash: Optional content hash for change detection.

    Example:
        >>> src = Source(id=1, title="Report", url="https://example.com/report.pdf")
        >>> src.id
        1
    """

    id: int
    title: str
    url: str
    page: int | str | None = None
    section: str | None = None
    hash: str | None = None


class Page(BaseModel):
    """A parsed Hyphex wiki page.

    Args:
        frontmatter: Raw YAML frontmatter as a dictionary.
        body: The markdown body content (everything after the frontmatter).
        entity_links: Entity links extracted from the body.
        citations: Citations extracted from the body.
        sources: Source documents parsed from frontmatter.

    Example:
        >>> page = Page(frontmatter={"type": "person"}, body="See [[Alice>knows]].")
        >>> page.frontmatter["type"]
        'person'
    """

    frontmatter: dict[str, Any] = Field(default_factory=dict)
    body: str = ""
    entity_links: list[EntityLink] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    def resolve_defaults(self, default_relationship: str = "relates_to") -> Page:
        """Fill in the default relationship for bare ``[[Target]]`` links.

        Bare links (no ``>relationship``) get ``relationship`` set to
        *default_relationship* and ``is_default`` set to ``True``. Links
        that already have an explicit relationship are left unchanged.

        Args:
            default_relationship: The relationship type to assign to bare
                links. Defaults to ``"relates_to"``.

        Returns:
            A new ``Page`` with defaults resolved. The original is not mutated.

        Example:
            >>> page = Page(entity_links=[EntityLink(target="ML")])
            >>> resolved = page.resolve_defaults()
            >>> resolved.entity_links[0].relationship
            'relates_to'
            >>> resolved.entity_links[0].is_default
            True
        """
        resolved_links = [
            link.model_copy(update={"relationship": default_relationship, "is_default": True})
            if link.relationship is None
            else link
            for link in self.entity_links
        ]
        return self.model_copy(update={"entity_links": resolved_links})
