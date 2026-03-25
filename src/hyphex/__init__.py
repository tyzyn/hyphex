"""Hyphex: agent-authored, citation-grounded knowledge graphs from unstructured documents."""

from __future__ import annotations

from hyphex.exceptions import FrontmatterError, HyphexError, ParseError
from hyphex.models import Citation, Direction, EntityLink, Page, Source
from hyphex.parser import Parser
from hyphex.version import __version__

__all__ = [
    "Citation",
    "Direction",
    "EntityLink",
    "FrontmatterError",
    "HyphexError",
    "Page",
    "ParseError",
    "Parser",
    "Source",
    "__version__",
]
