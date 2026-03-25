"""Hyphex: agent-authored, citation-grounded knowledge graphs from unstructured documents."""

from __future__ import annotations

from hyphex.exceptions import (
    FrontmatterError,
    HyphexError,
    ParseError,
    SchemaLoadError,
    StoreError,
)
from hyphex.models import Citation, Direction, EntityLink, Page, Source
from hyphex.parser import Parser
from hyphex.schema import Schema, SchemaBuilder, SchemaError
from hyphex.store import WikiStore, slugify
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
    "Schema",
    "SchemaBuilder",
    "SchemaError",
    "SchemaLoadError",
    "Source",
    "StoreError",
    "WikiStore",
    "__version__",
    "slugify",
]
