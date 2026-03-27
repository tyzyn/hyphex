"""Hyphex: agent-authored, citation-grounded knowledge graphs from unstructured documents."""

from __future__ import annotations

from hyphex.exceptions import (
    FrontmatterError,
    HookError,
    HyphexError,
    ParseError,
    SchemaLoadError,
    StoreError,
)
from hyphex.hooks import (
    HookRegistry,
    SourceDocument,
    WriteContext,
    citation_inject,
    cleanup_and_resolve,
    register_builtin_hooks,
)
from hyphex.models import Citation, EntityLink, EvidenceBlock, Page, Source
from hyphex.parser import Parser
from hyphex.schema import Schema, SchemaBuilder, SchemaError
from hyphex.store import WikiStore, slugify
from hyphex.version import __version__

__all__ = [
    "Citation",
    "EntityLink",
    "EvidenceBlock",
    "FrontmatterError",
    "HookError",
    "HookRegistry",
    "HyphexError",
    "Page",
    "ParseError",
    "Parser",
    "Schema",
    "SchemaBuilder",
    "SchemaError",
    "SchemaLoadError",
    "Source",
    "SourceDocument",
    "StoreError",
    "WikiStore",
    "WriteContext",
    "__version__",
    "citation_inject",
    "cleanup_and_resolve",
    "register_builtin_hooks",
    "slugify",
]
