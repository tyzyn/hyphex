"""Custom exceptions for the hyphex library."""

from __future__ import annotations


class HyphexError(Exception):
    """Base exception for all hyphex errors."""


class ParseError(HyphexError):
    """Raised when parsing of Hyphex-flavoured markdown fails.

    Args:
        message: Human-readable description of the parse failure.

    Example:
        >>> raise ParseError("Unterminated entity link at line 5")
    """


class FrontmatterError(ParseError):
    """Raised when frontmatter YAML is malformed or missing required fields.

    Args:
        message: Human-readable description of the frontmatter issue.

    Example:
        >>> raise FrontmatterError("Frontmatter must be a YAML mapping")
    """


class StoreError(HyphexError):
    """Raised when a wiki store operation fails.

    Args:
        message: Human-readable description of the store failure.

    Example:
        >>> raise StoreError("Page 'Unknown Topic' does not exist")
    """


class SchemaLoadError(HyphexError):
    """Raised when a schema file cannot be loaded or saved.

    Args:
        message: Human-readable description of the schema I/O failure.

    Example:
        >>> raise SchemaLoadError("Failed to load schema from ./hyphex.schema.yml")
    """


class HookError(HyphexError):
    """Raised when a hook function fails or is misconfigured.

    Args:
        message: Human-readable description of the hook failure.

    Example:
        >>> raise HookError("Hook 'citation_inject' failed: missing source_document")
    """
