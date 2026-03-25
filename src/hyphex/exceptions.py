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
