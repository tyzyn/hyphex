"""Hyphex agent integrations built on DeepAgents and LangChain.

Requires the ``agents`` optional dependency group::

    pip install hyphex[agents]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyphex.agents.backends import ReadOnlyBackend as ReadOnlyBackend
    from hyphex.agents.middleware import (
        CitationGroundingMiddleware as CitationGroundingMiddleware,
        LoggingMiddleware as LoggingMiddleware,
        RejectEmptyPageMiddleware as RejectEmptyPageMiddleware,
        RequireStubMiddleware as RequireStubMiddleware,
        SchemaValidationMiddleware as SchemaValidationMiddleware,
    )
    from hyphex.agents.note_taker import (
        create_hyphex_agent as create_hyphex_agent,
    )
    from hyphex.agents.preprocess import prepare_document as prepare_document
    from hyphex.agents.tools import make_create_stub_tool as make_create_stub_tool

__all__ = [
    "CitationGroundingMiddleware",
    "LoggingMiddleware",
    "ReadOnlyBackend",
    "RejectEmptyPageMiddleware",
    "RequireStubMiddleware",
    "SchemaValidationMiddleware",
    "create_hyphex_agent",
    "make_create_stub_tool",
    "prepare_document",
]


def __getattr__(name: str) -> object:
    """Lazy import to avoid requiring langchain at import time."""
    if name in {
        "CitationGroundingMiddleware",
        "LoggingMiddleware",
        "RejectEmptyPageMiddleware",
        "RequireStubMiddleware",
        "SchemaValidationMiddleware",
    }:
        from hyphex.agents import middleware

        return getattr(middleware, name)

    if name == "ReadOnlyBackend":
        from hyphex.agents.backends import ReadOnlyBackend

        return ReadOnlyBackend

    if name == "create_hyphex_agent":
        from hyphex.agents.note_taker import create_hyphex_agent

        return create_hyphex_agent

    if name == "prepare_document":
        from hyphex.agents.preprocess import prepare_document

        return prepare_document

    if name == "make_create_stub_tool":
        from hyphex.agents.tools import make_create_stub_tool

        return make_create_stub_tool

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
