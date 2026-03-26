"""Convenience factory for the Hyphex note-taking agent."""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph

from hyphex.agents.middleware import (
    CitationGroundingMiddleware,
    LoggingMiddleware,
    RejectEmptyPageMiddleware,
    RequireStubMiddleware,
    SchemaValidationMiddleware,
)
from hyphex.agents.tools import make_create_stub_tool
from hyphex.schema import Schema, _PROMPTS_ENV

logger = logging.getLogger(__name__)


def create_hyphex_agent(
    wiki_dir: str | Path,
    schema_path: str | Path,
    source_id: str,
    source_title: str,
    *,
    model: str = "gemini-3-flash-preview",
) -> CompiledStateGraph:
    """Create a ready-to-use Hyphex note-taking agent.

    Assembles a DeepAgent with a ``FilesystemBackend`` pointed at the wiki
    directory, one custom tool (``create_stub``), and four middleware classes
    for workflow enforcement, validation, and citation grounding.

    Args:
        wiki_dir: Path to the wiki directory. Created if it does not exist.
        schema_path: Path to the ``hyphex.schema.yml`` file.
        source_id: Identifier for the source document (stored as URL in
            frontmatter sources).
        source_title: Human-readable title of the source document.
        model: Google model name passed to ``ChatGoogleGenerativeAI``.

    Returns:
        A compiled LangGraph agent ready for ``.invoke()`` or ``.stream()``.

    Example:
        >>> agent = create_hyphex_agent(
        ...     wiki_dir="./wiki",
        ...     schema_path="./wiki/hyphex.schema.yml",
        ...     source_id="https://example.com/report.pdf",
        ...     source_title="Annual Report 2025",
        ... )
        >>> result = agent.invoke({"messages": [{"role": "user", "content": doc_text}]})
    """
    wiki_dir = Path(wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    schema = Schema.load(schema_path)

    blessed_paths: set[str] = set()
    stub_tool = make_create_stub_tool(schema, blessed_paths)

    template = _PROMPTS_ENV.get_template("note_taker.j2")
    system_prompt = template.render(schema_summary=schema.to_prompt())

    middleware = [
        LoggingMiddleware(),
        RequireStubMiddleware(blessed_paths, wiki_dir=wiki_dir),
        RejectEmptyPageMiddleware(),
        SchemaValidationMiddleware(schema),
        CitationGroundingMiddleware(wiki_dir, source_id, source_title),
    ]

    llm = ChatGoogleGenerativeAI(model=model)

    return create_deep_agent(
        name="hyphex-note-taker",
        model=llm,
        tools=[stub_tool],
        system_prompt=system_prompt,
        backend=FilesystemBackend(root_dir=str(wiki_dir), virtual_mode=True),
        middleware=middleware,
    )
