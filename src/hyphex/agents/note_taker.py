"""Convenience factory for the Hyphex note-taking agent."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph

from hyphex.agents.backends import ReadOnlyBackend
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
    workspace_dir: str | Path,
    schema_path: str | Path,
    doc_id: str,
    *,
    model: str = "gemini-3.1-flash-lite-preview",
) -> CompiledStateGraph:
    """Create a document-exploring Hyphex note-taking agent.

    The agent gets a ``CompositeBackend`` with a writable wiki directory
    (default route) and a read-only docs directory (``/docs/`` route).
    It reads preprocessed document pages from ``docs/{doc_id}/`` and
    writes wiki pages to the wiki root.

    Args:
        workspace_dir: Root directory containing ``wiki/`` and ``docs/`` subdirs.
        schema_path: Path to the ``hyphex.schema.yml`` file.
        doc_id: Identifier for the document to process. Must have been
            preprocessed with ``prepare_document()`` into
            ``workspace_dir/docs/{doc_id}/``.
        model: Google model name passed to ``ChatGoogleGenerativeAI``.

    Returns:
        A compiled LangGraph agent ready for ``.invoke()`` or ``.stream()``.

    Example:
        >>> agent = create_hyphex_agent("./workspace", "./schema.yml", "annual_report")
        >>> agent.invoke({"messages": [{"role": "user", "content": "Process docs/annual_report/"}]})
    """
    workspace_dir = Path(workspace_dir)
    wiki_dir = workspace_dir / "wiki"
    docs_dir = workspace_dir / "docs"
    wiki_dir.mkdir(parents=True, exist_ok=True)

    # Read document metadata.
    meta_path = docs_dir / doc_id / "_meta.yml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    source_url = meta.get("url", str(docs_dir / doc_id))
    source_title = meta.get("title", doc_id)

    schema = Schema.load(schema_path)

    blessed_paths: set[str] = set()
    stub_tool = make_create_stub_tool(schema, blessed_paths)

    template = _PROMPTS_ENV.get_template("note_taker.j2")
    system_prompt = template.render(
        schema_summary=schema.to_prompt(),
        doc_id=doc_id,
    )

    middleware = [
        LoggingMiddleware(),
        RequireStubMiddleware(blessed_paths, wiki_dir=wiki_dir),
        RejectEmptyPageMiddleware(),
        SchemaValidationMiddleware(schema),
        CitationGroundingMiddleware(wiki_dir, source_url, source_title),
    ]

    llm = ChatGoogleGenerativeAI(model=model)

    backend = lambda rt: CompositeBackend(  # noqa: E731
        default=FilesystemBackend(root_dir=str(wiki_dir), virtual_mode=True),
        routes={
            "/docs/": ReadOnlyBackend(root_dir=str(docs_dir), virtual_mode=True),
        },
    )

    return create_deep_agent(
        name="hyphex-note-taker",
        model=llm,
        tools=[stub_tool],
        system_prompt=system_prompt,
        backend=backend,
        middleware=middleware,
    )
