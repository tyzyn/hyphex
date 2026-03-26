"""Middleware for the Hyphex note-taking agent.

Five middleware classes: one for logging, four for workflow enforcement,
validation, and citation grounding.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

from hyphex._frontmatter import (
    parse_frontmatter,
    serialize_frontmatter,
    split_frontmatter,
)
from hyphex.exceptions import FrontmatterError
from hyphex.schema import Schema

logger = logging.getLogger(__name__)

_RE_REF = re.compile(r"\{\{ref\}\}")
_RE_HEADING = re.compile(r"^#+\s+.*$", re.MULTILINE)
_RE_RELATIONSHIP = re.compile(r"\[\[[^\]]*?([><])([a-z_]+)")


def _get_file_path(request: ToolCallRequest) -> str:
    """Extract the file path from a tool call's args."""
    args = request.tool_call["args"]
    return args.get("file_path", args.get("path", ""))


def _extract_filename(file_path: str) -> str:
    """Extract the bare filename from an absolute or relative path."""
    return file_path.rsplit("/", 1)[-1]


def _is_md_tool_call(request: ToolCallRequest, tool_name: str) -> bool:
    """Check whether a tool call targets an ``.md`` path."""
    return (
        request.tool_call["name"] == tool_name
        and _get_file_path(request).endswith(".md")
    )


def _reject(request: ToolCallRequest, message: str) -> ToolMessage:
    """Build a ToolMessage rejection for a tool call."""
    return ToolMessage(
        content=f"Error: {message}",
        tool_call_id=request.tool_call["id"],
    )


class LoggingMiddleware(AgentMiddleware):
    """Log each model call and tool call with a short summary line.

    Prints ``[model]`` before each LLM call and ``[tool]`` with the tool
    name and key arguments for each tool invocation.

    Example:
        >>> mw = LoggingMiddleware()
    """

    def __init__(self) -> None:
        self._step = 0

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log before each model call.

        Args:
            state: Current agent state.
            runtime: Agent runtime context.

        Returns:
            ``None`` — no state modification.

        Example:
            >>> mw.before_model(state, runtime)
        """
        self._step += 1
        n_messages = len(state.get("messages", []))
        print(f"  [{self._step}] [model] thinking... ({n_messages} messages in context)")
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Log each tool call with name and key args.

        Args:
            request: The incoming tool call request.
            handler: The next handler in the middleware chain.

        Returns:
            The ``ToolMessage`` from the handler.

        Example:
            >>> result = mw.wrap_tool_call(request, handler)
        """
        name = request.tool_call["name"]
        args = request.tool_call.get("args", {})
        summary = self._summarize_args(name, args)
        self._step += 1
        print(f"  [{self._step}] [tool]  {name}({summary})")
        t0 = time.perf_counter()
        result = handler(request)
        elapsed = time.perf_counter() - t0
        is_error = hasattr(result, "content") and str(result.content).startswith("Error:")
        status = "ERR" if is_error else "ok"
        print(f"           └─ {status} ({elapsed:.1f}s)")
        return result

    @staticmethod
    def _summarize_args(tool_name: str, args: dict[str, Any]) -> str:
        """Build a short summary of tool call arguments.

        Args:
            tool_name: Name of the tool being called.
            args: The tool call arguments dict.

        Returns:
            A short string summarizing the key arguments.

        Example:
            >>> LoggingMiddleware._summarize_args("write_file", {"path": "foo.md", "content": "..."})
            'path=foo.md'
        """
        if tool_name in ("write_file", "read_file", "edit_file"):
            return f"path={args.get('file_path', args.get('path', '?'))}"
        if tool_name == "create_stub":
            return f"{args.get('title', '?')}, type={args.get('entity_type', '?')}"
        if tool_name == "grep":
            return f"query={args.get('query', args.get('pattern', '?'))}"
        if tool_name == "glob":
            return f"pattern={args.get('pattern', '?')}"
        if tool_name == "write_todos":
            todos = args.get("todos", [])
            return f"{len(todos)} todos"
        # Fallback: show first key
        if args:
            first_key = next(iter(args))
            val = str(args[first_key])[:40]
            return f"{first_key}={val}"
        return ""


class RequireStubMiddleware(AgentMiddleware):
    """Reject ``write_file`` on ``.md`` paths not blessed by ``create_stub``.

    Tracks which file paths have been returned by the ``create_stub`` tool.
    Any ``write_file`` call to an ``.md`` file whose path is not in the
    blessed set is rejected — unless the file already exists on disk
    (enrichment of an existing page).

    ``edit_file`` is always allowed (the page already exists).

    Args:
        blessed_paths: Shared set of paths returned by ``create_stub``.
        wiki_dir: Path to the wiki directory, used to check if a page
            already exists on disk. If ``None``, only blessed paths are allowed.

    Example:
        >>> mw = RequireStubMiddleware(blessed_paths={"michelle_obama.md"})
    """

    def __init__(
        self, blessed_paths: set[str], wiki_dir: Path | None = None
    ) -> None:
        self._blessed = blessed_paths
        self._wiki_dir = wiki_dir

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Intercept ``write_file`` and reject unblessed ``.md`` paths.

        Args:
            request: The incoming tool call request.
            handler: The next handler in the middleware chain.

        Returns:
            A ``ToolMessage`` — either the rejection or the handler result.

        Example:
            >>> result = mw.wrap_tool_call(request, handler)
        """
        if _is_md_tool_call(request, "write_file"):
            file_path = _get_file_path(request)
            filename = _extract_filename(file_path)
            # Allow writes to blessed paths (new pages) or existing files (enrichment).
            is_blessed = filename in self._blessed
            exists_on_disk = (
                self._wiki_dir is not None
                and (self._wiki_dir / filename).is_file()
            )
            if not (is_blessed or exists_on_disk):
                return _reject(
                    request,
                    f"File '{file_path}' has not been templated. "
                    "Call create_stub(title, entity_type) first to get "
                    "the correct filename and template.",
                )
        return handler(request)


class RejectEmptyPageMiddleware(AgentMiddleware):
    """Reject ``write_file`` when content has citations but no substance.

    A page with ``{{ref}}`` markers indicates the agent intended a full page.
    If the body text (excluding frontmatter and headings) is fewer than ~50
    characters, the write is rejected.

    Pages without ``{{ref}}`` markers pass through freely — these are
    intentional stubs.

    Example:
        >>> mw = RejectEmptyPageMiddleware()
    """

    MINIMUM_BODY_CHARS: int = 50

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Intercept ``write_file`` and reject near-empty cited pages.

        Args:
            request: The incoming tool call request.
            handler: The next handler in the middleware chain.

        Returns:
            A ``ToolMessage`` — either the rejection or the handler result.

        Example:
            >>> result = mw.wrap_tool_call(request, handler)
        """
        if _is_md_tool_call(request, "write_file"):
            content = request.tool_call["args"].get("content", "")
            if _RE_REF.search(content):
                body = self._extract_body_text(content)
                if len(body.strip()) < self.MINIMUM_BODY_CHARS:
                    return _reject(
                        request,
                        "Page has citations but almost no content. "
                        "Add substantive text about the topic before writing.",
                    )
        return handler(request)

    @staticmethod
    def _extract_body_text(content: str) -> str:
        """Extract body text excluding frontmatter and headings.

        Args:
            content: Full page content including frontmatter.

        Returns:
            Body text with headings stripped.

        Example:
            >>> RejectEmptyPageMiddleware._extract_body_text("---\\ntype: x\\n---\\n# Title\\nBody")
            'Body'
        """
        try:
            _, body = split_frontmatter(content)
        except FrontmatterError:
            body = content
        return _RE_HEADING.sub("", body)


class SchemaValidationMiddleware(AgentMiddleware):
    """Validate frontmatter type and entity link relationships on writes.

    Intercepts ``write_file`` and ``edit_file`` on ``.md`` files. Checks:

    1. The frontmatter ``type`` field matches a known entity type.
    2. Entity link relationships (``[[Target>rel]]``) exist in the schema.

    Returns a ``ToolMessage`` error with valid options on failure.

    Args:
        schema: The loaded Hyphex schema.

    Example:
        >>> mw = SchemaValidationMiddleware(schema)
    """

    def __init__(self, schema: Schema) -> None:
        self._schema = schema
        self._valid_types = sorted(schema.node_types.keys())
        self._valid_rels = frozenset(schema.relationship_types.keys())

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Intercept writes and validate against the schema.

        Args:
            request: The incoming tool call request.
            handler: The next handler in the middleware chain.

        Returns:
            A ``ToolMessage`` — either the rejection or the handler result.

        Example:
            >>> result = mw.wrap_tool_call(request, handler)
        """
        if _is_md_tool_call(request, "write_file"):
            content = request.tool_call["args"].get("content", "")
            errors = self._validate_full_page(content)
            if errors:
                return _reject(request, " | ".join(errors))
        elif _is_md_tool_call(request, "edit_file"):
            new_string = request.tool_call["args"].get("new_string", "")
            errors = self._validate_relationships(new_string)
            if errors:
                return _reject(request, " | ".join(errors))
        return handler(request)

    def _validate_full_page(self, content: str) -> list[str]:
        """Validate frontmatter type and relationships in a full page.

        Args:
            content: Full page content string.

        Returns:
            List of error messages. Empty means valid.

        Example:
            >>> mw._validate_full_page("---\\ntype: unknown\\n---\\n")
            ["Unknown entity type 'unknown'. Valid types: ..."]
        """
        errors: list[str] = []

        try:
            fm_yaml, body = split_frontmatter(content)
            fm = parse_frontmatter(fm_yaml)
        except FrontmatterError as exc:
            return [f"Malformed frontmatter: {exc}"]

        page_type = fm.get("type")
        if page_type is not None and page_type not in self._schema.node_types:
            errors.append(
                f"Unknown entity type '{page_type}'. "
                f"Valid types: {', '.join(self._valid_types)}"
            )

        errors.extend(self._validate_relationships(body))
        return errors

    def _validate_relationships(self, text: str) -> list[str]:
        """Check that entity link relationships exist in the schema.

        Args:
            text: Text to scan for ``[[Target>relationship]]`` patterns.

        Returns:
            List of error messages for unknown relationships.

        Example:
            >>> mw._validate_relationships("See [[Bob>invented_by]].")
            ["Unknown relationship 'invented_by'. ..."]
        """
        errors: list[str] = []
        for match in _RE_RELATIONSHIP.finditer(text):
            rel_name = match.group(2)
            if rel_name not in self._valid_rels:
                errors.append(
                    f"Unknown relationship '{rel_name}'. "
                    f"Valid relationships: {', '.join(sorted(self._valid_rels))}"
                )
        return errors


class CitationGroundingMiddleware(AgentMiddleware):
    """Post-write fixup that resolves ``{{ref}}`` markers and manages sources.

    After any ``write_file`` or ``edit_file`` on an ``.md`` file, reads the
    page back from disk. If ``{{ref}}`` markers are present, replaces them
    with the source's citation ID and ensures the source is listed in
    frontmatter. Writes the corrected file back.

    Works identically for ``write_file`` and ``edit_file`` — the agent can
    use whichever tool is natural.

    Args:
        wiki_dir: Path to the wiki directory on disk.
        source_id: Identifier for the source document (stored as ``url``).
        source_title: Human-readable title of the source document.

    Example:
        >>> mw = CitationGroundingMiddleware(Path("./wiki"), "doc_123", "Annual Report 2025")
    """

    def __init__(self, wiki_dir: Path, source_id: str, source_title: str) -> None:
        self._wiki_dir = wiki_dir
        self._source_id = source_id
        self._source_title = source_title

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Let the write/edit proceed, then fix up citations on disk.

        Args:
            request: The incoming tool call request.
            handler: The next handler in the middleware chain.

        Returns:
            The ``ToolMessage`` from the handler.

        Example:
            >>> result = mw.wrap_tool_call(request, handler)
        """
        # Fast exit: check request content for {{ref}} before any disk I/O.
        tool_name = request.tool_call["name"]
        file_path = _get_file_path(request)
        args = request.tool_call["args"]

        has_ref = False
        if tool_name == "write_file" and file_path.endswith(".md"):
            has_ref = bool(_RE_REF.search(args.get("content", "")))
        elif tool_name == "edit_file" and file_path.endswith(".md"):
            has_ref = bool(_RE_REF.search(args.get("new_string", "")))

        result = handler(request)

        if not has_ref:
            return result

        filename = _extract_filename(file_path)
        disk_path = self._wiki_dir / filename

        try:
            content = disk_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return result

        if not _RE_REF.search(content):
            return result

        fixed = self._resolve_citations(content)
        disk_path.write_text(fixed, encoding="utf-8")
        return result

    def _resolve_citations(self, content: str) -> str:
        """Replace ``{{ref}}`` markers and ensure source is in frontmatter.

        Args:
            content: Full page content read from disk.

        Returns:
            Content with ``{{ref}}`` resolved and source entry present.

        Example:
            >>> mw._resolve_citations("---\\ntype: person\\nsources: []\\n---\\nBorn {{ref}}.")
            '---\\ntype: person\\nsources:\\n- id: 1\\n  ...'
        """
        try:
            fm_yaml, body = split_frontmatter(content)
            fm = parse_frontmatter(fm_yaml)
        except FrontmatterError:
            logger.warning("Could not parse frontmatter for citation grounding")
            return content

        raw_sources = fm.get("sources", [])
        # Keep only valid, middleware-managed sources (must have both id and url).
        # Strips agent-added entries like {"title": "Source Document"}.
        existing_sources: list[dict[str, Any]] = [
            s for s in raw_sources if isinstance(s, dict) and "id" in s and "url" in s
        ]

        # Reuse existing ID for this source, or assign the next one.
        source_id: int | None = None
        for s in existing_sources:
            if s.get("url") == self._source_id:
                source_id = s["id"]
                break

        if source_id is None:
            base_id = max((s["id"] for s in existing_sources), default=0)
            source_id = base_id + 1
            existing_sources.append({
                "id": source_id,
                "title": self._source_title,
                "url": self._source_id,
            })

        resolved_body = _RE_REF.sub(f"{{{{{source_id}}}}}", body)
        fm["sources"] = existing_sources
        return serialize_frontmatter(fm, resolved_body)
