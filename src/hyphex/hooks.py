"""Hook registry and built-in lifecycle hooks for Hyphex page writes."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from hyphex.exceptions import HookError
from hyphex.models import Page

logger = logging.getLogger(__name__)

HookFn = Callable[["WriteContext"], "WriteContext"]
"""Type alias for hook functions: receives WriteContext, returns WriteContext."""


# ── Models ────────────────────────────────────────────────────────────


class SourceDocument(BaseModel):
    """Metadata about the source document being processed by the agent.

    Args:
        url: URL of the source document.
        title: Human-readable title.
        page: Optional page number or range.
        section: Optional section heading.
        hash: Optional content hash for change detection.

    Example:
        >>> SourceDocument(url="https://example.com/report.pdf", title="Q4 Report")
        SourceDocument(url='https://example.com/report.pdf', ...)
    """

    url: str
    title: str
    page: int | str | None = None
    section: str | None = None
    hash: str | None = None


class WriteContext(BaseModel):
    """Context carried through the hook pipeline during a write operation.

    Args:
        title: The page title being written.
        page: The Page being written (hooks may modify this).
        source_document: Optional source document metadata.
        agent_context: Dict for passing hints to agents (e.g. citation_hint).
        warnings: Accumulated warning strings from hooks.
        store: Optional WikiStore reference (set by WikiStore.write).
        parser: Optional Parser reference.
        schema_ref: Optional Schema reference (use ``schema`` kwarg to set).

    Example:
        >>> ctx = WriteContext(title="HMRC", page=Page(body="Content"))
        >>> ctx.add_warning("Missing citation")
        >>> len(ctx.warnings)
        1
    """

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    title: str
    page: Page
    source_document: SourceDocument | None = None
    agent_context: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    store: Any = None
    parser: Any = None
    schema_ref: Any = Field(default=None, alias="schema")

    def add_warning(self, message: str) -> None:
        """Append a warning message to the context.

        Args:
            message: The warning string to add.

        Example:
            >>> ctx.add_warning("Source 3 was uncited and removed")
        """
        self.warnings.append(message)
        logger.warning("Hook warning: %s", message)


# ── Registry ──────────────────────────────────────────────────────────


class HookRegistry:
    """Registry for pre_write and post_write hook functions.

    Hooks run in registration order. Each receives a ``WriteContext``
    and must return a (possibly modified) ``WriteContext``.

    Example:
        >>> registry = HookRegistry()
        >>> @registry.hook("pre_write")
        ... def my_hook(ctx: WriteContext) -> WriteContext:
        ...     return ctx
    """

    VALID_EVENTS: frozenset[str] = frozenset({"pre_write", "post_write"})

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookFn]] = {
            "pre_write": [],
            "post_write": [],
        }

    def register(self, event: str, fn: HookFn) -> None:
        """Register a hook function for a given event.

        Args:
            event: Event name (``"pre_write"`` or ``"post_write"``).
            fn: The hook function.

        Raises:
            HookError: If the event name is invalid.

        Example:
            >>> registry.register("pre_write", my_hook_fn)
        """
        if event not in self.VALID_EVENTS:
            raise HookError(f"Invalid event {event!r}. Must be one of {sorted(self.VALID_EVENTS)}")
        self._hooks[event].append(fn)
        logger.debug("Registered %s hook: %s", event, fn.__name__)

    def hook(self, event: str) -> Callable[[HookFn], HookFn]:
        """Decorator to register a hook function.

        Args:
            event: Event name (``"pre_write"`` or ``"post_write"``).

        Returns:
            A decorator that registers the function and returns it unchanged.

        Raises:
            HookError: If the event name is invalid.

        Example:
            >>> @registry.hook("post_write")
            ... def validate(ctx: WriteContext) -> WriteContext:
            ...     return ctx
        """

        def decorator(fn: HookFn) -> HookFn:
            self.register(event, fn)
            return fn

        return decorator

    def run(self, event: str, context: WriteContext) -> WriteContext:
        """Run all hooks for a given event sequentially.

        Args:
            event: Event name.
            context: The WriteContext to pass through the hook chain.

        Returns:
            The WriteContext after all hooks have run.

        Raises:
            HookError: If the event is invalid or a hook raises.

        Example:
            >>> ctx = registry.run("pre_write", ctx)
        """
        if event not in self.VALID_EVENTS:
            raise HookError(f"Invalid event {event!r}. Must be one of {sorted(self.VALID_EVENTS)}")
        for fn in self._hooks[event]:
            logger.debug("Running %s hook: %s", event, fn.__name__)
            try:
                context = fn(context)
            except HookError:
                raise
            except Exception as exc:
                raise HookError(
                    f"Hook {fn.__name__!r} failed during {event!r}: {exc}"
                ) from exc
        return context

    def clear(self, event: str | None = None) -> None:
        """Remove all hooks, or hooks for a specific event.

        Args:
            event: If provided, clear only that event's hooks.
                If ``None``, clear all hooks.

        Raises:
            HookError: If the event name is invalid.

        Example:
            >>> registry.clear("pre_write")
        """
        if event is None:
            for hooks_list in self._hooks.values():
                hooks_list.clear()
        else:
            if event not in self.VALID_EVENTS:
                raise HookError(
                    f"Invalid event {event!r}. Must be one of {sorted(self.VALID_EVENTS)}"
                )
            self._hooks[event].clear()


# ── Built-in hooks ────────────────────────────────────────────────────


_RE_CITATION = re.compile(r"\[@(src_\d+)(?:,[^\]]+)?\]")


def citation_inject(context: WriteContext) -> WriteContext:
    """Pre-write hook: inject source document as a citation source.

    If ``context.source_document`` is set, appends it to the page's
    frontmatter sources and sets ``agent_context["citation_hint"]``.

    Args:
        context: The current WriteContext.

    Returns:
        Modified WriteContext with source added and hint set.

    Example:
        >>> ctx = citation_inject(ctx)
        >>> ctx.agent_context["citation_hint"]
        '{{1}}'
    """
    if context.source_document is None:
        return context

    fm = dict(context.page.frontmatter)
    existing_sources: list[dict[str, Any]] = list(fm.get("sources", []))

    existing_ids = [s["id"] for s in existing_sources if isinstance(s, dict) and "id" in s]
    existing_nums = [
        int(sid.removeprefix("src_"))
        for sid in existing_ids
        if isinstance(sid, str) and sid.startswith("src_")
    ]
    next_num = max(existing_nums, default=0) + 1
    next_id = f"src_{next_num:03d}"

    new_source: dict[str, Any] = {
        "id": next_id,
        "title": context.source_document.title,
        "url": context.source_document.url,
    }
    if context.source_document.page is not None:
        new_source["page"] = context.source_document.page
    if context.source_document.section is not None:
        new_source["section"] = context.source_document.section
    if context.source_document.hash is not None:
        new_source["hash"] = context.source_document.hash

    existing_sources.append(new_source)
    fm["sources"] = existing_sources

    updated_page = context.page.model_copy(update={"frontmatter": fm})
    agent_context = dict(context.agent_context)
    agent_context["citation_hint"] = f"[@{next_id}]"

    return context.model_copy(update={"page": updated_page, "agent_context": agent_context})


def cleanup_and_resolve(context: WriteContext) -> WriteContext:
    """Post-write hook: clean up citations, resolve links, validate schema.

    Steps:

    1. Parse citations from body (via Parser or regex fallback).
    2. Remove uncited sources from frontmatter.
    3. Renumber remaining sources sequentially and update ``{{id}}``
       references in body.
    4. Create stubs for entity link targets that don't exist in the store.
    5. Validate against schema if available; add errors as warnings.

    Args:
        context: The current WriteContext.

    Returns:
        Modified WriteContext with cleaned-up page and any warnings.

    Example:
        >>> ctx = cleanup_and_resolve(ctx)
    """
    page = context.page
    fm = dict(page.frontmatter)
    body = page.body

    # Step 1: Extract citations and entity links from body.
    entity_links = page.entity_links
    cited_ids: set[str] = set()
    if context.parser is not None:
        links, citations, _evidence = context.parser.parse_inline(body)
        entity_links = links
        cited_ids = {c.source_id for c in citations}
    else:
        cited_ids = {m.group(1) for m in _RE_CITATION.finditer(body)}

    # Step 2: Remove uncited sources.
    raw_sources: list[dict[str, Any]] = fm.get("sources", [])
    fm["sources"] = [
        s for s in raw_sources if isinstance(s, dict) and s.get("id") in cited_ids
    ]
    updated_body = body

    # Step 4: Create stubs for missing link targets.
    if context.store is not None:
        for link in entity_links:
            if not context.store.exists(link.target):
                context.store.create_stub(link.target)
                logger.info("Auto-created stub for %r", link.target)

    # Step 5: Validate against schema.
    updated_page = page.model_copy(
        update={"frontmatter": fm, "body": updated_body, "entity_links": entity_links}
    )

    if context.schema_ref is not None:
        errors = context.schema_ref.validate(updated_page)
        for err in errors:
            context.add_warning(str(err))

    return context.model_copy(update={"page": updated_page})


def register_builtin_hooks(registry: HookRegistry) -> None:
    """Register the default citation and cleanup hooks.

    Args:
        registry: The HookRegistry to register into.

    Example:
        >>> registry = HookRegistry()
        >>> register_builtin_hooks(registry)
    """
    registry.register("pre_write", citation_inject)
    registry.register("post_write", cleanup_and_resolve)
