"""Custom tools for the Hyphex note-taking agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hyphex._frontmatter import serialize_frontmatter
from hyphex.exceptions import StoreError
from hyphex.schema import FieldType, Schema
from hyphex.store import slugify

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def _build_template(title: str, entity_type: str, schema: Schema) -> str:
    """Render a Hyphex-flavoured markdown template for an entity type.

    Builds frontmatter with empty field values from the schema definition,
    an empty ``sources`` list, and a top-level heading.

    Args:
        title: Human-readable page title.
        entity_type: Schema node type name.
        schema: The loaded schema.

    Returns:
        A complete markdown string with ``---``-fenced frontmatter and heading.

    Example:
        >>> _build_template("Michelle Obama", "person", schema)
        '---\\ntype: person\\nborn: \"\"\\n...'
    """
    node = schema.node_types[entity_type]
    fm: dict[str, Any] = {"type": entity_type}
    for field_name, field_def in node.fields.items():
        if field_def.type == FieldType.INTEGER:
            fm[field_name] = 0
        elif field_def.type == FieldType.ENUM and field_def.values:
            fm[field_name] = ""
        else:
            fm[field_name] = ""
    fm["sources"] = []
    body = f"\n# {title}\n\n"
    return serialize_frontmatter(fm, body)


def make_create_stub_tool(schema: Schema, blessed_paths: set[str]) -> BaseTool:  # noqa: D417
    """Build the ``create_stub`` tool with schema and path tracking bound.

    The returned tool is a closure over ``schema`` and ``blessed_paths``.
    When called, it generates a filename and template for a new wiki page
    and registers the path so ``RequireStubMiddleware`` allows the write.

    Args:
        schema: The loaded Hyphex schema.
        blessed_paths: Shared mutable set that tracks paths returned by
            ``create_stub``. ``RequireStubMiddleware`` reads this set.

    Returns:
        A LangChain ``BaseTool`` that the agent calls as ``create_stub``.

    Example:
        >>> blessed = set()
        >>> stub_tool = make_create_stub_tool(schema, blessed)
        >>> result = stub_tool.invoke({"title": "Michelle Obama", "entity_type": "person"})
        >>> result["path"]
        'michelle_obama.md'
    """

    from langchain.tools import tool as _tool

    @_tool
    def create_stub(title: str, entity_type: str) -> dict[str, str]:
        """Get the filename and formatted template for a new wiki page.

        Returns a dict with:
          - path: the snake_case filename to write to (e.g. "michelle_obama.md")
          - template: Hyphex-flavoured markdown with pre-filled frontmatter

        Use the returned path when calling write_file. Fill in the template
        with content, then write it to that path.

        Args:
            title: Human-readable page title.
            entity_type: Schema node type (e.g. "person", "organisation").
        """
        valid_types = sorted(schema.node_types.keys())
        if entity_type not in schema.node_types:
            return {
                "error": (
                    f"Unknown entity type '{entity_type}'. "
                    f"Valid types: {', '.join(valid_types)}"
                ),
            }

        try:
            filename = f"{slugify(title)}.md"
        except StoreError as exc:
            return {"error": str(exc)}

        template = _build_template(title, entity_type, schema)
        blessed_paths.add(filename)
        logger.info("Stub created for %r at %s", title, filename)
        return {"path": filename, "template": template}

    return create_stub  # type: ignore[return-value]
