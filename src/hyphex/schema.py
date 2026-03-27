"""Schema definition, loading, validation, and prompt generation for Hyphex."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from jinja2 import Environment, PackageLoader
from pydantic import BaseModel, Field, ValidationError, model_validator

from hyphex.exceptions import SchemaLoadError
from hyphex.models import Page

_PROMPTS_ENV = Environment(
    loader=PackageLoader("hyphex", "prompts"),
    keep_trailing_newline=True,
    auto_reload=False,
)


# ── Enums ─────────────────────────────────────────────────────────────


class FieldType(str, Enum):
    """Allowed data types for node-type fields.

    Example:
        >>> FieldType.ENUM.value
        'enum'
    """

    STRING = "string"
    INTEGER = "integer"
    ENUM = "enum"
    LINK = "link"


class Enforcement(str, Enum):
    """Schema enforcement strictness levels.

    Example:
        >>> Enforcement.STRICT.value
        'strict'
    """

    STRICT = "strict"
    WARN = "warn"
    PERMISSIVE = "permissive"


# ── Data Models ───────────────────────────────────────────────────────


class SchemaError(BaseModel):
    """A single schema validation finding.

    This is a data object, not an exception. Validation returns a list of
    these — it never raises.

    Args:
        message: Human-readable description of the violation.
        severity: ``"error"`` under strict enforcement, ``"warning"`` under warn.

    Example:
        >>> err = SchemaError(message="Unknown relationship 'works_with'", severity="warning")
        >>> str(err)
        "warning: Unknown relationship 'works_with'"
    """

    message: str
    severity: Literal["error", "warning"]

    def __str__(self) -> str:
        """Return a formatted string representation.

        Returns:
            ``"{severity}: {message}"`` format.

        Example:
            >>> str(SchemaError(message="Bad field", severity="error"))
            'error: Bad field'
        """
        return f"{self.severity}: {self.message}"


class FieldDef(BaseModel):
    """Definition of a single field on a node type.

    Args:
        type: The field's data type.
        required: Whether the field must be present in frontmatter.
        values: Allowed values (enum type only).
        relationship: The relationship type this field maps to (link type only).

    Example:
        >>> FieldDef(type=FieldType.ENUM, values=["won", "lost"], required=True)
        FieldDef(type=<FieldType.ENUM: 'enum'>, ...)
    """

    type: FieldType = FieldType.STRING
    required: bool = False
    values: list[str] | None = None
    relationship: str | None = None

    @model_validator(mode="after")
    def _enum_requires_values(self) -> FieldDef:
        if self.type == FieldType.ENUM and not self.values:
            msg = "Enum field must specify a non-empty 'values' list"
            raise ValueError(msg)
        return self


class NodeType(BaseModel):
    """Definition of an entity/node type in the schema.

    Args:
        description: Human-readable description.
        fields: Mapping of field name to field definition.

    Example:
        >>> NodeType(description="A delivery", fields={"status": FieldDef(type=FieldType.ENUM, values=["won"])})
        NodeType(description='A delivery', ...)
    """

    description: str = ""
    fields: dict[str, FieldDef] = Field(default_factory=dict)


class RelationshipType(BaseModel):
    """Definition of a relationship type in the schema.

    Args:
        description: Human-readable description.
        from_types: Node types allowed as source, or ``None`` for unconstrained.
        to_types: Node types allowed as target, or ``None`` for unconstrained.

    Example:
        >>> RelationshipType(description="Client", from_types=["project"], to_types=["organisation"])
        RelationshipType(description='Client', ...)
    """

    description: str = ""
    from_types: list[str] | None = None
    to_types: list[str] | None = None


class SchemaSettings(BaseModel):
    """Global schema settings.

    Args:
        enforcement: Validation strictness level.
        auto_create_stubs: Whether to auto-create stub pages for link targets.
        default_relationship: Relationship assigned to bare ``[[Target]]`` links.
        warn_on_default_relationship: Whether to warn when the default is used.

    Example:
        >>> SchemaSettings().enforcement
        <Enforcement.WARN: 'warn'>
    """

    enforcement: Enforcement = Enforcement.WARN
    auto_create_stubs: bool = True
    default_relationship: str = "relates_to"
    warn_on_default_relationship: bool = True


# ── Validation helpers ────────────────────────────────────────────────


_Severity = Literal["error", "warning"]


def _resolve_node(page: Page, schema: Schema) -> tuple[str | None, NodeType | None]:
    """Look up the page's node type from the schema."""
    page_type = page.frontmatter.get("type")
    if page_type is None or page_type not in schema.node_types:
        return (page_type, None)
    return (page_type, schema.node_types[page_type])


def _check_page_type(
    page: Page, schema: Schema, severity: _Severity
) -> list[SchemaError]:
    page_type, node = _resolve_node(page, schema)
    if page_type is None:
        return []
    if node is None:
        return [SchemaError(message=f"Unknown node type '{page_type}'", severity=severity)]
    return []


def _check_required_fields(
    page: Page, schema: Schema, severity: _Severity
) -> list[SchemaError]:
    page_type, node = _resolve_node(page, schema)
    if node is None:
        return []
    errors: list[SchemaError] = []
    for field_name, field_def in node.fields.items():
        if field_def.required and field_name not in page.frontmatter:
            errors.append(
                SchemaError(
                    message=f"Required field '{field_name}' missing for type '{page_type}'",
                    severity=severity,
                )
            )
    return errors


def _check_field_values(
    page: Page, schema: Schema, severity: _Severity
) -> list[SchemaError]:
    _, node = _resolve_node(page, schema)
    if node is None:
        return []
    errors: list[SchemaError] = []
    for field_name, field_def in node.fields.items():
        if field_name not in page.frontmatter:
            continue
        value = page.frontmatter[field_name]
        if field_def.type == FieldType.ENUM and field_def.values:
            if value not in field_def.values:
                errors.append(
                    SchemaError(
                        message=(
                            f"Invalid value '{value}' for enum field '{field_name}'"
                            f" (allowed: {field_def.values})"
                        ),
                        severity=severity,
                    )
                )
        elif field_def.type == FieldType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(
                    SchemaError(
                        message=f"Field '{field_name}' must be an integer, got {type(value).__name__}",
                        severity=severity,
                    )
                )
    return errors


def _check_relationships(
    page: Page, schema: Schema, severity: _Severity
) -> list[SchemaError]:
    errors: list[SchemaError] = []
    page_type = page.frontmatter.get("type")
    for link in page.entity_links:
        rel_name = link.relationship
        if rel_name not in schema.relationship_types:
            errors.append(
                SchemaError(
                    message=f"Unknown relationship '{rel_name}'",
                    severity=severity,
                )
            )
            continue
        rel = schema.relationship_types[rel_name]
        if rel.from_types is not None and page_type is not None:
            if page_type not in rel.from_types:
                errors.append(
                    SchemaError(
                        message=(
                            f"Relationship '{rel_name}' not allowed from type '{page_type}'"
                            f" (allowed: {rel.from_types})"
                        ),
                        severity=severity,
                    )
                )
    return errors


# ── Schema ────────────────────────────────────────────────────────────


class Schema(BaseModel):
    """A complete Hyphex schema defining node types, relationships, and settings.

    Args:
        node_types: Mapping of type name to node type definition.
        relationship_types: Mapping of relationship name to definition.
        settings: Global schema settings.

    Example:
        >>> schema = Schema.load("./wiki/hyphex.schema.yml")
        >>> errors = schema.validate(page)
    """

    context: str = Field(
        default="",
        description="Free-text explanation of why this knowledge base exists, "
        "what the data will be used for, and what to prioritise when extracting. "
        "Injected into agent prompts to guide extraction decisions.",
    )
    node_types: dict[str, NodeType] = Field(default_factory=dict)
    relationship_types: dict[str, RelationshipType] = Field(default_factory=dict)
    settings: SchemaSettings = Field(default_factory=SchemaSettings)

    @classmethod
    def load(cls, path: str | Path) -> Schema:
        """Load a schema from a YAML file.

        Args:
            path: Path to the ``hyphex.schema.yml`` file.

        Returns:
            A validated Schema instance.

        Raises:
            SchemaLoadError: If the file cannot be read or contains
                invalid YAML or schema structure.

        Example:
            >>> schema = Schema.load("./wiki/hyphex.schema.yml")
        """
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SchemaLoadError(f"Failed to load schema from {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise SchemaLoadError(
                f"Schema file must be a YAML mapping, got {type(raw).__name__}"
            )

        # Transform YAML `from`/`to` keys to `from_types`/`to_types`.
        for rt_data in (raw.get("relationship_types") or {}).values():
            if isinstance(rt_data, dict):
                if "from" in rt_data:
                    rt_data["from_types"] = rt_data.pop("from")
                if "to" in rt_data:
                    rt_data["to_types"] = rt_data.pop("to")

        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise SchemaLoadError(f"Invalid schema: {exc}") from exc

    def save(self, path: str | Path) -> None:
        """Save the schema to a YAML file.

        Args:
            path: Destination file path.

        Raises:
            SchemaLoadError: If the file cannot be written.

        Example:
            >>> schema.save("./wiki/hyphex.schema.yml")
        """
        data = self.model_dump(mode="json", exclude_none=True)

        # Transform `from_types`/`to_types` back to `from`/`to` for YAML.
        for rt_data in data.get("relationship_types", {}).values():
            if "from_types" in rt_data:
                rt_data["from"] = rt_data.pop("from_types")
            if "to_types" in rt_data:
                rt_data["to"] = rt_data.pop("to_types")

        path = Path(path)
        try:
            path.write_text(
                yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise SchemaLoadError(f"Failed to save schema to {path}: {exc}") from exc

    def validate(self, page: Page) -> list[SchemaError]:  # type: ignore[override]
        """Validate a page against this schema.

        Returns structured findings — never raises for validation failures.

        Args:
            page: The parsed Page to validate.

        Returns:
            A list of ``SchemaError`` findings. Empty means valid.

        Example:
            >>> errors = schema.validate(page)
            >>> [str(e) for e in errors]
            ["warning: Unknown relationship 'works_with'"]
        """
        if self.settings.enforcement == Enforcement.PERMISSIVE:
            return []

        severity: _Severity = (
            "error" if self.settings.enforcement == Enforcement.STRICT else "warning"
        )

        errors: list[SchemaError] = []
        errors.extend(_check_page_type(page, self, severity))
        errors.extend(_check_required_fields(page, self, severity))
        errors.extend(_check_field_values(page, self, severity))
        errors.extend(_check_relationships(page, self, severity))
        return errors

    def valid_relationships(
        self,
        *,
        from_type: str | None = None,
        to_type: str | None = None,
    ) -> dict[str, RelationshipType]:
        """Return relationship types matching the given constraints.

        A relationship matches if its constraint list is ``None``
        (unconstrained) or contains the given type.

        Args:
            from_type: Filter to relationships whose ``from_types`` is
                unconstrained or includes this type.
            to_type: Filter to relationships whose ``to_types`` is
                unconstrained or includes this type.

        Returns:
            Dict mapping relationship name to ``RelationshipType``.

        Example:
            >>> rels = schema.valid_relationships(from_type="project")
            >>> "client" in rels
            True
        """
        result: dict[str, RelationshipType] = {}
        for name, rt in self.relationship_types.items():
            if from_type is not None and rt.from_types is not None:
                if from_type not in rt.from_types:
                    continue
            if to_type is not None and rt.to_types is not None:
                if to_type not in rt.to_types:
                    continue
            result[name] = rt
        return result

    def to_prompt(self) -> str:
        """Generate a markdown summary for agent prompt injection.

        Returns:
            A markdown string describing available entity types and
            relationships, rendered from a Jinja2 template.

        Example:
            >>> print(schema.to_prompt())
            ## Available Entity Types
            ...
        """
        template = _PROMPTS_ENV.get_template("schema_summary.md.j2")
        return template.render(
            context=self.context,
            node_types=self.node_types,
            relationship_types=self.relationship_types,
            default_relationship=self.settings.default_relationship,
        )


# ── SchemaBuilder ─────────────────────────────────────────────────────


class SchemaBuilder:
    """Fluent builder for constructing Schema instances programmatically.

    Example:
        >>> schema = (
        ...     SchemaBuilder()
        ...     .add_node_type("project", description="A delivery")
        ...     .add_field("project", "client", field_type="link", relationship="client", required=True)
        ...     .add_relationship("client", description="For this client",
        ...                       from_types=["project"], to_types=["organisation"])
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        self._context: str = ""
        self._node_types: dict[str, NodeType] = {}
        self._relationship_types: dict[str, RelationshipType] = {}
        self._settings: SchemaSettings = SchemaSettings()

    def set_context(self, context: str) -> SchemaBuilder:
        """Set the knowledge base context.

        Explains why this knowledge base exists and what to prioritise
        when extracting information. Injected into agent prompts.

        Args:
            context: Free-text description.

        Returns:
            ``self`` for chaining.

        Example:
            >>> SchemaBuilder().set_context("We are building a knowledge base of...")
            <...SchemaBuilder...>
        """
        self._context = context
        return self

    def add_node_type(self, name: str, *, description: str = "") -> SchemaBuilder:
        """Add a node type to the schema.

        Args:
            name: The node type name.
            description: Human-readable description.

        Returns:
            ``self`` for chaining.

        Example:
            >>> SchemaBuilder().add_node_type("person", description="An individual")
            <...SchemaBuilder...>
        """
        self._node_types[name] = NodeType(description=description)
        return self

    def add_field(
        self,
        node_type: str,
        field_name: str,
        *,
        field_type: str = "string",
        required: bool = False,
        values: list[str] | None = None,
        relationship: str | None = None,
    ) -> SchemaBuilder:
        """Add a field to an existing node type.

        Args:
            node_type: The node type to add the field to.
            field_name: The field name.
            field_type: Field type (``"string"``, ``"integer"``, ``"enum"``,
                ``"link"``).
            required: Whether the field is required.
            values: Allowed values for enum fields.
            relationship: Relationship type for link fields.

        Returns:
            ``self`` for chaining.

        Raises:
            SchemaLoadError: If the node type does not exist.

        Example:
            >>> builder.add_field("project", "status", field_type="enum", values=["won", "lost"])
            <...SchemaBuilder...>
        """
        if node_type not in self._node_types:
            raise SchemaLoadError(f"Node type '{node_type}' does not exist")
        self._node_types[node_type].fields[field_name] = FieldDef(
            type=FieldType(field_type),
            required=required,
            values=values,
            relationship=relationship,
        )
        return self

    def add_relationship(
        self,
        name: str,
        *,
        description: str = "",
        from_types: list[str] | None = None,
        to_types: list[str] | None = None,
    ) -> SchemaBuilder:
        """Add a relationship type.

        Args:
            name: Relationship name.
            description: Human-readable description.
            from_types: Allowed source node types, or ``None``.
            to_types: Allowed target node types, or ``None``.

        Returns:
            ``self`` for chaining.

        Example:
            >>> builder.add_relationship("client", from_types=["project"], to_types=["organisation"])
            <...SchemaBuilder...>
        """
        self._relationship_types[name] = RelationshipType(
            description=description,
            from_types=from_types,
            to_types=to_types,
        )
        return self

    def set_settings(self, **kwargs: Any) -> SchemaBuilder:
        """Configure schema settings.

        Accepts the same keyword arguments as ``SchemaSettings``:
        ``enforcement``, ``auto_create_stubs``, ``default_relationship``,
        ``warn_on_default_relationship``. Unspecified values keep their
        defaults.

        Args:
            **kwargs: Settings fields to override.

        Returns:
            ``self`` for chaining.

        Example:
            >>> builder.set_settings(enforcement="strict")
            <...SchemaBuilder...>
        """
        self._settings = SchemaSettings(**kwargs)
        return self

    def build(self) -> Schema:
        """Build the final Schema instance.

        Returns:
            A validated ``Schema``.

        Example:
            >>> schema = SchemaBuilder().build()
        """
        return Schema(
            context=self._context,
            node_types=dict(self._node_types),
            relationship_types=dict(self._relationship_types),
            settings=self._settings,
        )
