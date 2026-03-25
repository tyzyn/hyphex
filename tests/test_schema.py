from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyphex import (
    Direction,
    EntityLink,
    Page,
    Schema,
    SchemaBuilder,
    SchemaError,
    SchemaLoadError,
)
from hyphex.schema import Enforcement, FieldDef, FieldType, NodeType, RelationshipType, SchemaSettings


# ── Helpers ───────────────────────────────────────────────────────────


SPEC_SCHEMA_YAML = """\
node_types:
  person:
    description: "An individual"
    fields:
      role: { type: string, required: false }
      organisation: { type: link, required: false, relationship: member_of }
  organisation:
    description: "A company, government body, or institution"
    fields:
      sector: { type: link, required: false, relationship: sector }
      website: { type: string, required: false }
  project:
    description: "A specific engagement or delivery"
    fields:
      client: { type: link, required: true, relationship: client }
      status: { type: enum, values: [won, lost, in_progress, completed], required: true }
      year: { type: integer, required: false }

relationship_types:
  client:
    description: "The engagement was for this client"
    from: [project]
    to: [organisation]
  capability_demonstrated:
    description: "This work demonstrated this capability"
    from: [project]
    to: [capability]
  sector:
    description: "Belongs to this industry sector"
    from: [organisation, project]
    to: [sector]
  authored_by:
    description: "Created or written by this person"
    to: [person]
  relates_to:
    description: "General association"

settings:
  enforcement: warn
  auto_create_stubs: true
  default_relationship: relates_to
  warn_on_default_relationship: true
"""


@pytest.fixture
def spec_schema(tmp_path: Path) -> Schema:
    path = tmp_path / "hyphex.schema.yml"
    path.write_text(SPEC_SCHEMA_YAML)
    return Schema.load(path)


def _make_page(
    page_type: str | None = None,
    frontmatter: dict | None = None,
    entity_links: list[EntityLink] | None = None,
) -> Page:
    fm = dict(frontmatter or {})
    if page_type is not None:
        fm["type"] = page_type
    return Page(
        frontmatter=fm,
        body="test",
        entity_links=entity_links or [],
    )


# ── FieldDef ──────────────────────────────────────────────────────────


class TestFieldDef:
    def test_default_string_type(self) -> None:
        f = FieldDef()
        assert f.type == FieldType.STRING
        assert f.required is False

    def test_enum_without_values_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FieldDef(type=FieldType.ENUM)

    def test_enum_with_values(self) -> None:
        f = FieldDef(type=FieldType.ENUM, values=["a", "b"])
        assert f.values == ["a", "b"]

    def test_link_stores_relationship(self) -> None:
        f = FieldDef(type=FieldType.LINK, relationship="member_of")
        assert f.relationship == "member_of"


# ── NodeType / RelationshipType / SchemaSettings ──────────────────────


class TestModels:
    def test_node_type_empty_fields(self) -> None:
        nt = NodeType(description="Test")
        assert nt.fields == {}

    def test_relationship_unconstrained(self) -> None:
        rt = RelationshipType(description="General")
        assert rt.from_types is None
        assert rt.to_types is None

    def test_relationship_constrained(self) -> None:
        rt = RelationshipType(from_types=["project"], to_types=["organisation"])
        assert rt.from_types == ["project"]

    def test_settings_defaults(self) -> None:
        s = SchemaSettings()
        assert s.enforcement == Enforcement.WARN
        assert s.auto_create_stubs is True
        assert s.default_relationship == "relates_to"

    def test_empty_schema(self) -> None:
        s = Schema()
        assert s.node_types == {}
        assert s.relationship_types == {}


# ── Load / Save ───────────────────────────────────────────────────────


class TestLoadSave:
    def test_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "schema.yml"
        path.write_text(SPEC_SCHEMA_YAML)
        schema = Schema.load(path)

        assert "project" in schema.node_types
        assert "client" in schema.relationship_types
        assert schema.relationship_types["client"].from_types == ["project"]
        assert schema.relationship_types["client"].to_types == ["organisation"]
        assert schema.settings.enforcement == Enforcement.WARN

        # Save and reload.
        out_path = tmp_path / "schema_out.yml"
        schema.save(out_path)
        reloaded = Schema.load(out_path)
        assert reloaded.node_types.keys() == schema.node_types.keys()
        assert reloaded.relationship_types["client"].from_types == ["project"]

    def test_save_uses_from_to_keys(self, tmp_path: Path, spec_schema: Schema) -> None:
        out = tmp_path / "out.yml"
        spec_schema.save(out)
        raw = yaml.safe_load(out.read_text())
        assert "from" in raw["relationship_types"]["client"]
        assert "to" in raw["relationship_types"]["client"]
        assert "from_types" not in raw["relationship_types"]["client"]

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaLoadError, match="Failed to load"):
            Schema.load(tmp_path / "nonexistent.yml")

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yml"
        path.write_text(": : : bad")
        with pytest.raises(SchemaLoadError):
            Schema.load(path)

    def test_load_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yml"
        path.write_text("- item1\n- item2")
        with pytest.raises(SchemaLoadError, match="YAML mapping"):
            Schema.load(path)

    def test_from_to_key_transformation(self, spec_schema: Schema) -> None:
        assert spec_schema.relationship_types["authored_by"].from_types is None
        assert spec_schema.relationship_types["authored_by"].to_types == ["person"]
        assert spec_schema.relationship_types["relates_to"].from_types is None
        assert spec_schema.relationship_types["relates_to"].to_types is None


# ── SchemaBuilder ─────────────────────────────────────────────────────


class TestSchemaBuilder:
    def test_basic_build(self) -> None:
        schema = (
            SchemaBuilder()
            .add_node_type("project", description="A delivery")
            .add_field(
                "project", "client", field_type="link", relationship="client", required=True
            )
            .add_relationship(
                "client",
                description="For this client",
                from_types=["project"],
                to_types=["organisation"],
            )
            .build()
        )
        assert "project" in schema.node_types
        assert "client" in schema.node_types["project"].fields
        assert schema.node_types["project"].fields["client"].required is True
        assert "client" in schema.relationship_types

    def test_fluent_chaining(self) -> None:
        builder = SchemaBuilder()
        result = builder.add_node_type("a")
        assert result is builder

    def test_add_field_to_missing_type_raises(self) -> None:
        with pytest.raises(SchemaLoadError, match="does not exist"):
            SchemaBuilder().add_field("missing", "field")

    def test_set_settings(self) -> None:
        schema = SchemaBuilder().set_settings(enforcement="strict").build()
        assert schema.settings.enforcement == Enforcement.STRICT

    def test_build_empty(self) -> None:
        schema = SchemaBuilder().build()
        assert schema.node_types == {}
        assert schema.relationship_types == {}


# ── Validation ────────────────────────────────────────────────────────


class TestValidation:
    def test_permissive_skips_all(self, spec_schema: Schema) -> None:
        spec_schema.settings.enforcement = Enforcement.PERMISSIVE
        page = _make_page(page_type="nonexistent")
        assert spec_schema.validate(page) == []

    def test_unknown_page_type(self, spec_schema: Schema) -> None:
        page = _make_page(page_type="nonexistent")
        errors = spec_schema.validate(page)
        assert any("Unknown node type" in e.message for e in errors)

    def test_valid_page_type(self, spec_schema: Schema) -> None:
        page = _make_page(page_type="person")
        errors = spec_schema.validate(page)
        assert not any("Unknown node type" in e.message for e in errors)

    def test_no_type_field(self, spec_schema: Schema) -> None:
        page = _make_page()
        errors = spec_schema.validate(page)
        assert not any("Unknown node type" in e.message for e in errors)

    def test_missing_required_field(self, spec_schema: Schema) -> None:
        page = _make_page(page_type="project")
        errors = spec_schema.validate(page)
        msgs = [e.message for e in errors]
        assert any("client" in m and "missing" in m for m in msgs)
        assert any("status" in m and "missing" in m for m in msgs)

    def test_required_field_present(self, spec_schema: Schema) -> None:
        page = _make_page(
            page_type="project",
            frontmatter={"client": "[[HMRC]]", "status": "won"},
        )
        errors = spec_schema.validate(page)
        assert not any("missing" in e.message for e in errors)

    def test_enum_valid_value(self, spec_schema: Schema) -> None:
        page = _make_page(
            page_type="project",
            frontmatter={"client": "[[HMRC]]", "status": "won"},
        )
        errors = spec_schema.validate(page)
        assert not any("Invalid value" in e.message for e in errors)

    def test_enum_invalid_value(self, spec_schema: Schema) -> None:
        page = _make_page(
            page_type="project",
            frontmatter={"client": "[[HMRC]]", "status": "cancelled"},
        )
        errors = spec_schema.validate(page)
        assert any("Invalid value 'cancelled'" in e.message for e in errors)

    def test_integer_field_valid(self, spec_schema: Schema) -> None:
        page = _make_page(
            page_type="project",
            frontmatter={"client": "[[HMRC]]", "status": "won", "year": 2023},
        )
        errors = spec_schema.validate(page)
        assert not any("integer" in e.message for e in errors)

    def test_integer_field_invalid(self, spec_schema: Schema) -> None:
        page = _make_page(
            page_type="project",
            frontmatter={"client": "[[HMRC]]", "status": "won", "year": "not_a_number"},
        )
        errors = spec_schema.validate(page)
        assert any("must be an integer" in e.message for e in errors)

    def test_unknown_relationship(self, spec_schema: Schema) -> None:
        link = EntityLink(
            target="X",
            direction=Direction.OUTGOING,
            relationship="nonexistent",
        )
        page = _make_page(page_type="project", entity_links=[link])
        errors = spec_schema.validate(page)
        assert any("Unknown relationship 'nonexistent'" in e.message for e in errors)

    def test_known_relationship(self, spec_schema: Schema) -> None:
        link = EntityLink(
            target="HMRC",
            direction=Direction.OUTGOING,
            relationship="client",
        )
        page = _make_page(
            page_type="project",
            frontmatter={"client": "[[HMRC]]", "status": "won"},
            entity_links=[link],
        )
        errors = spec_schema.validate(page)
        assert not any("Unknown relationship" in e.message for e in errors)

    def test_relationship_from_constraint_violated(self, spec_schema: Schema) -> None:
        link = EntityLink(
            target="HMRC",
            direction=Direction.OUTGOING,
            relationship="client",
        )
        page = _make_page(page_type="person", entity_links=[link])
        errors = spec_schema.validate(page)
        assert any("not allowed from type 'person'" in e.message for e in errors)

    def test_relationship_from_unconstrained(self, spec_schema: Schema) -> None:
        link = EntityLink(
            target="Alice",
            direction=Direction.INCOMING,
            relationship="authored_by",
        )
        page = _make_page(page_type="project", entity_links=[link])
        errors = spec_schema.validate(page)
        assert not any("not allowed from" in e.message for e in errors)

    def test_default_relationship_warning(self, spec_schema: Schema) -> None:
        link = EntityLink(
            target="ML",
            relationship="relates_to",
            is_default=True,
        )
        page = _make_page(entity_links=[link])
        errors = spec_schema.validate(page)
        assert any("default relationship" in e.message for e in errors)
        assert all(e.severity == "warning" for e in errors if "default" in e.message)

    def test_default_relationship_no_warning(self, spec_schema: Schema) -> None:
        spec_schema.settings.warn_on_default_relationship = False
        link = EntityLink(
            target="ML",
            relationship="relates_to",
            is_default=True,
        )
        page = _make_page(entity_links=[link])
        errors = spec_schema.validate(page)
        assert not any("default relationship" in e.message for e in errors)

    def test_strict_produces_errors(self, spec_schema: Schema) -> None:
        spec_schema.settings.enforcement = Enforcement.STRICT
        page = _make_page(page_type="nonexistent")
        errors = spec_schema.validate(page)
        assert errors[0].severity == "error"

    def test_warn_produces_warnings(self, spec_schema: Schema) -> None:
        page = _make_page(page_type="nonexistent")
        errors = spec_schema.validate(page)
        assert errors[0].severity == "warning"

    def test_multiple_errors(self, spec_schema: Schema) -> None:
        link = EntityLink(
            target="X",
            direction=Direction.OUTGOING,
            relationship="nonexistent",
        )
        page = _make_page(page_type="project", entity_links=[link])
        errors = spec_schema.validate(page)
        # Missing required fields + unknown relationship.
        assert len(errors) >= 3


# ── valid_relationships ───────────────────────────────────────────────


class TestValidRelationships:
    def test_filter_by_from_type(self, spec_schema: Schema) -> None:
        rels = spec_schema.valid_relationships(from_type="project")
        assert "client" in rels
        assert "authored_by" in rels  # unconstrained from
        assert "relates_to" in rels

    def test_filter_by_to_type(self, spec_schema: Schema) -> None:
        rels = spec_schema.valid_relationships(to_type="person")
        assert "authored_by" in rels
        assert "client" not in rels  # to=[organisation]

    def test_no_filter(self, spec_schema: Schema) -> None:
        rels = spec_schema.valid_relationships()
        assert len(rels) == len(spec_schema.relationship_types)

    def test_both_filters(self, spec_schema: Schema) -> None:
        rels = spec_schema.valid_relationships(from_type="project", to_type="organisation")
        assert "client" in rels
        assert "authored_by" not in rels  # to=[person]


# ── to_prompt ─────────────────────────────────────────────────────────


class TestToPrompt:
    def test_contains_entity_types_header(self, spec_schema: Schema) -> None:
        output = spec_schema.to_prompt()
        assert "## Available Entity Types" in output

    def test_contains_relationships_header(self, spec_schema: Schema) -> None:
        output = spec_schema.to_prompt()
        assert "## Available Relationships" in output

    def test_required_fields_shown(self, spec_schema: Schema) -> None:
        output = spec_schema.to_prompt()
        assert "(requires: client, status)" in output

    def test_unconstrained_shows_any(self, spec_schema: Schema) -> None:
        output = spec_schema.to_prompt()
        assert "any → any" in output

    def test_constrained_shows_types(self, spec_schema: Schema) -> None:
        output = spec_schema.to_prompt()
        assert "project → organisation" in output

    def test_default_relationship_mentioned(self, spec_schema: Schema) -> None:
        output = spec_schema.to_prompt()
        assert "relates_to" in output

    def test_empty_schema(self) -> None:
        output = Schema().to_prompt()
        assert "## Available Entity Types" in output
