"""Tests for hyphex.agents.tools — create_stub tool factory."""

from __future__ import annotations

import pytest

from hyphex.agents.tools import _build_template
from hyphex.schema import SchemaBuilder
from hyphex.store import slugify


@pytest.fixture()
def schema():
    return (
        SchemaBuilder()
        .add_node_type("person", description="An individual")
        .add_field("person", "born", field_type="string")
        .add_field("person", "nationality", field_type="string")
        .add_field("person", "known_for", field_type="string")
        .add_node_type("organisation", description="A company or institution")
        .add_field("organisation", "founded", field_type="integer")
        .add_field("organisation", "sector", field_type="enum", values=["public", "private"])
        .add_relationship("works_at", from_types=["person"], to_types=["organisation"])
        .build()
    )


class TestBuildTemplate:
    def test_basic_person_template(self, schema):
        template = _build_template("Michelle Obama", "person", schema)
        assert "type: person" in template
        assert "# Michelle Obama" in template
        assert "sources: []" in template

    def test_includes_all_fields(self, schema):
        template = _build_template("Michelle Obama", "person", schema)
        assert "born:" in template
        assert "nationality:" in template
        assert "known_for:" in template

    def test_integer_field_default(self, schema):
        template = _build_template("ACME Corp", "organisation", schema)
        assert "founded: 0" in template

    def test_enum_field_default(self, schema):
        template = _build_template("ACME Corp", "organisation", schema)
        assert "sector:" in template


class TestCreateStubTool:
    """Tests for the tool via make_create_stub_tool.

    These tests import langchain and are skipped if it's not installed.
    """

    @pytest.fixture()
    def stub_tool(self, schema):
        langchain = pytest.importorskip("langchain.tools")  # noqa: F841
        from hyphex.agents.tools import make_create_stub_tool

        blessed: set[str] = set()
        return make_create_stub_tool(schema, blessed), blessed

    def test_returns_path_and_template(self, stub_tool):
        tool, _ = stub_tool
        result = tool.invoke({"title": "Michelle Obama", "entity_type": "person"})
        assert result["path"] == "michelle_obama.md"
        assert "# Michelle Obama" in result["template"]

    def test_blesses_path(self, stub_tool):
        tool, blessed = stub_tool
        tool.invoke({"title": "Michelle Obama", "entity_type": "person"})
        assert "michelle_obama.md" in blessed

    def test_unknown_entity_type_returns_error(self, stub_tool):
        tool, _ = stub_tool
        result = tool.invoke({"title": "Foo", "entity_type": "spaceship"})
        assert "error" in result
        assert "spaceship" in result["error"]
        assert "person" in result["error"]

    def test_empty_title_returns_error(self, stub_tool):
        tool, _ = stub_tool
        result = tool.invoke({"title": "!!!", "entity_type": "person"})
        assert "error" in result

    def test_idempotent_path(self, stub_tool):
        tool, blessed = stub_tool
        r1 = tool.invoke({"title": "Michelle Obama", "entity_type": "person"})
        r2 = tool.invoke({"title": "Michelle Obama", "entity_type": "person"})
        assert r1["path"] == r2["path"]
        assert len(blessed) == 1

    def test_path_matches_slugify(self, stub_tool):
        tool, _ = stub_tool
        result = tool.invoke({"title": "São Paulo", "entity_type": "organisation"})
        expected = f"{slugify('São Paulo')}.md"
        assert result["path"] == expected

    def test_special_characters_in_title(self, stub_tool):
        tool, _ = stub_tool
        result = tool.invoke(
            {"title": "HMRC — Digital Tax Platform", "entity_type": "organisation"}
        )
        assert result["path"].endswith(".md")
        assert "error" not in result
