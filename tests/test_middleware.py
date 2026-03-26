"""Tests for hyphex.agents.middleware — all four middleware classes.

Middleware logic is tested by calling internal methods directly and
by simulating tool call request/handler patterns without requiring
the full langchain agent runtime.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from hyphex._frontmatter import serialize_frontmatter, split_frontmatter, parse_frontmatter
from hyphex.schema import SchemaBuilder

langchain = pytest.importorskip("langchain")


from hyphex.agents.middleware import (  # noqa: E402
    CitationGroundingMiddleware,
    RejectEmptyPageMiddleware,
    RequireStubMiddleware,
    SchemaValidationMiddleware,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_request(
    tool_name: str, args: dict[str, Any], *, call_id: str = "call_1"
) -> MagicMock:
    """Build a mock ToolCallRequest."""
    request = MagicMock()
    request.tool_call = {"name": tool_name, "args": args, "id": call_id}
    request.tool = None
    request.state = {}
    request.runtime = MagicMock()
    return request


def _make_handler(return_content: str = "ok") -> MagicMock:
    """Build a mock handler that returns a ToolMessage-like object."""
    handler = MagicMock()
    result = MagicMock()
    result.content = return_content
    handler.return_value = result
    return handler


def _page_content(
    frontmatter: dict[str, Any], body: str
) -> str:
    return serialize_frontmatter(frontmatter, body)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def schema():
    return (
        SchemaBuilder()
        .add_node_type("person", description="An individual")
        .add_field("person", "born", field_type="string")
        .add_node_type("organisation", description="A company or institution")
        .add_relationship("works_at", from_types=["person"], to_types=["organisation"])
        .add_relationship("knows", description="Personal relationship")
        .build()
    )


# ── RequireStubMiddleware ─────────────────────────────────────────────


class TestRequireStubMiddleware:
    def test_allows_blessed_path(self):
        mw = RequireStubMiddleware({"michelle_obama.md"})
        request = _make_request("write_file", {"file_path": "michelle_obama.md", "content": "x"})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
        assert result == handler.return_value

    def test_rejects_unblessed_md_path(self):
        mw = RequireStubMiddleware(set())
        request = _make_request("write_file", {"file_path": "unknown.md", "content": "x"})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "create_stub" in result.content

    def test_allows_edit_file(self):
        mw = RequireStubMiddleware(set())
        request = _make_request(
            "edit_file", {"file_path": "whatever.md", "old_string": "a", "new_string": "b"}
        )
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_allows_non_md_write(self):
        mw = RequireStubMiddleware(set())
        request = _make_request("write_file", {"file_path": "notes.txt", "content": "x"})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_allows_blessed_absolute_path(self):
        mw = RequireStubMiddleware({"michelle_obama.md"})
        request = _make_request(
            "write_file", {"file_path": "/michelle_obama.md", "content": "x"}
        )
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_allows_non_write_tools(self):
        mw = RequireStubMiddleware(set())
        request = _make_request("read_file", {"file_path": "anything.md"})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_called_once()


# ── RejectEmptyPageMiddleware ─────────────────────────────────────────


class TestRejectEmptyPageMiddleware:
    def test_allows_stub_without_refs(self):
        mw = RejectEmptyPageMiddleware()
        content = _page_content({"type": "person"}, "\n# Stub Page\n\n")
        request = _make_request("write_file", {"file_path": "stub.md", "content": content})
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_allows_page_with_refs_and_content(self):
        mw = RejectEmptyPageMiddleware()
        body = (
            "\n# Michelle Obama\n\n"
            "Michelle LaVaughn Robinson Obama is an American attorney and author "
            "who served as the first lady of the United States. {{ref}}\n"
        )
        content = _page_content({"type": "person", "sources": []}, body)
        request = _make_request("write_file", {"file_path": "michelle_obama.md", "content": content})
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_rejects_page_with_refs_but_empty_body(self):
        mw = RejectEmptyPageMiddleware()
        body = "\n# Title\n\n{{ref}}\n"
        content = _page_content({"type": "person"}, body)
        request = _make_request("write_file", {"file_path": "empty.md", "content": content})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "almost no content" in result.content

    def test_allows_edit_file_always(self):
        mw = RejectEmptyPageMiddleware()
        request = _make_request(
            "edit_file", {"file_path": "x.md", "old_string": "a", "new_string": "{{ref}}"}
        )
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()


class TestExtractBodyText:
    def test_strips_frontmatter_and_headings(self):
        content = _page_content({"type": "person"}, "\n# Title\n\n## Section\n\nBody text here.")
        result = RejectEmptyPageMiddleware._extract_body_text(content)
        assert "Title" not in result
        assert "Section" not in result
        assert "Body text here." in result

    def test_handles_no_frontmatter(self):
        result = RejectEmptyPageMiddleware._extract_body_text("# Just a heading\n\nSome text.")
        assert "Just a heading" not in result
        assert "Some text." in result


# ── SchemaValidationMiddleware ────────────────────────────────────────


class TestSchemaValidationMiddleware:
    def test_allows_valid_page(self, schema):
        mw = SchemaValidationMiddleware(schema)
        body = "\n# Alice\n\nWorks at [[ACME>works_at]].\n"
        content = _page_content({"type": "person", "born": "1990"}, body)
        request = _make_request("write_file", {"file_path": "alice.md", "content": content})
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_rejects_unknown_entity_type(self, schema):
        mw = SchemaValidationMiddleware(schema)
        content = _page_content({"type": "spaceship"}, "\n# USS Enterprise\n\n")
        request = _make_request("write_file", {"file_path": "uss.md", "content": content})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "spaceship" in result.content
        assert "person" in result.content

    def test_rejects_unknown_relationship(self, schema):
        mw = SchemaValidationMiddleware(schema)
        body = "\n# Alice\n\nLinked to [[Bob>invented_by]].\n"
        content = _page_content({"type": "person"}, body)
        request = _make_request("write_file", {"file_path": "alice.md", "content": content})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "invented_by" in result.content

    def test_allows_no_type_field(self, schema):
        mw = SchemaValidationMiddleware(schema)
        content = _page_content({}, "\n# Misc\n\n")
        request = _make_request("write_file", {"file_path": "misc.md", "content": content})
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_validates_edit_file_relationships(self, schema):
        mw = SchemaValidationMiddleware(schema)
        request = _make_request(
            "edit_file",
            {
                "file_path": "boo.md",
                "old_string": "old text",
                "new_string": "Linked to [[Bob>invented_by]].",
            },
        )
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "invented_by" in result.content

    def test_allows_edit_file_valid_relationship(self, schema):
        mw = SchemaValidationMiddleware(schema)
        request = _make_request(
            "edit_file",
            {
                "file_path": "alice.md",
                "old_string": "old",
                "new_string": "Works at [[ACME>works_at]].",
            },
        )
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_allows_non_md_file(self, schema):
        mw = SchemaValidationMiddleware(schema)
        request = _make_request("write_file", {"file_path": "data.json", "content": "{}"})
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_rejects_malformed_frontmatter(self, schema):
        mw = SchemaValidationMiddleware(schema)
        content = "---\n: invalid yaml [\n---\nBody"
        request = _make_request("write_file", {"file_path": "bad.md", "content": content})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "Malformed frontmatter" in result.content


# ── CitationGroundingMiddleware ───────────────────────────────────────


class TestCitationGroundingMiddleware:
    """Tests for post-write citation fixup on disk."""

    def _make_mw(self, tmp_path):
        return CitationGroundingMiddleware(tmp_path, "https://example.com/doc", "My Document")

    def _write_page(self, tmp_path, filename, frontmatter, body):
        """Write a page to the tmp wiki dir and return a handler that simulates the write."""
        content = _page_content(frontmatter, body)
        (tmp_path / filename).write_text(content, encoding="utf-8")

        def handler(req):
            result = MagicMock()
            result.content = "ok"
            return result

        return handler

    def test_resolves_refs_on_write_file(self, tmp_path):
        mw = self._make_mw(tmp_path)
        body = "\nFact one {{ref}}. Fact two {{ref}}.\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "test.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/test.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "test.md").read_text()
        _, resolved_body = split_frontmatter(fixed)
        assert resolved_body.count("{{1}}") == 2
        assert "{{ref}}" not in resolved_body

    def test_adds_source_to_frontmatter(self, tmp_path):
        mw = self._make_mw(tmp_path)
        body = "\nClaim {{ref}}.\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "test.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/test.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "test.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 1
        assert fm["sources"][0]["id"] == 1
        assert fm["sources"][0]["title"] == "My Document"
        assert fm["sources"][0]["url"] == "https://example.com/doc"

    def test_increments_from_existing_sources(self, tmp_path):
        mw = CitationGroundingMiddleware(tmp_path, "doc_2", "Second Source")
        existing = [{"id": 1, "title": "First", "url": "http://first.com"}]
        body = "\nNew claim {{ref}}.\n"
        content = _page_content({"type": "person", "sources": existing}, body)
        (tmp_path / "page.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/page.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "page.md").read_text()
        fm_yaml, resolved_body = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert "{{2}}" in resolved_body
        assert len(fm["sources"]) == 2
        assert fm["sources"][1]["id"] == 2

    def test_reuses_id_for_same_source(self, tmp_path):
        mw = self._make_mw(tmp_path)
        existing = [{"id": 1, "title": "My Document", "url": "https://example.com/doc"}]
        body = "\nAnother fact {{ref}}.\n"
        content = _page_content({"type": "person", "sources": existing}, body)
        (tmp_path / "page.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/page.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "page.md").read_text()
        fm_yaml, resolved_body = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert "{{1}}" in resolved_body
        assert len(fm["sources"]) == 1  # no duplicate

    def test_no_fixup_without_refs(self, tmp_path):
        mw = self._make_mw(tmp_path)
        body = "\nNo citations here.\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "clean.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/clean.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        # File should be unchanged — no sources added
        after = (tmp_path / "clean.md").read_text()
        assert after == content

    def test_resolves_refs_in_edit_file(self, tmp_path):
        mw = self._make_mw(tmp_path)
        # Existing page with no refs
        body = "\nOld content.\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "person.md").write_text(content, encoding="utf-8")

        # Simulate edit_file that adds a ref — the handler writes the patched file
        new_body = "\nOld content.\n\nNew fact {{ref}}.\n"
        new_content = _page_content({"type": "person", "sources": []}, new_body)
        (tmp_path / "person.md").write_text(new_content, encoding="utf-8")

        request = _make_request(
            "edit_file",
            {"file_path": "/person.md", "old_string": "x", "new_string": "New fact {{ref}}."},
        )
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "person.md").read_text()
        fm_yaml, resolved_body = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert "{{1}}" in resolved_body
        assert "{{ref}}" not in resolved_body
        assert len(fm["sources"]) == 1

    def test_ignores_non_md_files(self, tmp_path):
        mw = self._make_mw(tmp_path)
        (tmp_path / "data.json").write_text('{"x": "{{ref}}"}', encoding="utf-8")
        request = _make_request("write_file", {"file_path": "/data.json", "content": "{{ref}}"})
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)


# ── Middleware chain integration ──────────────────────────────────────


class TestMiddlewareChain:
    """Test that middleware compose correctly in execution order."""

    def test_require_stub_blocks_before_other_middleware(self, schema, tmp_path):
        blessed: set[str] = set()
        chain = [
            RequireStubMiddleware(blessed),
            RejectEmptyPageMiddleware(),
            SchemaValidationMiddleware(schema),
            CitationGroundingMiddleware(tmp_path, "doc_1", "Source"),
        ]

        content = _page_content({"type": "person"}, "\n# Test\n\n")
        request = _make_request("write_file", {"file_path": "test.md", "content": content})
        handler = _make_handler()

        def run_chain(mws, req, final_handler):
            if not mws:
                return final_handler(req)
            return mws[0].wrap_tool_call(
                req, lambda r: run_chain(mws[1:], r, final_handler)
            )

        result = run_chain(chain, request, handler)
        handler.assert_not_called()
        assert "create_stub" in result.content

    def test_blessed_path_passes_full_chain(self, schema, tmp_path):
        blessed: set[str] = {"good_page.md"}
        chain = [
            RequireStubMiddleware(blessed),
            RejectEmptyPageMiddleware(),
            SchemaValidationMiddleware(schema),
            CitationGroundingMiddleware(tmp_path, "doc_1", "Source"),
        ]

        body = (
            "\n# Good Page\n\n"
            "This is a well-written page with real content about the topic. {{ref}}\n"
        )
        content = _page_content({"type": "person", "sources": []}, body)
        request = _make_request("write_file", {"file_path": "/good_page.md", "content": content})

        def write_handler(req):
            """Simulate the actual write to disk so post-write fixup can read it."""
            (tmp_path / "good_page.md").write_text(
                req.tool_call["args"]["content"], encoding="utf-8"
            )
            result = MagicMock()
            result.content = "ok"
            return result

        def run_chain(mws, req, handler):
            if not mws:
                return handler(req)
            return mws[0].wrap_tool_call(
                req, lambda r: run_chain(mws[1:], r, handler)
            )

        run_chain(chain, request, write_handler)

        # Post-write fixup should have resolved citations on disk
        fixed = (tmp_path / "good_page.md").read_text()
        assert "{{ref}}" not in fixed
        assert "{{1}}" in fixed
