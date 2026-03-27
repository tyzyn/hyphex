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

    def test_allows_page_with_citations_and_content(self):
        mw = RejectEmptyPageMiddleware()
        body = (
            "\n# Michelle Obama\n\n"
            "Michelle LaVaughn Robinson Obama is an American attorney and author "
            "who served as the first lady of the United States. [@src_001, p. 1]\n"
        )
        content = _page_content({"type": "person", "sources": []}, body)
        request = _make_request("write_file", {"file_path": "michelle_obama.md", "content": content})
        handler = _make_handler()
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_rejects_page_with_citations_but_empty_body(self):
        mw = RejectEmptyPageMiddleware()
        body = "\n# Title\n\n[@src_001, p. 1]\n"
        content = _page_content({"type": "person"}, body)
        request = _make_request("write_file", {"file_path": "empty.md", "content": content})
        handler = _make_handler()
        result = mw.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "almost no content" in result.content

    def test_allows_edit_file_always(self):
        mw = RejectEmptyPageMiddleware()
        request = _make_request(
            "edit_file", {"file_path": "x.md", "old_string": "a", "new_string": "[@src_001]"}
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
    """Tests for post-write pandoc citation source management."""

    def _make_mw(self, tmp_path):
        return CitationGroundingMiddleware(
            tmp_path, "src_001", "https://example.com/doc", "My Document"
        )

    def test_adds_source_on_write(self, tmp_path):
        mw = self._make_mw(tmp_path)
        body = "\nClaim [@src_001, p. 5].\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "test.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/test.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "test.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 1
        assert fm["sources"][0]["id"] == "src_001"
        assert fm["sources"][0]["title"] == "My Document"
        assert fm["sources"][0]["url"] == "https://example.com/doc"

    def test_adds_source_on_edit(self, tmp_path):
        mw = self._make_mw(tmp_path)
        body = "\nOld content.\n\nNew fact [@src_001, p. 3].\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "page.md").write_text(content, encoding="utf-8")

        request = _make_request(
            "edit_file",
            {"file_path": "/page.md", "old_string": "x", "new_string": "[@src_001, p. 3]"},
        )
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "page.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 1
        assert fm["sources"][0]["id"] == "src_001"

    def test_does_not_duplicate_existing_source(self, tmp_path):
        mw = self._make_mw(tmp_path)
        existing = [{"id": "src_001", "title": "My Document", "url": "https://example.com/doc"}]
        body = "\nAnother fact [@src_001, p. 10].\n"
        content = _page_content({"type": "person", "sources": existing}, body)
        (tmp_path / "page.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/page.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "page.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 1

    def test_preserves_other_sources(self, tmp_path):
        mw = CitationGroundingMiddleware(
            tmp_path, "src_002", "https://other.com", "Other Doc"
        )
        existing = [{"id": "src_001", "title": "First", "url": "https://first.com"}]
        body = "\nClaim from second source [@src_002, p. 1].\n"
        content = _page_content({"type": "person", "sources": existing}, body)
        (tmp_path / "page.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/page.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "page.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 2
        ids = {s["id"] for s in fm["sources"]}
        assert ids == {"src_001", "src_002"}

    def test_no_fixup_without_citations(self, tmp_path):
        mw = self._make_mw(tmp_path)
        body = "\nNo citations here.\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "clean.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/clean.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        after = (tmp_path / "clean.md").read_text()
        assert after == content

    def test_strips_invalid_source_entries(self, tmp_path):
        mw = self._make_mw(tmp_path)
        invalid = [{"title": "Source Document"}, {"id": "src_001", "url": "https://example.com/doc", "title": "My Document"}]
        body = "\nFact [@src_001, p. 1].\n"
        content = _page_content({"type": "person", "sources": invalid}, body)
        (tmp_path / "page.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/page.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "page.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 1
        assert fm["sources"][0]["id"] == "src_001"

    def test_citation_without_page_number(self, tmp_path):
        mw = self._make_mw(tmp_path)
        body = "\nGeneral claim [@src_001].\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "test.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/test.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "test.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 1

    def test_page_range_citation(self, tmp_path):
        mw = self._make_mw(tmp_path)
        body = "\nSpanning claim [@src_001, pp. 12-15].\n"
        content = _page_content({"type": "person", "sources": []}, body)
        (tmp_path / "test.md").write_text(content, encoding="utf-8")

        request = _make_request("write_file", {"file_path": "/test.md", "content": content})
        mw.wrap_tool_call(request, _make_handler())

        fixed = (tmp_path / "test.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 1

    def test_ignores_non_md_files(self, tmp_path):
        mw = self._make_mw(tmp_path)
        request = _make_request("write_file", {"file_path": "/data.json", "content": "[@src_001]"})
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
            CitationGroundingMiddleware(tmp_path, "src_001", "doc_1", "Source"),
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
            CitationGroundingMiddleware(tmp_path, "src_001", "doc_1", "Source"),
        ]

        body = (
            "\n# Good Page\n\n"
            "This is a well-written page with real content about the topic. "
            "[@src_001, p. 1]\n"
        )
        content = _page_content({"type": "person", "sources": []}, body)
        request = _make_request("write_file", {"file_path": "/good_page.md", "content": content})

        def write_handler(req):
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

        fixed = (tmp_path / "good_page.md").read_text()
        fm_yaml, _ = split_frontmatter(fixed)
        fm = parse_frontmatter(fm_yaml)
        assert len(fm["sources"]) == 1
        assert fm["sources"][0]["id"] == "src_001"
