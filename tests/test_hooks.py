from __future__ import annotations

from pathlib import Path

import pytest

from hyphex import (
    HookError,
    HookRegistry,
    Page,
    Parser,
    Schema,
    SchemaBuilder,
    SourceDocument,
    WikiStore,
    WriteContext,
    citation_inject,
    cleanup_and_resolve,
    register_builtin_hooks,
)
from hyphex.models import Citation, EntityLink


# ── Helpers ───────────────────────────────────────────────────────────


def _ctx(
    title: str = "Test",
    body: str = "",
    frontmatter: dict | None = None,
    source_document: SourceDocument | None = None,
    entity_links: list[EntityLink] | None = None,
    citations: list[Citation] | None = None,
    **kwargs: object,
) -> WriteContext:
    page = Page(
        frontmatter=frontmatter or {},
        body=body,
        entity_links=entity_links or [],
        citations=citations or [],
        evidence_blocks=[],
    )
    return WriteContext(title=title, page=page, source_document=source_document, **kwargs)  # type: ignore[arg-type]


# ── SourceDocument ────────────────────────────────────────────────────


class TestSourceDocument:
    def test_basic(self) -> None:
        doc = SourceDocument(url="https://example.com", title="Report")
        assert doc.url == "https://example.com"
        assert doc.section is None

    def test_all_fields(self) -> None:
        doc = SourceDocument(
            url="https://example.com", title="Report", page=14, section="Intro", hash="abc"
        )
        assert doc.page == 14
        assert doc.section == "Intro"


# ── WriteContext ──────────────────────────────────────────────────────


class TestWriteContext:
    def test_minimal(self) -> None:
        ctx = _ctx()
        assert ctx.title == "Test"
        assert ctx.warnings == []
        assert ctx.store is None
        assert ctx.parser is None
        assert ctx.schema_ref is None

    def test_add_warning(self) -> None:
        ctx = _ctx()
        ctx.add_warning("something bad")
        assert ctx.warnings == ["something bad"]

    def test_agent_context_default(self) -> None:
        ctx = _ctx()
        assert ctx.agent_context == {}


# ── HookRegistry ─────────────────────────────────────────────────────


class TestHookRegistry:
    def test_register_and_run(self) -> None:
        registry = HookRegistry()
        called = []

        def hook(ctx: WriteContext) -> WriteContext:
            called.append("ran")
            return ctx

        registry.register("pre_write", hook)
        registry.run("pre_write", _ctx())
        assert called == ["ran"]

    def test_hooks_run_in_order(self) -> None:
        registry = HookRegistry()
        order: list[int] = []

        def first(ctx: WriteContext) -> WriteContext:
            order.append(1)
            return ctx

        def second(ctx: WriteContext) -> WriteContext:
            order.append(2)
            return ctx

        registry.register("post_write", first)
        registry.register("post_write", second)
        registry.run("post_write", _ctx())
        assert order == [1, 2]

    def test_decorator(self) -> None:
        registry = HookRegistry()

        @registry.hook("pre_write")
        def my_hook(ctx: WriteContext) -> WriteContext:
            return ctx

        assert my_hook.__name__ == "my_hook"
        ctx = registry.run("pre_write", _ctx())
        assert isinstance(ctx, WriteContext)

    def test_invalid_event_raises(self) -> None:
        registry = HookRegistry()
        with pytest.raises(HookError, match="Invalid event"):
            registry.register("invalid", lambda ctx: ctx)

    def test_run_invalid_event_raises(self) -> None:
        registry = HookRegistry()
        with pytest.raises(HookError, match="Invalid event"):
            registry.run("invalid", _ctx())

    def test_exception_wrapped(self) -> None:
        registry = HookRegistry()

        def bad_hook(ctx: WriteContext) -> WriteContext:
            msg = "boom"
            raise RuntimeError(msg)

        registry.register("pre_write", bad_hook)
        with pytest.raises(HookError, match="bad_hook"):
            registry.run("pre_write", _ctx())

    def test_hook_error_not_double_wrapped(self) -> None:
        registry = HookRegistry()

        def hook_raises(ctx: WriteContext) -> WriteContext:
            raise HookError("direct")

        registry.register("pre_write", hook_raises)
        with pytest.raises(HookError, match="direct"):
            registry.run("pre_write", _ctx())

    def test_clear_specific(self) -> None:
        registry = HookRegistry()
        registry.register("pre_write", lambda ctx: ctx)
        registry.register("post_write", lambda ctx: ctx)
        registry.clear("pre_write")
        # pre_write is empty, post_write still has one.
        assert registry._hooks["pre_write"] == []
        assert len(registry._hooks["post_write"]) == 1

    def test_clear_all(self) -> None:
        registry = HookRegistry()
        registry.register("pre_write", lambda ctx: ctx)
        registry.register("post_write", lambda ctx: ctx)
        registry.clear()
        assert registry._hooks["pre_write"] == []
        assert registry._hooks["post_write"] == []

    def test_hook_modifies_context(self) -> None:
        registry = HookRegistry()

        def add_hint(ctx: WriteContext) -> WriteContext:
            ac = dict(ctx.agent_context)
            ac["hint"] = "hello"
            return ctx.model_copy(update={"agent_context": ac})

        registry.register("pre_write", add_hint)
        result = registry.run("pre_write", _ctx())
        assert result.agent_context["hint"] == "hello"


# ── citation_inject ──────────────────────────────────────────────────


class TestCitationInject:
    def test_no_source_document_noop(self) -> None:
        ctx = _ctx()
        result = citation_inject(ctx)
        assert result.page.frontmatter.get("sources") is None

    def test_first_source(self) -> None:
        doc = SourceDocument(url="https://example.com", title="Report")
        ctx = _ctx(source_document=doc)
        result = citation_inject(ctx)
        sources = result.page.frontmatter["sources"]
        assert len(sources) == 1
        assert sources[0]["id"] == "src_001"
        assert sources[0]["title"] == "Report"

    def test_incremental_id(self) -> None:
        doc = SourceDocument(url="https://example.com", title="New")
        ctx = _ctx(
            source_document=doc,
            frontmatter={"sources": [{"id": "src_001", "title": "Old", "url": "https://old.com"}]},
        )
        result = citation_inject(ctx)
        sources = result.page.frontmatter["sources"]
        assert len(sources) == 2
        assert sources[1]["id"] == "src_002"

    def test_citation_hint(self) -> None:
        doc = SourceDocument(url="https://example.com", title="Report")
        ctx = _ctx(source_document=doc)
        result = citation_inject(ctx)
        assert result.agent_context["citation_hint"] == "[@src_001]"

    def test_optional_fields_included(self) -> None:
        doc = SourceDocument(
            url="https://example.com", title="Report", page=14, section="Intro", hash="abc"
        )
        ctx = _ctx(source_document=doc)
        result = citation_inject(ctx)
        source = result.page.frontmatter["sources"][0]
        assert source["page"] == 14
        assert source["section"] == "Intro"
        assert source["hash"] == "abc"

    def test_optional_fields_excluded(self) -> None:
        doc = SourceDocument(url="https://example.com", title="Report")
        ctx = _ctx(source_document=doc)
        result = citation_inject(ctx)
        source = result.page.frontmatter["sources"][0]
        assert "page" not in source
        assert "section" not in source


# ── cleanup_and_resolve ──────────────────────────────────────────────


class TestCleanupAndResolve:
    def test_removes_uncited_sources(self) -> None:
        ctx = _ctx(
            body="Cited. [@src_001]",
            frontmatter={
                "sources": [
                    {"id": "src_001", "title": "Cited", "url": "https://a.com"},
                    {"id": "src_002", "title": "Uncited", "url": "https://b.com"},
                ]
            },
        )
        result = cleanup_and_resolve(ctx)
        sources = result.page.frontmatter["sources"]
        assert len(sources) == 1
        assert sources[0]["title"] == "Cited"

    def test_removes_uncited_keeps_cited(self) -> None:
        ctx = _ctx(
            body="First. [@src_001] Third. [@src_003]",
            frontmatter={
                "sources": [
                    {"id": "src_001", "title": "A", "url": "https://a.com"},
                    {"id": "src_002", "title": "B", "url": "https://b.com"},
                    {"id": "src_003", "title": "C", "url": "https://c.com"},
                ]
            },
        )
        result = cleanup_and_resolve(ctx)
        sources = result.page.frontmatter["sources"]
        assert [s["id"] for s in sources] == ["src_001", "src_003"]
        assert "[@src_001]" in result.page.body
        assert "[@src_003]" in result.page.body

    def test_creates_stubs(self, tmp_path: Path) -> None:
        store = WikiStore(tmp_path)
        links = [EntityLink(target="Python", relationship="uses")]
        ctx = _ctx(body="[Python](rel:uses)", entity_links=links, store=store)
        cleanup_and_resolve(ctx)
        assert store.exists("Python")

    def test_no_stub_if_exists(self, tmp_path: Path) -> None:
        store = WikiStore(tmp_path)
        store.create_stub("Python", page_type="technology")
        original = store.read("Python")

        links = [EntityLink(target="Python", relationship="uses")]
        ctx = _ctx(body="[Python](rel:uses)", entity_links=links, store=store)
        cleanup_and_resolve(ctx)
        after = store.read("Python")
        assert after.frontmatter["type"] == original.frontmatter["type"]

    def test_schema_validation_warnings(self) -> None:
        schema = SchemaBuilder().add_node_type("person").build()
        ctx = _ctx(body="", frontmatter={"type": "nonexistent"}, schema=schema)
        result = cleanup_and_resolve(ctx)
        assert any("Unknown node type" in w for w in result.warnings)

    def test_works_without_parser(self) -> None:
        ctx = _ctx(body="Fact. [@src_001]", frontmatter={"sources": [{"id": "src_001", "title": "A", "url": "https://a.com"}]})
        result = cleanup_and_resolve(ctx)
        assert len(result.page.frontmatter["sources"]) == 1

    def test_works_without_store(self) -> None:
        links = [EntityLink(target="Missing", relationship="relates_to")]
        ctx = _ctx(body="[[Missing]]", entity_links=links)
        result = cleanup_and_resolve(ctx)
        assert isinstance(result, WriteContext)

    def test_works_without_schema(self) -> None:
        ctx = _ctx(body="Content")
        result = cleanup_and_resolve(ctx)
        assert result.warnings == []

    def test_with_parser(self) -> None:
        parser = Parser()
        ctx = _ctx(
            body="Fact. [@src_001] Link to [Python](rel:uses).",
            frontmatter={
                "sources": [
                    {"id": "src_001", "title": "A", "url": "https://a.com"},
                    {"id": "src_002", "title": "B", "url": "https://b.com"},
                ]
            },
            parser=parser,
        )
        result = cleanup_and_resolve(ctx)
        assert len(result.page.frontmatter["sources"]) == 1
        assert len(result.page.entity_links) == 1
        assert result.page.entity_links[0].target == "Python"


# ── register_builtin_hooks ───────────────────────────────────────────


class TestRegisterBuiltinHooks:
    def test_registers_both(self) -> None:
        registry = HookRegistry()
        register_builtin_hooks(registry)
        assert len(registry._hooks["pre_write"]) == 1
        assert len(registry._hooks["post_write"]) == 1


# ── WikiStore integration ────────────────────────────────────────────


class TestWikiStoreHookIntegration:
    def test_write_without_hooks(self, tmp_path: Path) -> None:
        store = WikiStore(tmp_path)
        result = store.write("Test", Page(body="Content"))
        assert result is None
        assert store.exists("Test")

    def test_write_returns_context(self, tmp_path: Path) -> None:
        registry = HookRegistry()
        store = WikiStore(tmp_path, hooks=registry)
        result = store.write("Test", Page(body="Content"))
        assert isinstance(result, WriteContext)

    def test_pre_write_modifies_page(self, tmp_path: Path) -> None:
        registry = HookRegistry()

        def add_tag(ctx: WriteContext) -> WriteContext:
            fm = dict(ctx.page.frontmatter)
            fm["tagged"] = True
            return ctx.model_copy(update={"page": ctx.page.model_copy(update={"frontmatter": fm})})

        registry.register("pre_write", add_tag)
        store = WikiStore(tmp_path, hooks=registry)
        store.write("Test", Page(body="Content"))
        page = store.read("Test")
        assert page.frontmatter["tagged"] is True

    def test_post_write_re_writes(self, tmp_path: Path) -> None:
        registry = HookRegistry()

        def change_body(ctx: WriteContext) -> WriteContext:
            return ctx.model_copy(
                update={"page": ctx.page.model_copy(update={"body": "Modified by hook"})}
            )

        registry.register("post_write", change_body)
        store = WikiStore(tmp_path, hooks=registry)
        store.write("Test", Page(body="Original"))
        page = store.read("Test")
        assert page.body == "Modified by hook"

    def test_source_document_passed(self, tmp_path: Path) -> None:
        registry = HookRegistry()
        register_builtin_hooks(registry)
        doc = SourceDocument(url="https://example.com", title="Report")
        store = WikiStore(tmp_path, hooks=registry)
        ctx = store.write("Test", Page(body="Fact. [@src_001]"), source_document=doc)
        assert ctx is not None
        assert ctx.agent_context["citation_hint"] == "[@src_001]"

    def test_warnings_accessible(self, tmp_path: Path) -> None:
        registry = HookRegistry()

        def warn_hook(ctx: WriteContext) -> WriteContext:
            ctx.add_warning("test warning")
            return ctx

        registry.register("post_write", warn_hook)
        store = WikiStore(tmp_path, hooks=registry)
        ctx = store.write("Test", Page(body="Content"))
        assert ctx is not None
        assert "test warning" in ctx.warnings

    def test_full_pipeline(self, tmp_path: Path) -> None:
        registry = HookRegistry()
        parser = Parser()
        register_builtin_hooks(registry)
        store = WikiStore(tmp_path, hooks=registry, parser=parser)

        doc = SourceDocument(url="https://example.com/report.pdf", title="Q4 Report", page=5)
        ctx = store.write(
            "HMRC Project",
            Page(body="The team delivered results. [@src_001]\nUsing [Python](rel:implemented_in)."),
            source_document=doc,
        )

        assert ctx is not None
        assert ctx.agent_context["citation_hint"] == "[@src_001]"

        page = store.read("HMRC Project")
        sources = page.frontmatter.get("sources", [])
        assert len(sources) == 1
        assert sources[0]["title"] == "Q4 Report"

        assert store.exists("Python")
