from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hyphex import Page, Source, StoreError, WikiStore, slugify


# ── slugify ───────────────────────────────────────────────────────────


class TestSlugify:
    def test_basic_title(self) -> None:
        assert slugify("HMRC Digital Tax Platform") == "hmrc_digital_tax_platform"

    def test_unicode_accents(self) -> None:
        assert slugify("Caf\u00e9 Ren\u00e9e") == "cafe_renee"

    def test_special_chars(self) -> None:
        assert slugify("C++ & C#") == "c_c"

    def test_hyphens(self) -> None:
        assert slugify("machine-learning") == "machine_learning"

    def test_collapse_underscores(self) -> None:
        assert slugify("too   many   spaces") == "too_many_spaces"

    def test_empty_raises(self) -> None:
        with pytest.raises(StoreError, match="empty slug"):
            slugify("")

    def test_only_special_chars_raises(self) -> None:
        with pytest.raises(StoreError, match="empty slug"):
            slugify("!@#$%")

    @given(
        title=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
            min_size=1,
            max_size=50,
        ).filter(lambda s: any(c.isascii() and c.isalnum() for c in s))
    )
    def test_slug_is_valid_filename(self, title: str) -> None:
        slug = slugify(title)
        assert slug
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in slug)
        assert not slug.startswith("_")
        assert not slug.endswith("_")


# ── WikiStore fixtures ────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path, created_by="test_agent")


@pytest.fixture
def sample_page() -> Page:
    return Page(
        frontmatter={"type": "organisation"},
        body="HMRC is a UK government department.",
    )


# ── exists ────────────────────────────────────────────────────────────


class TestExists:
    def test_false_before_write(self, store: WikiStore) -> None:
        assert store.exists("HMRC") is False

    def test_true_after_write(self, store: WikiStore, sample_page: Page) -> None:
        store.write("HMRC", sample_page)
        assert store.exists("HMRC") is True


# ── path_for ──────────────────────────────────────────────────────────


class TestPathFor:
    def test_returns_md_path(self, store: WikiStore) -> None:
        path = store.path_for("Machine Learning")
        assert path.name == "machine_learning.md"
        assert path.parent == store._root


# ── read / write ──────────────────────────────────────────────────────


class TestReadWrite:
    def test_roundtrip(self, store: WikiStore, sample_page: Page) -> None:
        store.write("HMRC", sample_page)
        page = store.read("HMRC")
        assert page.frontmatter["type"] == "organisation"
        assert page.body == "HMRC is a UK government department."

    def test_read_nonexistent_raises(self, store: WikiStore) -> None:
        with pytest.raises(StoreError, match="does not exist"):
            store.read("Nonexistent")

    def test_write_sets_title(self, store: WikiStore, sample_page: Page) -> None:
        store.write("HMRC", sample_page)
        page = store.read("HMRC")
        assert page.frontmatter["title"] == "HMRC"

    def test_write_sets_timestamps(self, store: WikiStore, sample_page: Page) -> None:
        store.write("HMRC", sample_page)
        page = store.read("HMRC")
        assert "created" in page.frontmatter
        assert "modified" in page.frontmatter
        assert "created_by" in page.frontmatter
        assert page.frontmatter["created_by"] == "test_agent"

    def test_write_preserves_created(self, store: WikiStore, sample_page: Page) -> None:
        store.write("HMRC", sample_page)
        first = store.read("HMRC")
        created = first.frontmatter["created"]

        store.write("HMRC", Page(frontmatter={"type": "organisation"}, body="Updated."))
        second = store.read("HMRC")
        assert second.frontmatter["created"] == created
        assert second.body == "Updated."

    def test_write_overwrite_false_raises(self, store: WikiStore, sample_page: Page) -> None:
        store.write("HMRC", sample_page)
        with pytest.raises(StoreError, match="already exists"):
            store.write("HMRC", sample_page, overwrite=False)

    def test_write_with_sources(self, store: WikiStore) -> None:
        page = Page(
            frontmatter={"type": "case_study"},
            body="Content. {{1}}",
            sources=[Source(id=1, title="Doc", url="https://example.com")],
        )
        store.write("Project X", page)
        read_back = store.read("Project X")
        assert len(read_back.sources) == 1
        assert read_back.sources[0].title == "Doc"


# ── create_stub ───────────────────────────────────────────────────────


class TestCreateStub:
    def test_creates_file(self, store: WikiStore) -> None:
        store.create_stub("NHS", page_type="organisation")
        assert store.exists("NHS")

    def test_stub_frontmatter(self, store: WikiStore) -> None:
        store.create_stub("NHS", page_type="organisation")
        page = store.read("NHS")
        assert page.frontmatter["type"] == "organisation"
        assert page.frontmatter["aliases"] == []
        assert page.frontmatter["title"] == "NHS"
        assert "created" in page.frontmatter
        assert page.body == ""

    def test_stub_noop_if_exists(self, store: WikiStore, sample_page: Page) -> None:
        store.write("NHS", sample_page)
        original_body = store.read("NHS").body
        store.create_stub("NHS", page_type="organisation")
        assert store.read("NHS").body == original_body

    def test_stub_with_aliases(self, store: WikiStore) -> None:
        store.create_stub("NHS", page_type="organisation", aliases=["National Health Service", "NHSE"])
        page = store.read("NHS")
        assert page.frontmatter["aliases"] == ["National Health Service", "NHSE"]


# ── list_pages ────────────────────────────────────────────────────────


class TestListPages:
    def test_list_all(self, store: WikiStore) -> None:
        store.create_stub("Alpha", page_type="person")
        store.create_stub("Beta", page_type="organisation")
        store.create_stub("Gamma", page_type="person")
        titles = store.list_pages()
        assert sorted(titles) == ["Alpha", "Beta", "Gamma"]

    def test_list_filtered_by_type(self, store: WikiStore) -> None:
        store.create_stub("Alpha", page_type="person")
        store.create_stub("Beta", page_type="organisation")
        store.create_stub("Gamma", page_type="person")
        titles = store.list_pages(page_type="person")
        assert sorted(titles) == ["Alpha", "Gamma"]

    def test_list_empty(self, store: WikiStore) -> None:
        assert store.list_pages() == []


# ── grep ──────────────────────────────────────────────────────────────


class TestGrep:
    def test_grep_title_match(self, store: WikiStore) -> None:
        store.create_stub("National Health Service", page_type="organisation")
        assert store.grep("Health") == ["National Health Service"]

    def test_grep_alias_match(self, store: WikiStore) -> None:
        store.create_stub(
            "National Health Service",
            page_type="organisation",
            aliases=["NHS", "NHSE"],
        )
        assert store.grep("NHS") == ["National Health Service"]

    def test_grep_body_match(self, store: WikiStore) -> None:
        store.write(
            "Report",
            Page(frontmatter={"type": "document"}, body="The NHS delivered results."),
        )
        assert store.grep("NHS") == ["Report"]

    def test_grep_case_insensitive(self, store: WikiStore) -> None:
        store.create_stub("NHS", page_type="organisation")
        assert store.grep("nhs") == ["NHS"]

    def test_grep_no_match(self, store: WikiStore) -> None:
        store.create_stub("HMRC", page_type="organisation")
        assert store.grep("nonexistent") == []

    def test_grep_multiple_matches(self, store: WikiStore) -> None:
        store.write("Page A", Page(frontmatter={}, body="Both mention NHS."))
        store.create_stub("NHS", page_type="organisation")
        results = store.grep("NHS")
        assert len(results) == 2
