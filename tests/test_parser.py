"""Tests for the Hyphex markdown parser.

Covers entity links [Target](rel:type), citations [@src_NNN, p. N],
evidence blocks (::: evidence ... :::), frontmatter parsing, and
the full page round-trip.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hyphex.models import Citation, EntityLink, EvidenceBlock, Page
from hyphex.parser import Parser


@pytest.fixture()
def parser():
    return Parser()


# ── Entity Links ─────────────────────────────────────────────────────


class TestEntityLinks:
    def test_basic_link(self, parser):
        links, _, _ = parser.parse_inline("[Python](rel:has_skill)")
        assert len(links) == 1
        assert links[0].target == "Python"
        assert links[0].relationship == "has_skill"

    def test_multiple_links(self, parser):
        text = "[Elsewhen](rel:employed_by) and [HMRC](rel:client) are linked."
        links, _, _ = parser.parse_inline(text)
        assert len(links) == 2
        assert links[0].target == "Elsewhen"
        assert links[0].relationship == "employed_by"
        assert links[1].target == "HMRC"
        assert links[1].relationship == "client"

    def test_link_with_relates_to(self, parser):
        links, _, _ = parser.parse_inline("[Market Research Society](rel:relates_to)")
        assert links[0].relationship == "relates_to"

    def test_link_in_prose(self, parser):
        text = "He works at [Elsewhen](rel:employed_by) as a researcher."
        links, _, _ = parser.parse_inline(text)
        assert len(links) == 1
        assert links[0].target == "Elsewhen"

    def test_link_with_spaces_in_target(self, parser):
        links, _, _ = parser.parse_inline("[Machine Learning](rel:has_skill)")
        assert links[0].target == "Machine Learning"

    def test_regular_markdown_link_ignored(self, parser):
        links, _, _ = parser.parse_inline("See [this](https://example.com) here.")
        assert len(links) == 0

    def test_link_target_with_special_chars(self, parser):
        links, _, _ = parser.parse_inline("[HMRC — Digital Tax](rel:client)")
        assert links[0].target == "HMRC — Digital Tax"


# ── Citations ────────────────────────────────────────────────────────


class TestCitations:
    def test_citation_with_page(self, parser):
        _, cites, _ = parser.parse_inline("Fact. [@src_001, p. 5]")
        assert len(cites) == 1
        assert cites[0].source_id == "src_001"
        assert cites[0].page_ref == "p. 5"

    def test_citation_without_page(self, parser):
        _, cites, _ = parser.parse_inline("Fact. [@src_001]")
        assert len(cites) == 1
        assert cites[0].source_id == "src_001"
        assert cites[0].page_ref is None

    def test_citation_page_range(self, parser):
        _, cites, _ = parser.parse_inline("Fact. [@src_001, pp. 12-15]")
        assert cites[0].page_ref == "pp. 12-15"

    def test_multiple_citations(self, parser):
        _, cites, _ = parser.parse_inline("A [@src_001, p. 1] and B [@src_002, p. 3]")
        assert len(cites) == 2
        assert cites[0].source_id == "src_001"
        assert cites[1].source_id == "src_002"

    def test_high_numbered_source(self, parser):
        _, cites, _ = parser.parse_inline("[@src_042, p. 99]")
        assert cites[0].source_id == "src_042"


# ── Evidence Blocks ──────────────────────────────────────────────────


class TestEvidenceBlocks:
    def test_basic_evidence(self, parser):
        text = "Claim.\n::: evidence\nVerbatim text.\n:::"
        _, _, evidence = parser.parse_inline(text)
        assert len(evidence) == 1
        assert evidence[0].content == "Verbatim text."

    def test_multiline_evidence(self, parser):
        text = "Claim.\n::: evidence\nLine one.\nLine two.\nLine three.\n:::"
        _, _, evidence = parser.parse_inline(text)
        assert "Line one." in evidence[0].content
        assert "Line three." in evidence[0].content

    def test_evidence_with_surrounding_text(self, parser):
        text = "Before.\n::: evidence\nQuoted text.\n:::\nAfter."
        _, _, evidence = parser.parse_inline(text)
        assert len(evidence) == 1
        assert evidence[0].content == "Quoted text."

    def test_multiple_evidence_blocks(self, parser):
        text = (
            "First claim.\n::: evidence\nFirst quote.\n:::\n"
            "Second claim.\n::: evidence\nSecond quote.\n:::"
        )
        _, _, evidence = parser.parse_inline(text)
        assert len(evidence) == 2


# ── Mixed Elements ───────────────────────────────────────────────────


class TestMixed:
    def test_links_and_citations(self, parser):
        text = "[Elsewhen](rel:employed_by) did great work. [@src_001, p. 3]"
        links, cites, _ = parser.parse_inline(text)
        assert len(links) == 1
        assert len(cites) == 1

    def test_all_three_elements(self, parser):
        text = (
            "[Elsewhen](rel:client) delivered. [@src_001, p. 1]\n"
            "::: evidence\nElsewhen was the delivery partner.\n:::"
        )
        links, cites, evidence = parser.parse_inline(text)
        assert len(links) == 1
        assert len(cites) == 1
        assert len(evidence) == 1

    def test_plain_text_only(self, parser):
        links, cites, evidence = parser.parse_inline("Just regular text.")
        assert links == []
        assert cites == []
        assert evidence == []

    def test_empty_string(self, parser):
        links, cites, evidence = parser.parse_inline("")
        assert links == []
        assert cites == []
        assert evidence == []


# ── Frontmatter ──────────────────────────────────────────────────────


class TestFrontmatter:
    def test_basic_frontmatter(self, parser):
        page = parser.parse("---\ntype: person\n---\nBody text.")
        assert page.frontmatter["type"] == "person"
        assert page.body == "Body text."

    def test_no_frontmatter(self, parser):
        page = parser.parse("[Python](rel:has_skill)")
        assert page.frontmatter == {}
        assert len(page.entity_links) == 1

    def test_frontmatter_with_sources(self, parser):
        text = (
            "---\ntype: person\nsources:\n"
            "  - id: src_001\n    title: Report\n    url: https://x.com\n"
            "---\nBody."
        )
        page = parser.parse(text)
        assert len(page.sources) == 1
        assert page.sources[0].id == "src_001"
        assert page.sources[0].title == "Report"

    def test_frontmatter_error(self, parser):
        from hyphex.exceptions import FrontmatterError

        with pytest.raises(FrontmatterError):
            parser.parse("---\n: invalid [\n---\nBody")


# ── Full Page ────────────────────────────────────────────────────────


class TestFullPage:
    def test_complete_page(self, parser):
        text = (
            "---\n"
            "type: person\n"
            "sources:\n"
            "  - id: src_001\n"
            "    title: Report\n"
            "    url: https://example.com\n"
            "---\n"
            "[Alice](rel:knows) works at [ACME](rel:employed_by). [@src_001, p. 1]\n"
            "\n"
            "::: evidence\n"
            "Alice has worked at ACME since 2020.\n"
            ":::\n"
        )
        page = parser.parse(text)
        assert page.frontmatter["type"] == "person"
        assert len(page.entity_links) == 2
        assert len(page.citations) == 1
        assert len(page.evidence_blocks) == 1
        assert len(page.sources) == 1

    def test_model_types(self, parser):
        page = parser.parse("[X](rel:knows) text. [@src_001]")
        assert isinstance(page, Page)
        assert isinstance(page.entity_links[0], EntityLink)
        assert isinstance(page.citations[0], Citation)


# ── Property-based tests ─────────────────────────────────────────────


_SAFE_CHARS = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs"),
        whitelist_characters=("-", "_"),
    ),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip())

_RELATIONSHIP = st.from_regex(r"[a-z_]{2,15}", fullmatch=True)


class TestPropertyBased:
    @given(target=_SAFE_CHARS, rel=_RELATIONSHIP)
    @settings(max_examples=50)
    def test_entity_link_roundtrip(self, target, rel):
        p = Parser()
        text = f"[{target}](rel:{rel})"
        links, _, _ = p.parse_inline(text)
        assert len(links) == 1
        assert links[0].target == target.strip()
        assert links[0].relationship == rel

    @given(num=st.integers(min_value=1, max_value=999))
    @settings(max_examples=30)
    def test_citation_roundtrip(self, num):
        p = Parser()
        src_id = f"src_{num:03d}"
        text = f"Claim. [@{src_id}, p. {num}]"
        _, cites, _ = p.parse_inline(text)
        assert len(cites) == 1
        assert cites[0].source_id == src_id
