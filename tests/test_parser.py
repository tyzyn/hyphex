from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hyphex import Citation, Direction, EntityLink, FrontmatterError, Page, ParseError, Parser


@pytest.fixture
def parser() -> Parser:
    return Parser()


# ── Entity links ──────────────────────────────────────────────────────


class TestEntityLinks:
    def test_bare_link(self, parser: Parser) -> None:
        page = parser.parse("[[Machine Learning]]")
        assert len(page.entity_links) == 1
        link = page.entity_links[0]
        assert link.target == "Machine Learning"
        assert link.direction is None
        assert link.relationship is None

    def test_outgoing_link(self, parser: Parser) -> None:
        page = parser.parse("[[Python>implemented_in]]")
        link = page.entity_links[0]
        assert link.target == "Python"
        assert link.direction == Direction.OUTGOING
        assert link.relationship == "implemented_in"

    def test_incoming_link(self, parser: Parser) -> None:
        page = parser.parse("[[Jane Smith<authored_by]]")
        link = page.entity_links[0]
        assert link.target == "Jane Smith"
        assert link.direction == Direction.INCOMING
        assert link.relationship == "authored_by"

    def test_link_with_properties(self, parser: Parser) -> None:
        page = parser.parse("[[HMRC>client|engagement_size=large,duration=18m]]")
        link = page.entity_links[0]
        assert link.target == "HMRC"
        assert link.direction == Direction.OUTGOING
        assert link.relationship == "client"
        assert link.properties == {"engagement_size": "large", "duration": "18m"}

    def test_multiple_links(self, parser: Parser) -> None:
        text = "We used [[Python>implemented_in]] and [[TensorFlow>uses_framework]]."
        page = parser.parse(text)
        assert len(page.entity_links) == 2
        assert page.entity_links[0].target == "Python"
        assert page.entity_links[1].target == "TensorFlow"

    def test_link_in_prose(self, parser: Parser) -> None:
        text = "The project was led by [[Alice<authored_by]] and funded by [[HMRC>client]]."
        page = parser.parse(text)
        assert len(page.entity_links) == 2


# ── Citations ─────────────────────────────────────────────────────────


class TestCitations:
    def test_single_citation(self, parser: Parser) -> None:
        page = parser.parse("The team worked across three workstreams. {{1}}")
        assert len(page.citations) == 1
        assert page.citations[0].source_id == 1

    def test_multiple_citations(self, parser: Parser) -> None:
        text = "First claim. {{1}} Second claim. {{2}}"
        page = parser.parse(text)
        assert len(page.citations) == 2
        assert page.citations[0].source_id == 1
        assert page.citations[1].source_id == 2

    def test_multidigit_citation(self, parser: Parser) -> None:
        page = parser.parse("Some fact. {{42}}")
        assert page.citations[0].source_id == 42


# ── Mixed inline ──────────────────────────────────────────────────────


class TestMixedInline:
    def test_links_and_citations(self, parser: Parser) -> None:
        text = "[[Cabinet Office<referenced_by]] was involved. {{1}}"
        page = parser.parse(text)
        assert len(page.entity_links) == 1
        assert len(page.citations) == 1
        assert page.entity_links[0].target == "Cabinet Office"
        assert page.citations[0].source_id == 1

    def test_plain_text_only(self, parser: Parser) -> None:
        page = parser.parse("Just plain text, no links or citations.")
        assert page.entity_links == []
        assert page.citations == []

    def test_empty_body(self, parser: Parser) -> None:
        page = parser.parse("")
        assert page.entity_links == []
        assert page.citations == []


# ── Frontmatter ───────────────────────────────────────────────────────


class TestFrontmatter:
    def test_basic_frontmatter(self, parser: Parser) -> None:
        text = "---\ntype: person\n---\nHello world."
        page = parser.parse(text)
        assert page.frontmatter["type"] == "person"
        assert page.body == "Hello world."

    def test_no_frontmatter(self, parser: Parser) -> None:
        page = parser.parse("Just body content.")
        assert page.frontmatter == {}
        assert page.body == "Just body content."

    def test_frontmatter_with_sources(self, parser: Parser) -> None:
        text = (
            "---\n"
            "type: case_study\n"
            "sources:\n"
            "  - id: 1\n"
            '    title: "Test Doc"\n'
            '    url: "https://example.com/doc.pdf"\n'
            "    page: 14\n"
            "---\n"
            "Content here. {{1}}"
        )
        page = parser.parse(text)
        assert len(page.sources) == 1
        assert page.sources[0].id == 1
        assert page.sources[0].title == "Test Doc"
        assert page.sources[0].url == "https://example.com/doc.pdf"
        assert page.sources[0].page == 14

    def test_frontmatter_with_entity_link_fields(self, parser: Parser) -> None:
        text = '---\ntype: project\nclient: "[[HMRC]]"\n---\nBody.'
        page = parser.parse(text)
        assert page.frontmatter["client"] == "[[HMRC]]"

    def test_unterminated_frontmatter(self, parser: Parser) -> None:
        with pytest.raises(FrontmatterError, match="missing closing"):
            parser.parse("---\ntype: person\nNo closing fence")

    def test_invalid_yaml(self, parser: Parser) -> None:
        with pytest.raises(FrontmatterError, match="Invalid YAML"):
            parser.parse("---\n: : : bad\n---\n")

    def test_non_mapping_frontmatter(self, parser: Parser) -> None:
        with pytest.raises(FrontmatterError, match="must be a YAML mapping"):
            parser.parse("---\n- item1\n- item2\n---\n")

    def test_empty_frontmatter(self, parser: Parser) -> None:
        page = parser.parse("---\n---\nBody text.")
        assert page.frontmatter == {}
        assert page.body == "Body text."


# ── Full page round-trip ──────────────────────────────────────────────


class TestFullPage:
    def test_spec_example(self, parser: Parser) -> None:
        text = (
            "---\n"
            "type: case_study\n"
            "sources:\n"
            "  - id: 1\n"
            '    url: "https://drive.google.com/file/d/1abc2def3ghi4jkl"\n'
            '    title: "HMRC Digital Tax Platform - Final Bid Response"\n'
            "    page: 14\n"
            "  - id: 2\n"
            '    url: "https://example.com/hmrc-case-study.pdf"\n'
            '    title: "HMRC Case Study - Capability Deck v3"\n'
            "    page: 3\n"
            "---\n"
            "The team of 12 worked across three workstreams. {{1}}\n"
            "\n"
            "Key outcomes included a 40% reduction in processing time. {{2}}\n"
            "\n"
            "The project was referenced as a success by\n"
            "[[Cabinet Office<referenced_by]]. {{1}}\n"
        )
        page = parser.parse(text)
        assert page.frontmatter["type"] == "case_study"
        assert len(page.sources) == 2
        assert len(page.entity_links) == 1
        assert page.entity_links[0].target == "Cabinet Office"
        assert page.entity_links[0].direction == Direction.INCOMING
        assert len(page.citations) == 3

    def test_page_model_fields(self, parser: Parser) -> None:
        page = parser.parse("---\ntype: person\n---\n[[Alice>knows]]")
        assert isinstance(page, Page)
        assert isinstance(page.entity_links[0], EntityLink)


# ── parse_inline ──────────────────────────────────────────────────────


class TestParseInline:
    def test_parse_inline_links(self, parser: Parser) -> None:
        links, cites = parser.parse_inline("[[Python>uses]] and {{3}}")
        assert len(links) == 1
        assert links[0].relationship == "uses"
        assert len(cites) == 1
        assert cites[0].source_id == 3

    def test_parse_inline_empty(self, parser: Parser) -> None:
        links, cites = parser.parse_inline("no special syntax")
        assert links == []
        assert cites == []


# ── resolve_defaults ──────────────────────────────────────────────────


class TestResolveDefaults:
    def test_bare_link_gets_default(self, parser: Parser) -> None:
        page = parser.parse("[[Machine Learning]]")
        assert page.entity_links[0].relationship is None
        assert page.entity_links[0].is_default is False

        resolved = page.resolve_defaults()
        assert resolved.entity_links[0].relationship == "relates_to"
        assert resolved.entity_links[0].is_default is True

    def test_explicit_link_unchanged(self, parser: Parser) -> None:
        page = parser.parse("[[Python>implemented_in]]")
        resolved = page.resolve_defaults()
        assert resolved.entity_links[0].relationship == "implemented_in"
        assert resolved.entity_links[0].is_default is False

    def test_custom_default_relationship(self, parser: Parser) -> None:
        page = parser.parse("[[Blah]]")
        resolved = page.resolve_defaults(default_relationship="part_of")
        assert resolved.entity_links[0].relationship == "part_of"
        assert resolved.entity_links[0].is_default is True

    def test_mixed_links(self, parser: Parser) -> None:
        page = parser.parse("[[Bare]] and [[Explicit>knows]]")
        resolved = page.resolve_defaults()
        assert resolved.entity_links[0].relationship == "relates_to"
        assert resolved.entity_links[0].is_default is True
        assert resolved.entity_links[1].relationship == "knows"
        assert resolved.entity_links[1].is_default is False

    def test_does_not_mutate_original(self, parser: Parser) -> None:
        page = parser.parse("[[Bare]]")
        page.resolve_defaults()
        assert page.entity_links[0].relationship is None
        assert page.entity_links[0].is_default is False

    def test_no_links(self, parser: Parser) -> None:
        page = parser.parse("Just text.")
        resolved = page.resolve_defaults()
        assert resolved.entity_links == []


# ── Hypothesis property tests ─────────────────────────────────────────


_target_strategy = st.text(
    alphabet=st.characters(blacklist_characters="][><|\n"),
    min_size=1,
    max_size=50,
).map(str.strip).filter(lambda s: len(s) > 0)

_relationship_strategy = st.from_regex(r"[a-z_]{1,20}", fullmatch=True)

_source_id_strategy = st.integers(min_value=1, max_value=9999)


class TestPropertyBased:
    @given(target=_target_strategy)
    def test_bare_link_roundtrip(self, target: str) -> None:
        parser = Parser()
        text = f"[[{target}]]"
        page = parser.parse(text)
        assert len(page.entity_links) == 1
        assert page.entity_links[0].target == target

    @given(target=_target_strategy, rel=_relationship_strategy)
    def test_outgoing_link_roundtrip(self, target: str, rel: str) -> None:
        parser = Parser()
        text = f"[[{target}>{rel}]]"
        page = parser.parse(text)
        assert len(page.entity_links) == 1
        assert page.entity_links[0].target == target
        assert page.entity_links[0].relationship == rel
        assert page.entity_links[0].direction == Direction.OUTGOING

    @given(source_id=_source_id_strategy)
    def test_citation_roundtrip(self, source_id: int) -> None:
        parser = Parser()
        text = f"{{{{{source_id}}}}}"
        page = parser.parse(text)
        assert len(page.citations) == 1
        assert page.citations[0].source_id == source_id
