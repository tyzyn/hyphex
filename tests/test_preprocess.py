"""Tests for hyphex.agents.preprocess — document preparation."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
import yaml


pymupdf_available = pytest.importorskip("pymupdf")

from hyphex.agents.preprocess import prepare_document  # noqa: E402


def _create_pdf(path: Path, pages: list[str]) -> None:
    """Create a simple PDF with text pages for testing."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


class TestPrepareDocument:
    def test_auto_assigns_src_001(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        _create_pdf(pdf, ["Page one content", "Page two content"])

        workspace = tmp_path / "workspace"
        doc_id = prepare_document(pdf, workspace)

        assert doc_id == "src_001"
        doc_dir = workspace / "docs" / "src_001"
        assert (doc_dir / "page_001.md").is_file()
        assert (doc_dir / "page_002.md").is_file()
        assert "Page one content" in (doc_dir / "page_001.md").read_text()

    def test_increments_src_id(self, tmp_path):
        workspace = tmp_path / "workspace"

        pdf1 = tmp_path / "first.pdf"
        _create_pdf(pdf1, ["Content"])
        id1 = prepare_document(pdf1, workspace)
        assert id1 == "src_001"

        pdf2 = tmp_path / "second.pdf"
        _create_pdf(pdf2, ["Content"])
        id2 = prepare_document(pdf2, workspace)
        assert id2 == "src_002"

    def test_meta_yml_has_src_id(self, tmp_path):
        pdf = tmp_path / "my_document.pdf"
        _create_pdf(pdf, ["Content"])

        workspace = tmp_path / "workspace"
        doc_id = prepare_document(pdf, workspace)

        meta_path = workspace / "docs" / doc_id / "_meta.yml"
        meta = yaml.safe_load(meta_path.read_text())
        assert meta["src_id"] == "src_001"
        assert meta["page_count"] == 1
        assert meta["pages_with_text"] == 1
        assert meta["url"] == str(pdf.resolve())

    def test_custom_doc_id(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        _create_pdf(pdf, ["Content"])

        workspace = tmp_path / "workspace"
        doc_id = prepare_document(pdf, workspace, doc_id="custom_id")

        assert doc_id == "custom_id"
        assert (workspace / "docs" / "custom_id" / "page_001.md").is_file()

    def test_skips_empty_pages(self, tmp_path):
        pdf = tmp_path / "sparse.pdf"
        doc = pymupdf.open()
        page1 = doc.new_page()
        page1.insert_text((72, 72), "Has content")
        doc.new_page()  # empty page
        page3 = doc.new_page()
        page3.insert_text((72, 72), "Also has content")
        doc.save(str(pdf))
        doc.close()

        workspace = tmp_path / "workspace"
        doc_id = prepare_document(pdf, workspace)

        doc_dir = workspace / "docs" / doc_id
        meta = yaml.safe_load((doc_dir / "_meta.yml").read_text())
        assert meta["page_count"] == 3
        assert meta["pages_with_text"] == 2
        assert (doc_dir / "page_001.md").is_file()
        assert not (doc_dir / "page_002.md").is_file()
        assert (doc_dir / "page_003.md").is_file()

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare_document(tmp_path / "nonexistent.pdf", tmp_path / "workspace")

    def test_zero_padded_filenames(self, tmp_path):
        pdf = tmp_path / "long.pdf"
        _create_pdf(pdf, [f"Page {i}" for i in range(1, 15)])

        workspace = tmp_path / "workspace"
        doc_id = prepare_document(pdf, workspace)

        doc_dir = workspace / "docs" / doc_id
        assert (doc_dir / "page_001.md").is_file()
        assert (doc_dir / "page_014.md").is_file()
