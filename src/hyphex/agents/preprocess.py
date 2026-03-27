"""Document preprocessing for the Hyphex note-taking agent.

Converts PDFs into per-page markdown files that the agent can explore
via its filesystem tools.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from hyphex.store import slugify

logger = logging.getLogger(__name__)


def prepare_document(
    pdf_path: str | Path,
    workspace_dir: str | Path,
    *,
    doc_id: str | None = None,
) -> str:
    """Convert a PDF into per-page markdown files for agent exploration.

    Creates a directory at ``workspace_dir/docs/{doc_id}/`` containing one
    markdown file per PDF page and a ``_meta.yml`` metadata file.

    Args:
        pdf_path: Path to the source PDF.
        workspace_dir: Root workspace directory (parent of ``wiki/`` and ``docs/``).
        doc_id: Identifier for this document. If ``None``, derived from the
            PDF filename via ``slugify()``.

    Returns:
        The ``doc_id`` used (auto-generated or provided).

    Raises:
        FileNotFoundError: If the PDF does not exist.

    Example:
        >>> doc_id = prepare_document("report.pdf", "./workspace")
        >>> doc_id
        'report'
    """
    pdf_path = Path(pdf_path)
    workspace_dir = Path(workspace_dir)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if doc_id is None:
        doc_id = slugify(pdf_path.stem)

    doc_dir = workspace_dir / "docs" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    written_pages = 0

    for page_num in range(total_pages):
        text = doc[page_num].get_text()
        if not text.strip():
            logger.debug("Skipping empty page %d of %s", page_num + 1, pdf_path.name)
            continue

        page_file = doc_dir / f"page_{page_num + 1:03d}.md"
        page_file.write_text(text, encoding="utf-8")
        written_pages += 1

    title = doc.metadata.get("title", "") or pdf_path.stem.replace("_", " ").replace("-", " ")
    doc.close()

    meta: dict[str, Any] = {
        "title": title,
        "url": str(pdf_path.resolve()),
        "page_count": total_pages,
        "pages_with_text": written_pages,
    }
    meta_path = doc_dir / "_meta.yml"
    meta_path.write_text(
        yaml.dump(meta, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    logger.info(
        "Prepared %s: %d pages (%d with text) → %s",
        pdf_path.name,
        total_pages,
        written_pages,
        doc_dir,
    )
    return doc_id
