"""Document preprocessing for the Hyphex note-taking agent.

Converts PDFs into per-page markdown files that the agent can explore
via its filesystem tools. Each document gets an auto-assigned ``src_NNN``
identifier to avoid naming collisions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pymupdf
import yaml

logger = logging.getLogger(__name__)

_RE_SRC_ID = re.compile(r"^src_(\d+)$")


def _next_src_id(docs_dir: Path) -> str:
    """Scan ``docs/`` for existing ``src_NNN`` directories and return the next ID.

    Args:
        docs_dir: The ``workspace/docs/`` directory.

    Returns:
        The next available source ID, e.g. ``"src_003"``.

    Example:
        >>> _next_src_id(Path("workspace/docs"))  # contains src_001, src_002
        'src_003'
    """
    max_n = 0
    if docs_dir.is_dir():
        for child in docs_dir.iterdir():
            if child.is_dir():
                match = _RE_SRC_ID.match(child.name)
                if match:
                    max_n = max(max_n, int(match.group(1)))
    return f"src_{max_n + 1:03d}"


def prepare_document(
    pdf_path: str | Path,
    workspace_dir: str | Path,
    *,
    doc_id: str | None = None,
) -> str:
    """Convert a PDF into per-page markdown files for agent exploration.

    Creates a directory at ``workspace_dir/docs/{src_id}/`` containing one
    markdown file per PDF page and a ``_meta.yml`` metadata file. If
    ``doc_id`` is not provided, auto-assigns the next ``src_NNN`` identifier.

    Args:
        pdf_path: Path to the source PDF.
        workspace_dir: Root workspace directory (parent of ``wiki/`` and ``docs/``).
        doc_id: Explicit identifier for this document. If ``None``,
            auto-assigns the next ``src_NNN`` ID.

    Returns:
        The ``doc_id`` used (auto-generated or provided).

    Raises:
        FileNotFoundError: If the PDF does not exist.

    Example:
        >>> doc_id = prepare_document("report.pdf", "./workspace")
        >>> doc_id
        'src_001'
    """
    pdf_path = Path(pdf_path)
    workspace_dir = Path(workspace_dir)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs_dir = workspace_dir / "docs"
    if doc_id is None:
        doc_id = _next_src_id(docs_dir)

    doc_dir = docs_dir / doc_id
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
        "src_id": doc_id,
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
        "Prepared %s as %s: %d pages (%d with text) → %s",
        pdf_path.name,
        doc_id,
        total_pages,
        written_pages,
        doc_dir,
    )
    return doc_id
