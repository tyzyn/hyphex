"""Filesystem-backed wiki store for Hyphex pages."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyphex._frontmatter import (
    parse_frontmatter,
    parse_sources,
    serialize_frontmatter,
    split_frontmatter,
)
from hyphex.exceptions import FrontmatterError, StoreError
from hyphex.models import Page

logger = logging.getLogger(__name__)

_RE_WHITESPACE_HYPHEN = re.compile(r"[\s\-]+")
_RE_NON_WORD = re.compile(r"[^a-z0-9_]")
_RE_MULTI_UNDERSCORE = re.compile(r"_+")


def slugify(title: str) -> str:
    """Convert a page title to a filesystem-safe slug.

    Args:
        title: Human-readable page title.

    Returns:
        A lowercase, underscore-separated slug suitable for use as a
        filename (without extension).

    Raises:
        StoreError: If the title produces an empty slug.

    Example:
        >>> slugify("HMRC Digital Tax Platform")
        'hmrc_digital_tax_platform'
    """
    normalized = unicodedata.normalize("NFKD", title)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = _RE_WHITESPACE_HYPHEN.sub("_", ascii_only.lower())
    cleaned = _RE_NON_WORD.sub("", lowered)
    slug = _RE_MULTI_UNDERSCORE.sub("_", cleaned).strip("_")
    if not slug:
        raise StoreError(f"Title {title!r} produces an empty slug")
    return slug


class WikiStore:
    """Filesystem-backed store for Hyphex wiki pages.

    Manages ``.md`` files with YAML frontmatter in a root directory.
    Does **not** parse Hyphex inline syntax (entity links, citations) —
    that is the ``Parser``'s responsibility.

    Not thread-safe. Suitable for single-agent or sequential-agent usage.

    Args:
        root: Path to the wiki directory. Created if it does not exist.
        created_by: Identifier for the auto-populated ``created_by`` field.

    Example:
        >>> store = WikiStore(Path("/tmp/wiki"))
        >>> store.write("HMRC", Page(frontmatter={"type": "organisation"}, body="Content."))
        >>> page = store.read("HMRC")
        >>> page.frontmatter["type"]
        'organisation'
    """

    def __init__(self, root: Path, created_by: str = "unknown") -> None:
        self._root = root
        self._created_by = created_by
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, title: str) -> Path:
        """Return the filesystem path for a given page title.

        Args:
            title: Human-readable page title.

        Returns:
            Absolute path to the ``.md`` file.

        Example:
            >>> store = WikiStore(Path("/tmp/wiki"))
            >>> store.path_for("Machine Learning")
            PosixPath('/tmp/wiki/machine_learning.md')
        """
        return self._root / f"{slugify(title)}.md"

    def exists(self, title: str) -> bool:
        """Check whether a page exists on disk.

        Args:
            title: Human-readable page title.

        Returns:
            ``True`` if the ``.md`` file exists.

        Example:
            >>> store = WikiStore(Path("/tmp/wiki"))
            >>> store.exists("Nonexistent")
            False
        """
        return self.path_for(title).is_file()

    def read(self, title: str) -> Page:
        """Read a page from disk.

        Returns a ``Page`` with ``frontmatter``, ``body``, and ``sources``
        populated. ``entity_links`` and ``citations`` are left empty — use
        ``Parser`` to extract those from the body.

        Args:
            title: Human-readable page title.

        Returns:
            The parsed page.

        Raises:
            StoreError: If the page does not exist or cannot be read.

        Example:
            >>> store = WikiStore(Path("/tmp/wiki"))
            >>> page = store.read("HMRC")
        """
        path = self.path_for(title)
        if not path.is_file():
            raise StoreError(f"Page {title!r} does not exist at {path}")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StoreError(f"Failed to read {path}: {exc}") from exc

        try:
            fm_yaml, body = split_frontmatter(text)
            frontmatter = parse_frontmatter(fm_yaml)
        except FrontmatterError as exc:
            raise StoreError(f"Failed to parse frontmatter in {path}: {exc}") from exc

        sources = parse_sources(frontmatter)
        return Page(frontmatter=frontmatter, body=body, sources=sources)

    def write(self, title: str, page: Page, *, overwrite: bool = True) -> None:
        """Write a page to disk.

        Auto-populates ``title``, ``modified``, and (on first write)
        ``created`` and ``created_by`` in the frontmatter.

        Args:
            title: Human-readable page title.
            page: The page to write.
            overwrite: If ``False``, raise ``StoreError`` when the file
                already exists.

        Raises:
            StoreError: If ``overwrite`` is ``False`` and the page exists,
                or if the write fails.

        Example:
            >>> store = WikiStore(Path("/tmp/wiki"))
            >>> store.write("ML", Page(frontmatter={"type": "capability"}, body="Content."))
        """
        path = self.path_for(title)

        fm = dict(page.frontmatter)
        fm["title"] = title
        now = datetime.now(timezone.utc).isoformat()
        fm["modified"] = now

        # Preserve created/created_by from existing file or set on first write.
        existing_fm: dict[str, Any] | None = None
        if path.is_file():
            if not overwrite:
                raise StoreError(f"Page {title!r} already exists and overwrite=False")
            try:
                existing_yaml, _ = split_frontmatter(path.read_text(encoding="utf-8"))
                existing_fm = parse_frontmatter(existing_yaml)
            except (FrontmatterError, OSError):
                existing_fm = None

        if existing_fm is not None:
            fm.setdefault("created", existing_fm.get("created", now))
            fm.setdefault("created_by", existing_fm.get("created_by", self._created_by))
        else:
            fm.setdefault("created", now)
            fm.setdefault("created_by", self._created_by)

        # Ensure sources from the Page model are in frontmatter for roundtrip.
        if page.sources and "sources" not in fm:
            fm["sources"] = [s.model_dump(exclude_none=True) for s in page.sources]

        content = serialize_frontmatter(fm, page.body)
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise StoreError(f"Failed to write {path}: {exc}") from exc

    def create_stub(
        self,
        title: str,
        *,
        page_type: str | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Create a minimal stub page.

        No-op if the page already exists — stubs never overwrite real content.

        Args:
            title: Human-readable page title.
            page_type: Optional entity type for the stub.
            aliases: Optional list of alternative names.

        Example:
            >>> store = WikiStore(Path("/tmp/wiki"))
            >>> store.create_stub("NHS", page_type="organisation", aliases=["National Health Service"])
        """
        if self.exists(title):
            logger.debug("Stub not created: page %r already exists", title)
            return

        fm: dict[str, Any] = {}
        if page_type is not None:
            fm["type"] = page_type
        fm["aliases"] = aliases or []

        self.write(title, Page(frontmatter=fm, body=""))

    def list_pages(self, *, page_type: str | None = None) -> list[str]:
        """List page titles, optionally filtered by type.

        Args:
            page_type: If provided, return only pages with this ``type``
                in frontmatter.

        Returns:
            List of page title strings.

        Example:
            >>> store = WikiStore(Path("/tmp/wiki"))
            >>> store.list_pages(page_type="organisation")
            ['HMRC', 'Cabinet Office']
        """
        titles: list[str] = []
        for _, fm, _ in self._iter_pages():
            page_title = fm.get("title", "")
            if page_type is not None and fm.get("type") != page_type:
                continue
            titles.append(page_title)
        return titles

    def grep(self, query: str) -> list[str]:
        """Search pages by title, aliases, and body content.

        Performs case-insensitive substring matching across all ``.md``
        files in the wiki.

        Args:
            query: The search string.

        Returns:
            List of matching page titles.

        Example:
            >>> store = WikiStore(Path("/tmp/wiki"))
            >>> store.grep("NHS")
            ['National Health Service']
        """
        query_lower = query.lower()
        matches: list[str] = []
        for _, fm, body in self._iter_pages():
            page_title: str = fm.get("title", "")
            aliases: list[str] = fm.get("aliases", [])

            if query_lower in page_title.lower():
                matches.append(page_title)
            elif any(query_lower in alias.lower() for alias in aliases):
                matches.append(page_title)
            elif query_lower in body.lower():
                matches.append(page_title)

        return matches

    def _iter_pages(self) -> Iterator[tuple[Path, dict[str, Any], str]]:
        """Yield ``(path, frontmatter, body)`` for each readable page.

        Args: None.

        Returns:
            An iterator of ``(path, frontmatter_dict, body_str)`` tuples.

        Example:
            >>> store = WikiStore(Path("/tmp/wiki"))
            >>> for path, fm, body in store._iter_pages():
            ...     print(fm.get("title"))
        """
        for path in sorted(self._root.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                fm_yaml, body = split_frontmatter(text)
                fm = parse_frontmatter(fm_yaml)
            except (FrontmatterError, OSError):
                logger.warning("Skipping unreadable page: %s", path)
                continue
            yield (path, fm, body)
