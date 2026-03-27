"""Custom backends for the Hyphex agent."""

from __future__ import annotations

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import EditResult, WriteResult


class ReadOnlyBackend(FilesystemBackend):
    """A ``FilesystemBackend`` that rejects all write and edit operations.

    All read operations (``read``, ``ls_info``, ``glob_info``, ``grep_raw``)
    pass through to the underlying filesystem. Writes and edits return an
    error result.

    Use with ``CompositeBackend`` to mount a read-only document directory
    alongside a writable wiki directory.

    Example:
        >>> from deepagents.backends import CompositeBackend, FilesystemBackend
        >>> backend = lambda rt: CompositeBackend(
        ...     default=FilesystemBackend(root_dir="./wiki", virtual_mode=True),
        ...     routes={"/docs/": ReadOnlyBackend(root_dir="./docs", virtual_mode=True)},
        ... )
    """

    def write(self, file_path: str, content: str) -> WriteResult:
        """Reject write operations.

        Args:
            file_path: Target file path.
            content: File content.

        Returns:
            A ``WriteResult`` with an error message.

        Example:
            >>> backend.write("/page.md", "content")
            WriteResult(error='Read-only: ...')
        """
        return WriteResult(error=f"Read-only: cannot write to {file_path}")

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Reject async write operations.

        Args:
            file_path: Target file path.
            content: File content.

        Returns:
            A ``WriteResult`` with an error message.

        Example:
            >>> await backend.awrite("/page.md", "content")
            WriteResult(error='Read-only: ...')
        """
        return WriteResult(error=f"Read-only: cannot write to {file_path}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Reject edit operations.

        Args:
            file_path: Target file path.
            old_string: Text to find.
            new_string: Replacement text.
            replace_all: Whether to replace all occurrences.

        Returns:
            An ``EditResult`` with an error message.

        Example:
            >>> backend.edit("/page.md", "old", "new")
            EditResult(error='Read-only: ...')
        """
        return EditResult(error=f"Read-only: cannot edit {file_path}")

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Reject async edit operations.

        Args:
            file_path: Target file path.
            old_string: Text to find.
            new_string: Replacement text.
            replace_all: Whether to replace all occurrences.

        Returns:
            An ``EditResult`` with an error message.

        Example:
            >>> await backend.aedit("/page.md", "old", "new")
            EditResult(error='Read-only: ...')
        """
        return EditResult(error=f"Read-only: cannot edit {file_path}")
