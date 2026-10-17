"""Marker for code that a later milestone implements.

Every unimplemented function in the skeleton calls :func:`todo` with the milestone
that delivers it, so a stub fails loudly and names where its spec lives.
"""

from __future__ import annotations

from typing import NoReturn


class NotImplementedYet(NotImplementedError):
    """Raised by skeleton stubs; ``milestone`` is the ROADMAP.md milestone (``"M1"`` ...)."""

    def __init__(self, milestone: str, what: str) -> None:
        self.milestone = milestone
        self.what = what
        super().__init__(
            f"[{milestone}] {what} is not implemented yet (see ROADMAP.md, milestone {milestone})"
        )


def todo(milestone: str, what: str) -> NoReturn:
    """Raise :class:`NotImplementedYet` for ``what``, delivered in ``milestone``."""
    raise NotImplementedYet(milestone, what)
