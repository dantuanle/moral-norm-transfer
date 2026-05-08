"""Strict parsing for model action outputs."""

from __future__ import annotations


_PREFIXES = ("Answer:", "Final:", "FINAL:")
_QUOTES = {"'", '"'}


def _strip_matching_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in _QUOTES:
        return text[1:-1].strip()
    return text


def parse_choice(text: str, valid_choices: list[str]) -> str | None:
    """Return a valid choice only when the output is a short action answer.

    The parser intentionally accepts a small set of answer-like wrappers while
    rejecting sentences or explanations that merely mention a valid action.
    """
    if not valid_choices:
        raise ValueError("valid_choices must be non-empty")

    candidate = text.strip()

    for prefix in _PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
            break

    candidate = _strip_matching_quotes(candidate)

    if candidate.endswith("."):
        candidate = candidate[:-1].strip()
        candidate = _strip_matching_quotes(candidate)

    if candidate in valid_choices:
        return candidate

    return None