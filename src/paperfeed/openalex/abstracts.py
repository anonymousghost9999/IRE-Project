"""Abstract reconstruction utilities for OpenAlex inverted index format (Phase 1.5)."""

from typing import Any, Dict, Optional


def reconstruct_abstract(abstract_inverted_index: Optional[Dict[str, list[int]]]) -> str:
    """
    Reconstruct human-readable abstract string from OpenAlex abstract_inverted_index.

    OpenAlex stores abstracts as a dictionary mapping each word to a list of integer
    0-based token positions, e.g.:
        {"We": [0], "propose": [1], "a": [2, 5], "novel": [3], "method": [4], "for...": [6]}

    Returns:
        Clean plain-text string, or empty string "" if inverted index is None or empty.
    """
    if not abstract_inverted_index:
        return ""

    positions: list[tuple[int, str]] = []
    for word, pos_list in abstract_inverted_index.items():
        if not isinstance(word, str) or not isinstance(pos_list, (list, tuple)):
            continue
        for pos in pos_list:
            if isinstance(pos, int) and pos >= 0:
                positions.append((pos, word))

    if not positions:
        return ""

    positions.sort(key=lambda x: x[0])
    tokens = [word for _, word in positions]
    return " ".join(tokens).strip()


def has_valid_abstract(abstract_inverted_index: Optional[Dict[str, list[int]]]) -> bool:
    """Check whether a paper record has a non-empty reconstructed abstract."""
    if not abstract_inverted_index:
        return False
    return any(isinstance(v, (list, tuple)) and len(v) > 0 for v in abstract_inverted_index.values())

