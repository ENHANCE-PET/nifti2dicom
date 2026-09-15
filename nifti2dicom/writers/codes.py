"""Basic completeness checks for explicitly optional copied code sequences."""

from __future__ import annotations

from pydicom.dataset import Dataset


def _has_text(item: Dataset, keyword: str) -> bool:
    value = getattr(item, keyword, None)
    return isinstance(value, str) and bool(value.strip())


def _complete_basic_code(item: Dataset) -> bool:
    values = [name for name in ("CodeValue", "LongCodeValue", "URNCodeValue") if name in item]
    return (
        len(values) == 1
        and _has_text(item, values[0])
        and _has_text(item, "CodeMeaning")
        and (values[0] == "URNCodeValue" or _has_text(item, "CodingSchemeDesignator"))
    )


def omit_incomplete_optional_codes(dataset: Dataset, keywords: tuple[str, ...]) -> tuple[str, ...]:
    """Clean only named optional sequences on an output copy, returning warnings.

    The writer owns the allowlist and must never pass required sequences here.
    Empty sequences are retained; a present item needs a complete basic code
    (PS3.3 8.8). This is not terminology or full Code Sequence Macro validation.
    """
    warnings = []
    for keyword in keywords:
        sequence = getattr(dataset, keyword, None)
        if sequence is not None and any(not _complete_basic_code(item) for item in sequence):
            delattr(dataset, keyword)
            warnings.append(
                f"Omitted optional {keyword}: the reference contains an incomplete coded "
                "description. Source files are unchanged; no missing code was guessed."
            )
    return tuple(warnings)
