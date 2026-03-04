"""Safe DICOM tag copying — two-pass to avoid mutating during iteration.

Fixes the old bug where iterating over a Dataset and deleting tags
simultaneously could skip entries or raise RuntimeError.
"""

from __future__ import annotations

import pydicom

# Tags that carry per-slice spatial/pixel info — never copy from header source
TAGS_TO_EXCLUDE = frozenset(
    {
        "Pixel Data",
        "Image Index",
        "Number of Slices",
        "Rows",
        "Columns",
        "Pixel Spacing",
        "Image Position (Patient)",
        "Image Orientation (Patient)",
        "Instance Number",
        "Slice Thickness",
        "Slice Location",
    }
)


def copy_tags(
    target: pydicom.Dataset,
    source: pydicom.Dataset,
    exclude: frozenset[str] = TAGS_TO_EXCLUDE,
) -> None:
    """Copy non-excluded tags from *source* into *target*.

    Uses a two-pass approach:

    1. **Collect** — iterate *target*, find tags absent from *source*.
    2. **Apply** — delete orphan tags, then copy values from *source*.

    This avoids the old dict-mutation-during-iteration bug.
    """
    # Pass 1: collect tags in target that are not in source
    tags_to_delete = [
        tag.tag for tag in target if tag.tag not in source
    ]

    # Delete orphan tags
    for tag_key in tags_to_delete:
        del target[tag_key]

    # Pass 2: copy values from source, skipping excluded names
    for elem in source:
        if elem.name in exclude:
            continue
        target[elem.tag] = elem
