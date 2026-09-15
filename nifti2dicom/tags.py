"""Compatibility metadata copying restricted to patient and study context."""

from __future__ import annotations

from copy import deepcopy

from pydicom.dataset import Dataset

from nifti2dicom.writers.image import _CONTEXT

TAGS_TO_EXCLUDE = frozenset(
    {
        "Pixel Data",
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


def copy_tags(target: Dataset, source: Dataset, exclude: frozenset[str] = TAGS_TO_EXCLUDE) -> None:
    """Copy allowed context without deleting pixels, geometry, scaling or UIDs."""
    for keyword in _CONTEXT:
        elem = source.data_element(keyword) if keyword in source else None
        if elem is not None and elem.name not in exclude:
            target.add(deepcopy(elem))
