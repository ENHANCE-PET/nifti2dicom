"""Compatibility DICOM helpers; validated series discovery lives in readers."""

from __future__ import annotations

from pathlib import Path

import pydicom

from nifti2dicom.errors import ReferenceError
from nifti2dicom.exceptions import NoDicomFilesError
from nifti2dicom.readers.dicom import read_reference


def is_dicom_file(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_file() or candidate.name.startswith("."):
        return False
    try:
        ds = pydicom.dcmread(candidate, stop_before_pixels=True, force=True)
        return bool(getattr(ds, "SOPClassUID", None) and getattr(ds, "SOPInstanceUID", None))
    except (OSError, ValueError, EOFError):
        return False


def load_dicom_series(directory: str | Path) -> tuple[list[pydicom.Dataset], list[str]]:
    """Load one coherent series in physical frame order, including pixel data."""
    try:
        series = read_reference(directory)
    except ReferenceError as exc:
        if exc.message == "No DICOM reference series was found.":
            raise NoDicomFilesError(str(directory)) from exc
        raise
    # Discovery has already validated identity and geometry, including datasets
    # without a Part 10 preamble. Preserve that compatibility when loading pixels.
    return [pydicom.dcmread(p, force=True) for p in series.paths], [str(p) for p in series.paths]


def is_dicom_compressed(ds: pydicom.Dataset) -> bool:
    syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if syntax is None:
        return False
    try:
        return bool(pydicom.uid.UID(str(syntax)).is_compressed)
    except ValueError:
        return False
