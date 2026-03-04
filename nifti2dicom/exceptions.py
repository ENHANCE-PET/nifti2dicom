"""Nifti2Dicom exception hierarchy — no silent failures."""

from __future__ import annotations


class Nifti2DicomError(Exception):
    """Base exception for all nifti2dicom errors."""


class ShapeMismatchError(Nifti2DicomError):
    """NIfTI data shape does not match DICOM reference series."""

    def __init__(self, expected: tuple[int, ...], got: tuple[int, ...]) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"Shape mismatch: expected {expected}, got {got}")


class NoDicomFilesError(Nifti2DicomError):
    """No valid DICOM files found in directory."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        super().__init__(f"No DICOM files found in {directory}")


class InvalidNiftiError(Nifti2DicomError):
    """NIfTI file could not be loaded or is invalid."""


class PixelEncodingError(Nifti2DicomError):
    """Pixel data could not be encoded to DICOM-compatible bytes."""
