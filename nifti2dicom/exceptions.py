"""Compatibility names for the structured error hierarchy."""

from nifti2dicom.errors import ConversionError, InputError, PixelEncodingError, ReferenceError

Nifti2DicomError = ConversionError
InvalidNiftiError = InputError
__all__ = [
    "Nifti2DicomError",
    "InvalidNiftiError",
    "PixelEncodingError",
    "ShapeMismatchError",
    "NoDicomFilesError",
]


class ShapeMismatchError(ConversionError):
    """Historical shape error, retained for downstream imports."""

    code = "shape_mismatch"

    def __init__(self, expected: tuple[int, ...], got: tuple[int, ...]) -> None:
        self.expected, self.got = expected, got
        super().__init__(f"Shape mismatch: expected {expected}, got {got}")


class NoDicomFilesError(ReferenceError):
    def __init__(self, directory: str) -> None:
        self.directory = directory
        super().__init__(
            f"No DICOM files found in {directory}",
            hint="Choose a directory containing a DICOM image series.",
        )
