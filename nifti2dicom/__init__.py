"""nifti2dicom — Convert NIfTI images to DICOM using a reference series."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nifti2dicom")
except PackageNotFoundError:
    __version__ = "2.0.0.dev0"

# Public API re-exports
from nifti2dicom.convert_image import convert_nifti_to_dicom
from nifti2dicom.convert_rgb import convert_rgb_nifti_to_dicom
from nifti2dicom.convert_seg import convert_nifti_seg_to_dicom
from nifti2dicom.exceptions import Nifti2DicomError, ShapeMismatchError

__all__ = [
    "__version__",
    "convert_nifti_to_dicom",
    "convert_rgb_nifti_to_dicom",
    "convert_nifti_seg_to_dicom",
    "Nifti2DicomError",
    "ShapeMismatchError",
]
