"""Validated input readers for the conversion pipeline."""

from nifti2dicom.readers.dicom import read_reference
from nifti2dicom.readers.nifti import read_nifti

__all__ = ["read_nifti", "read_reference"]
