"""Compatibility entry point for RGB conversion."""

from __future__ import annotations

from pathlib import Path

from nifti2dicom.api import convert
from nifti2dicom.models import ConversionResult


def convert_rgb_nifti_to_dicom(
    ref_dir: str | Path,
    nifti_path: str | Path,
    output_dir: str | Path,
) -> ConversionResult:
    return convert(nifti_path, ref_dir, output_dir, kind="rgb")
