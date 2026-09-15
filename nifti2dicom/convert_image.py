"""Compatibility entry point for image conversion; see nifti2dicom.convert."""

from __future__ import annotations

import warnings
from pathlib import Path

from nifti2dicom.api import convert
from nifti2dicom.models import ConversionResult


def convert_nifti_to_dicom(
    ref_dir: str | Path,
    nifti_path: str | Path,
    output_dir: str | Path,
    *,
    vendor: str | None = None,
    series_description: str = "converted by nifti2dicom",
    header_dir: str | Path | None = None,
    force_overwrite: bool = False,
) -> ConversionResult:
    if vendor is not None:
        warnings.warn(
            "The vendor parameter is ignored; orientation comes from the NIfTI affine.",
            DeprecationWarning,
            stacklevel=2,
        )
    return convert(
        nifti_path,
        ref_dir,
        output_dir,
        kind="image",
        description=series_description,
        header_source=header_dir,
        overwrite=force_overwrite,
    )
