"""Compatibility entry point for segmentation conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nifti2dicom.api import convert
from nifti2dicom.models import ConversionResult


def convert_nifti_seg_to_dicom(
    ref_dir: str | Path,
    nifti_path: str | Path,
    output_path: str | Path,
    organ_index: dict[str, Any],
    *,
    manufacturer: str = "nifti2dicom",
    manufacturer_model_name: str = "nifti2dicom",
    software_versions: str = "2.1",
    algorithm_type: str | None = None,
    algorithm_name: str | None = None,
    algorithm_version: str | None = None,
) -> ConversionResult:
    """Convert labels; provenance is required explicitly or in the label JSON.

    software_versions remains accepted for signature compatibility. Encoding
    provenance records this package rather than a caller-supplied version.
    """
    return convert(
        nifti_path,
        ref_dir,
        output_path,
        kind="seg",
        labels=organ_index,
        manufacturer=manufacturer,
        manufacturer_model_name=manufacturer_model_name,
        algorithm_type=algorithm_type,
        algorithm_name=algorithm_name,
        algorithm_version=algorithm_version,
    )
