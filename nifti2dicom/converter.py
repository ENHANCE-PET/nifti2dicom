"""Historical PUMA and v1 interfaces, routed through the shared conversion API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nifti2dicom.api import convert
from nifti2dicom.convert_image import convert_nifti_to_dicom
from nifti2dicom.convert_rgb import convert_rgb_nifti_to_dicom
from nifti2dicom.convert_seg import convert_nifti_seg_to_dicom
from nifti2dicom.models import ConversionResult
from nifti2dicom.resample import resample_image
from nifti2dicom.writer import save_slice

__all__ = [
    "nifti_to_dicom_with_resampling",
    "write_rgb_dicom_from_nifti",
    "save_dicom_from_nifti_image",
    "save_dicom_from_nifti_seg",
    "convert_nifti_to_dicom",
    "convert_rgb_nifti_to_dicom",
    "convert_nifti_seg_to_dicom",
    "resample_image",
    "save_slice",
]


def nifti_to_dicom_with_resampling(
    nifti_image_path: str,
    original_dicom_directory: str,
    dicom_output_directory: str,
    spatial_info_dicom_directory: str,
    series_description: str = "converted by nifti2dicom",
    verbose: bool = False,
) -> ConversionResult:
    """Use the spatial reference's complete grid, without flips or relabeling."""
    return convert(
        nifti_image_path,
        spatial_info_dicom_directory,
        dicom_output_directory,
        kind="image",
        geometry="reference",
        description=series_description,
        header_source=original_dicom_directory,
    )


def write_rgb_dicom_from_nifti(
    nifti_file_path: str, reference_dicom_series: str, output_directory: str
) -> ConversionResult:
    return convert(nifti_file_path, reference_dicom_series, output_directory, kind="rgb")


def save_dicom_from_nifti_image(
    ref_dir: str,
    nifti_path: str,
    output_dir: str,
    vendor: str | None = None,
    series_description: str = "converted by nifti2dicom",
    header_dir: str | None = None,
    force_overwrite: bool = False,
    verbose: bool = False,
) -> ConversionResult:
    return convert_nifti_to_dicom(
        ref_dir,
        nifti_path,
        output_dir,
        vendor=vendor,
        series_description=series_description,
        header_dir=header_dir,
        force_overwrite=force_overwrite,
    )


def save_dicom_from_nifti_seg(
    nifti_file: str,
    ref_dicom_series_dir: str,
    output_path: str,
    ORGAN_INDEX: dict[str, Any],
    verbose: bool = False,
    **kwargs: Any,
) -> ConversionResult:
    # v1 took NIfTI first; the early v2 alias took the reference first.
    if Path(nifti_file).is_dir() or str(ref_dicom_series_dir).lower().endswith((".nii", ".nii.gz")):
        nifti_file, ref_dicom_series_dir = ref_dicom_series_dir, nifti_file
    return convert_nifti_seg_to_dicom(
        ref_dicom_series_dir, nifti_file, output_path, ORGAN_INDEX, **kwargs
    )
