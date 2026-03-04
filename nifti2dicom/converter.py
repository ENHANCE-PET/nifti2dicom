"""Backward-compatibility shim for PUMA and other downstream consumers.

Re-exports all public functions that PUMA depends on::

    from nifti2dicom.converter import nifti_to_dicom_with_resampling
    from nifti2dicom.converter import write_rgb_dicom_from_nifti

All functions delegate to the new split modules.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pydicom
import SimpleITK as sitk

from nifti2dicom import cli_theme as theme
from nifti2dicom.convert_image import convert_nifti_to_dicom
from nifti2dicom.convert_rgb import convert_rgb_nifti_to_dicom
from nifti2dicom.convert_seg import convert_nifti_seg_to_dicom
from nifti2dicom.resample import resample_image
from nifti2dicom.writer import save_slice

# ── PUMA-facing API ───────────────────────────────────────────────


def nifti_to_dicom_with_resampling(
    nifti_image_path: str,
    original_dicom_directory: str,
    dicom_output_directory: str,
    spatial_info_dicom_directory: str,
    series_description: str = "converted by nifti2dicom",
    verbose: bool = False,
) -> None:
    """Convert a NIfTI to DICOM with resampling to match original geometry.

    This is the PUMA-facing signature — kept identical for backward compat.
    Internally delegates to the new modules.
    """
    os.makedirs(dicom_output_directory, exist_ok=True)

    theme.section("LOADING", number="01")
    theme.info(f"Original DICOM: {original_dicom_directory}")
    theme.info(f"Spatial reference: {spatial_info_dicom_directory}")
    theme.info(f"NIfTI: {nifti_image_path}")

    # Load original DICOM geometry
    native_img = _load_sitk_series(original_dicom_directory)
    native_size = native_img.GetSize()
    native_spacing = native_img.GetSpacing()

    # Load spatial reference
    ref_img = _load_sitk_series(spatial_info_dicom_directory)

    # Load and resample NIfTI
    nifti_img = sitk.ReadImage(nifti_image_path)
    flipped = sitk.Flip(nifti_img, [False, True, False])
    resampled = resample_image(flipped, "linear", native_spacing, native_size)

    # Align to reference geometry
    resampled.SetOrigin(ref_img.GetOrigin())
    resampled.SetDirection(ref_img.GetDirection())

    # Load DICOM slices for writing
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(original_dicom_directory)
    dicom_slices = [pydicom.dcmread(f) for f in dicom_names]

    arr = sitk.GetArrayFromImage(resampled)  # (Z, Y, X)

    theme.section("WRITING", number="02")
    total = len(dicom_slices)
    with theme.progress(total, "Writing DICOM slices") as tick:
        with ThreadPoolExecutor() as pool:
            futures = []
            for idx, ds in enumerate(dicom_slices):
                ipp_raw = resampled.TransformIndexToPhysicalPoint((0, 0, idx))
                ipp = np.array(ipp_raw)
                futures.append(
                    pool.submit(
                        save_slice,
                        ds,
                        arr[idx],
                        series_description,
                        f"slice_{idx:04d}.dcm",
                        dicom_output_directory,
                        ds.Modality,
                        ipp=ipp,
                        instance_number=idx + 1,
                    )
                )
            for f in as_completed(futures):
                f.result()
                tick()

    theme.ok(f"Wrote {total} slices to {dicom_output_directory}")


def write_rgb_dicom_from_nifti(
    nifti_file_path: str,
    reference_dicom_series: str,
    output_directory: str,
) -> None:
    """Convert an RGB NIfTI to DICOM — PUMA-facing wrapper."""
    convert_rgb_nifti_to_dicom(reference_dicom_series, nifti_file_path, output_directory)


# ── Legacy aliases ────────────────────────────────────────────────

save_dicom_from_nifti_image = convert_nifti_to_dicom
save_dicom_from_nifti_seg = convert_nifti_seg_to_dicom


# ── Internal helpers ──────────────────────────────────────────────


def _load_sitk_series(directory: str) -> sitk.Image:
    """Load a DICOM series via SimpleITK."""
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(names)
    return reader.Execute()  # type: ignore[no-any-return]
