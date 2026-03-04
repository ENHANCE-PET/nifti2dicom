"""NIfTI → DICOM image conversion (3-D and 4-D)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import nibabel as nib
from pydicom.uid import generate_uid

from nifti2dicom import cli_theme as theme
from nifti2dicom.dicom_io import load_dicom_series
from nifti2dicom.exceptions import ShapeMismatchError
from nifti2dicom.orientation import orient_nifti
from nifti2dicom.writer import write_slices_parallel


def convert_nifti_to_dicom(
    ref_dir: str | Path,
    nifti_path: str | Path,
    output_dir: str | Path,
    *,
    vendor: str | None = None,
    series_description: str = "converted by nifti2dicom",
    header_dir: str | Path | None = None,
    force_overwrite: bool = False,
) -> None:
    """Convert a NIfTI image to a DICOM series using a reference DICOM.

    Parameters
    ----------
    ref_dir : path
        Directory containing the reference DICOM series (provides spatial
        information and per-slice templates).
    nifti_path : path
        Path to the NIfTI file.
    output_dir : path
        Where to write the DICOM series.
    vendor : str, optional
        **Deprecated and ignored.** Kept for API compatibility.
    series_description : str
        Text appended to SeriesDescription.
    header_dir : path, optional
        If given, non-spatial DICOM tags are copied from the first slice
        in this directory instead of *ref_dir*.
    force_overwrite : bool
        If *True*, delete *output_dir* if it already exists.
    """
    ref_dir = Path(ref_dir)
    nifti_path = Path(nifti_path)
    output_dir = Path(output_dir)

    theme.section("LOADING", number="01")
    theme.info(f"NIfTI: {nifti_path}")
    theme.info(f"Reference DICOM: {ref_dir}")

    # Load NIfTI and orient
    img: nib.Nifti1Image = nib.load(str(nifti_path))  # type: ignore[assignment]
    data, ipp_list, iop = orient_nifti(img, vendor=vendor)
    ndim = len(img.shape)
    theme.info(f"Image dimensions: {ndim}D, oriented shape: {data.shape}")

    # Load reference DICOM
    dicom_slices, filenames = load_dicom_series(ref_dir)
    reference_slice = dicom_slices[0]
    modality = getattr(reference_slice, "Modality", "OT")

    # Validate shapes
    expected = (len(dicom_slices), reference_slice.Columns, reference_slice.Rows)
    if expected != data.shape:
        raise ShapeMismatchError(expected, data.shape)

    theme.ok(f"Shape match: {data.shape}")

    # Optional header source
    header_source = None
    if header_dir is not None:
        from nifti2dicom.dicom_io import load_dicom_series as _load
        header_slices, _ = _load(header_dir)
        header_source = header_slices[0]
        theme.info(f"Header source: {header_dir}")

    # Prepare output
    if output_dir.exists():
        if force_overwrite:
            shutil.rmtree(output_dir)
        else:
            theme.warn(f"{output_dir} already exists, skipping. Use --force to overwrite.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate fresh SeriesInstanceUID for the output
    series_uid = generate_uid()
    for ds in dicom_slices:
        ds.SeriesInstanceUID = series_uid

    # Write
    theme.section("WRITING", number="02")
    basenames = [os.path.basename(f) for f in filenames]

    with theme.progress(len(dicom_slices), "Writing DICOM slices") as tick:
        write_slices_parallel(
            dicom_slices,
            basenames,
            data,
            series_description,
            output_dir,
            modality,
            header_source=header_source,
            ipp_list=ipp_list,
            iop=iop,
            on_progress=tick,
        )

    theme.ok(f"Wrote {len(dicom_slices)} slices to {output_dir}")
