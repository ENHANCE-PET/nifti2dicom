"""Thread-safe DICOM slice writing.

Fixes the old bug where ``save_slice()`` mutated the shared Dataset
from multiple threads. Now every write deep-copies first.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from pydicom.filewriter import dcmwrite
from pydicom.uid import generate_uid

from nifti2dicom.dicom_io import is_dicom_compressed
from nifti2dicom.pixel import encode_pixel_data, normalize_for_dicom, normalize_pt_dynamic_range
from nifti2dicom.tags import copy_tags


def save_slice(
    slice_ds: pydicom.Dataset,
    pixel_data: np.ndarray,
    series_description: str,
    filename: str,
    output_dir: str | Path,
    modality: str,
    header_source: pydicom.Dataset | None = None,
    ipp: np.ndarray | None = None,
    iop: np.ndarray | None = None,
    instance_number: int | None = None,
) -> None:
    """Write a single DICOM slice to disk (thread-safe).

    Parameters
    ----------
    slice_ds : pydicom.Dataset
        Reference DICOM slice (will be deep-copied, never mutated).
    pixel_data : np.ndarray
        2-D array of real-valued pixel data for this slice.
    series_description : str
        Text appended to the original SeriesDescription.
    filename : str
        Output filename (basename only).
    output_dir : str | Path
        Output directory.
    modality : str
        DICOM modality (CT, PT, etc.).
    header_source : pydicom.Dataset, optional
        If provided, copy non-spatial tags from this dataset.
    ipp : np.ndarray, optional
        Image Position (Patient) for this slice [x, y, z].
    iop : np.ndarray, optional
        Image Orientation (Patient) — 6 elements.
    instance_number : int, optional
        Override InstanceNumber.
    """
    ds = copy.deepcopy(slice_ds)

    if is_dicom_compressed(ds):
        ds.decompress()

    # Pixel encoding
    if modality == "PT":
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        stored, new_slope, new_intercept = normalize_pt_dynamic_range(
            pixel_data, slope, intercept
        )
        ds.RescaleSlope = str(new_slope)
        ds.RescaleIntercept = str(new_intercept)
        ds.PixelData = encode_pixel_data(stored)
        ds.Rows, ds.Columns = stored.shape
    else:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixel_rep = int(getattr(ds, "PixelRepresentation", 1))
        stored = normalize_for_dicom(pixel_data, slope, intercept, pixel_rep)
        ds.PixelData = encode_pixel_data(stored)
        ds.Rows, ds.Columns = stored.shape

    # Header tag copying
    if header_source is not None:
        copy_tags(ds, header_source)

    # Spatial tags
    if ipp is not None:
        ds.ImagePositionPatient = [str(v) for v in ipp]
    if iop is not None:
        ds.ImageOrientationPatient = [str(v) for v in iop]
    if instance_number is not None:
        ds.InstanceNumber = instance_number

    # Series metadata
    ds.SeriesNumber = int(getattr(ds, "SeriesNumber", 1)) * 10
    existing_desc = getattr(ds, "SeriesDescription", "")
    if existing_desc:
        ds.SeriesDescription = f"{existing_desc}_{series_description}"
    else:
        ds.SeriesDescription = series_description

    # Fresh UIDs to avoid collision
    ds.SOPInstanceUID = generate_uid()

    out_path = os.path.join(str(output_dir), filename)
    dcmwrite(out_path, ds, write_like_original=False)


def write_slices_parallel(
    dicom_slices: list[pydicom.Dataset],
    filenames: list[str],
    pixel_data_3d: np.ndarray,
    series_description: str,
    output_dir: str | Path,
    modality: str,
    header_source: pydicom.Dataset | None = None,
    ipp_list: np.ndarray | None = None,
    iop: np.ndarray | None = None,
    on_progress: Callable[[], None] | None = None,
) -> None:
    """Write all slices in parallel using a thread pool.

    Parameters
    ----------
    dicom_slices : list[pydicom.Dataset]
        Reference DICOM datasets (one per slice).
    filenames : list[str]
        Output filenames.
    pixel_data_3d : np.ndarray
        3-D array ``(num_slices, rows, cols)``.
    series_description : str
        Appended to SeriesDescription.
    output_dir : str | Path
        Output directory.
    modality : str
        DICOM modality.
    header_source : pydicom.Dataset, optional
        Header source for tag copying.
    ipp_list : np.ndarray, optional
        IPP array ``(num_slices, 3)``.
    iop : np.ndarray, optional
        IOP — 6-element array.
    on_progress : callable, optional
        Called once per completed slice (for progress bars).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor() as pool:
        futures = []
        for idx, (ds, fname) in enumerate(zip(dicom_slices, filenames, strict=True)):
            ipp = ipp_list[idx] if ipp_list is not None else None
            futures.append(
                pool.submit(
                    save_slice,
                    ds,
                    pixel_data_3d[idx],
                    series_description,
                    fname,
                    output_dir,
                    modality,
                    header_source=header_source,
                    ipp=ipp,
                    iop=iop,
                    instance_number=idx + 1,
                )
            )

        for future in as_completed(futures):
            future.result()  # raise any exception from the thread
            if on_progress:
                on_progress()
