"""Legacy in-memory writer adapters. Encoding belongs to writers.image."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pydicom

from nifti2dicom.errors import GeometryError, OutputError
from nifti2dicom.geometry import validate_geometry
from nifti2dicom.models import Geometry, ImageVolume, ReferenceSeries
from nifti2dicom.writers.image import iter_image_datasets


def _geometry(
    ds: pydicom.Dataset,
    size: tuple[int, int, int],
    positions: np.ndarray | None = None,
    iop: np.ndarray | None = None,
) -> Geometry:
    orientation = np.asarray(ds.ImageOrientationPatient if iop is None else iop, dtype=float)
    row, col = orientation[:3], orientation[3:]
    spacing = np.asarray(ds.PixelSpacing, dtype=float)
    affine = np.eye(4)
    affine[:3, 0], affine[:3, 1] = row * spacing[1], col * spacing[0]
    affine[:3, 2] = np.cross(row, col) * float(getattr(ds, "SliceThickness", 1) or 1)
    affine[:3, 3] = np.asarray(ds.ImagePositionPatient if positions is None else positions[0])
    if positions is not None and len(positions) > 1:
        affine[:3, 2] = positions[1] - positions[0]
        expected = affine[:3, 3] + np.arange(len(positions))[:, None] * affine[:3, 2]
        if not np.allclose(positions, expected):
            raise GeometryError(
                "Legacy slice writing requires a uniformly spaced ordered volume.",
                hint="Use nifti2dicom.convert for 4D data.",
            )
    geometry = Geometry(affine, size)
    validate_geometry(geometry)
    return geometry


def _destination(output: str | Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in ("", ".", ".."):
        raise OutputError("A slice filename must be a basename within the output directory.")
    destination = Path(output) / filename
    if destination.exists():
        raise OutputError(f"Output file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


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
    """Write one slice with the caller's series identity and fresh instance identity."""
    rows, cols = pixel_data.shape
    geometry = _geometry(slice_ds, (cols, rows, 1), None if ipp is None else np.asarray([ipp]), iop)
    ref_geometry = _geometry(slice_ds, (int(slice_ds.Columns), int(slice_ds.Rows), 1))
    reference = ReferenceSeries((slice_ds,), (), ref_geometry)
    image = ImageVolume(np.asarray(pixel_data)[None, None], geometry)
    ds = next(
        iter_image_datasets(
            image, reference, description=series_description, header_source=header_source
        )
    )
    ds.SeriesInstanceUID = getattr(slice_ds, "SeriesInstanceUID", None) or ds.SeriesInstanceUID
    ds.InstanceNumber = instance_number if instance_number is not None else 1
    pydicom.dcmwrite(_destination(output_dir, filename), ds, enforce_file_format=True)


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
    """Historical batch entry point; modern serialization streams a volume."""
    if (
        not dicom_slices
        or len(dicom_slices) != len(filenames)
        or len(filenames) != len(pixel_data_3d)
    ):
        raise GeometryError(
            "Slice templates, filenames and pixel planes must have matching lengths."
        )
    if len(set(filenames)) != len(filenames):
        raise OutputError("Output slice filenames must be unique.")
    nz, rows, cols = pixel_data_3d.shape
    positions = np.asarray([ds.ImagePositionPatient for ds in dicom_slices], dtype=float)
    ref_geometry = _geometry(
        dicom_slices[0], (int(dicom_slices[0].Columns), int(dicom_slices[0].Rows), nz), positions
    )
    geometry = _geometry(
        dicom_slices[0],
        (cols, rows, nz),
        positions if ipp_list is None else np.asarray(ipp_list),
        iop,
    )
    reference = ReferenceSeries(tuple(dicom_slices), (), ref_geometry)
    image = ImageVolume(np.asarray(pixel_data_3d)[None], geometry)
    destinations = [_destination(output_dir, name) for name in filenames]
    datasets = iter_image_datasets(
        image, reference, description=series_description, header_source=header_source
    )
    for ds, destination in zip(datasets, destinations, strict=True):
        pydicom.dcmwrite(destination, ds, enforce_file_format=True)
        if on_progress:
            on_progress()
