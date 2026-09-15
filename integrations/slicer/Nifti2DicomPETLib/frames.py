"""Validate native classic dynamic PET time blocks without Slicer dependencies.

ImageIndex follows DICOM PS3.3 C.8.9.4.1.9: slice index varies fastest,
then time-slice index. FrameReferenceTime describes each original plane and
need not be constant within a native time block.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import numpy as np
from pydicom.dataset import Dataset
from pydicom.errors import BytesLengthException, InvalidDicomError
from pydicom.uid import UID

_CLASSIC_PET = "1.2.840.10008.5.1.4.1.1.128"
_POSITION_TOLERANCE_MM = 0.001


@dataclass(frozen=True)
class PETFrameSet:
    frame_files: tuple[tuple[str, ...], ...]
    index_values: tuple[str, ...]
    index_name: str
    index_unit: str
    tag_name: str


def _integer(ds: Dataset, name: str, *, minimum: int = 1) -> int:
    try:
        value = getattr(ds, name)
        integer = int(value)
        if integer < minimum or float(value) != integer:
            raise ValueError
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"PET {name} must be an integer of at least {minimum}.") from exc
    return integer


def _number(ds: Dataset, name: str) -> float:
    try:
        number = float(getattr(ds, name))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"PET {name} must be a finite number.") from exc
    if not np.isfinite(number):
        raise ValueError(f"PET {name} must be a finite number.")
    return number


def _decimal(ds: Dataset, name: str) -> Decimal:
    _number(ds, name)
    try:
        return Decimal(str(getattr(ds, name)))
    except InvalidOperation as exc:
        raise ValueError(f"PET {name} must be a finite number.") from exc


def _vector(ds: Dataset, name: str, length: int) -> np.ndarray:
    try:
        values = np.asarray(getattr(ds, name), dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"PET {name} must contain {length} finite numbers.") from exc
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"PET {name} must contain {length} finite numbers.")
    return values


def _text(ds: Dataset, name: str, *, required: bool = False) -> str:
    raw = getattr(ds, name, None)
    value = "" if raw is None else str(raw).strip()
    if required and not value:
        raise ValueError(f"PET {name} is required.")
    return value


def _uid(ds: Dataset, name: str) -> str:
    value = _text(ds, name, required=True)
    if not UID(value).is_valid:
        raise ValueError(f"PET {name} must be a valid DICOM UID.")
    return value


def _check_shared_headers(headers: Sequence[Dataset]) -> None:
    first = headers[0]
    for name in ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID"):
        values = {_uid(ds, name) for ds in headers}
        if len(values) != 1:
            raise ValueError(f"PET images have inconsistent {name} values.")
    for name in (
        "PatientID",
        "PatientName",
        "IssuerOfPatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientPosition",
        "DecayCorrection",
        "CountsSource",
        "CorrectedImage",
        "Units",
        "SeriesDate",
        "SeriesTime",
    ):
        values = {_text(ds, name, required=name == "Units") for ds in headers}
        if len(values) != 1:
            raise ValueError(f"PET images have inconsistent {name} values.")

    seen_instances = set()
    for ds in headers:
        if _uid(ds, "SOPClassUID") != _CLASSIC_PET or _text(ds, "Modality") != "PT":
            raise ValueError("PET importer requires classic PET Image Storage with Modality PT.")
        instance = _uid(ds, "SOPInstanceUID")
        if instance in seen_instances:
            raise ValueError("PET selection contains duplicate SOPInstanceUID values.")
        seen_instances.add(instance)
        file_meta = getattr(ds, "file_meta", None)
        if file_meta is not None:
            for name, expected in (
                ("MediaStorageSOPClassUID", _CLASSIC_PET),
                ("MediaStorageSOPInstanceUID", instance),
            ):
                if name in file_meta and str(getattr(file_meta, name)) != expected:
                    raise ValueError(f"PET file meta {name} contradicts the dataset identity.")
        series_type = getattr(ds, "SeriesType", ())
        if (
            not isinstance(series_type, Sequence)
            or isinstance(series_type, str)
            or tuple(series_type) != ("DYNAMIC", "IMAGE")
        ):
            raise ValueError("PET SeriesType must be DYNAMIC\\IMAGE.")
        if (
            ("NumberOfFrames" in ds and _integer(ds, "NumberOfFrames") != 1)
            or "SharedFunctionalGroupsSequence" in ds
            or "PerFrameFunctionalGroupsSequence" in ds
        ):
            raise ValueError("PET enhanced or multiframe storage is unsupported.")
        for name, expected in (
            ("SamplesPerPixel", 1),
            ("BitsAllocated", 16),
            ("BitsStored", 16),
            ("HighBit", 15),
        ):
            if _integer(ds, name) != expected:
                raise ValueError(f"PET {name} must be {expected} for native 16-bit scalar storage.")
        representation = _integer(ds, "PixelRepresentation", minimum=0)
        if representation not in (0, 1) or representation != _integer(
            first, "PixelRepresentation", minimum=0
        ):
            raise ValueError("PET PixelRepresentation must be consistently unsigned or signed.")
        if _text(ds, "PhotometricInterpretation") != "MONOCHROME2":
            raise ValueError("PET PhotometricInterpretation must be MONOCHROME2.")
        if _number(ds, "RescaleSlope") <= 0:
            raise ValueError("PET RescaleSlope must be finite and positive for every image.")
        if _decimal(ds, "RescaleIntercept") != 0:
            raise ValueError("PET RescaleIntercept must be zero for every image.")


def _check_geometry(headers: Sequence[Dataset], slices: int) -> None:
    first = headers[0]
    orientation = _vector(first, "ImageOrientationPatient", 6)
    spacing = _vector(first, "PixelSpacing", 2)
    size = (_integer(first, "Rows"), _integer(first, "Columns"))
    axes = np.column_stack((orientation[:3], orientation[3:]))
    if not np.allclose(axes.T @ axes, np.eye(2), atol=1e-4, rtol=0):
        raise ValueError("PET ImageOrientationPatient axes must be unit and orthogonal.")
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    positions = []
    for ds in headers:
        current_spacing = _vector(ds, "PixelSpacing", 2)
        if np.any(current_spacing <= 0):
            raise ValueError("PET PixelSpacing must be positive.")
        if (
            (_integer(ds, "Rows"), _integer(ds, "Columns")) != size
            or not np.allclose(current_spacing, spacing, atol=1e-5, rtol=1e-5)
            or not np.allclose(
                _vector(ds, "ImageOrientationPatient", 6), orientation, atol=1e-5, rtol=0
            )
        ):
            raise ValueError("PET geometry has inconsistent dimensions, spacing or orientation.")
        positions.append(_vector(ds, "ImagePositionPatient", 3))
    all_positions = np.asarray(positions).reshape(-1, slices, 3)
    original = all_positions[0]
    projections = original @ normal
    for block in all_positions:
        if np.any(np.diff(block @ normal) <= 1e-4):
            raise ValueError("PET ImageIndex slice order must have unique increasing positions.")
    if slices > 1:
        step = (projections[-1] - projections[0]) / (slices - 1)
        ideal = original[0] + np.arange(slices)[:, None] * step * normal
    else:
        ideal = original
    if np.any(np.linalg.norm(all_positions - ideal, axis=2) > _POSITION_TOLERANCE_MM + 1e-9):
        raise ValueError(
            "PET geometry must share a regular slice grid across time; shifted, irregular "
            "or sheared positions require resampling and cannot be imported natively."
        )


def group_frames(records: Sequence[tuple[str, Dataset]]) -> PETFrameSet:
    """Return validated, spatially ordered native PET blocks without mutating headers.

    This intentionally validates only the classic dynamic scalar PET profile.
    Pixel decoding and its physical-value precision are the caller's responsibility.
    """
    try:
        return _group_frames(records)
    except (InvalidDicomError, BytesLengthException):
        # pydicom may decode raw elements only when a field is accessed. Its
        # exceptions can contain raw patient metadata, including in tracebacks.
        raise ValueError("PET image header contains malformed DICOM metadata.") from None


def _group_frames(records: Sequence[tuple[str, Dataset]]) -> PETFrameSet:
    if not records:
        raise ValueError("PET selection is empty; select a complete dynamic series.")
    paths = [path for path, _ in records]
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("PET selected image paths must be nonempty strings.")
    if len(set(paths)) != len(paths):
        raise ValueError("PET selection contains duplicate image paths.")
    headers = [ds for _, ds in records]
    _check_shared_headers(headers)
    dimensions = []
    for name in ("NumberOfSlices", "NumberOfTimeSlices"):
        values = {_integer(ds, name) for ds in headers}
        if len(values) != 1:
            raise ValueError(f"PET images have inconsistent {name} values.")
        dimensions.append(values.pop())
    slices, timepoints = dimensions
    indices = [_integer(ds, "ImageIndex") for ds in headers]
    if len(records) != slices * timepoints or sorted(indices) != list(range(1, len(records) + 1)):
        raise ValueError(
            "PET ImageIndex must uniquely cover all NumberOfSlices × NumberOfTimeSlices images; "
            "select the complete dynamic series."
        )
    ordered = [record for _, record in sorted(zip(indices, records, strict=True))]
    headers = [ds for _, ds in ordered]
    _check_geometry(headers, slices)

    times = [_decimal(ds, "FrameReferenceTime") for ds in headers]
    blocks = [times[t * slices : (t + 1) * slices] for t in range(timepoints)]
    for previous, current in zip(blocks, blocks[1:], strict=False):
        if any(after <= before for before, after in zip(previous, current, strict=True)):
            raise ValueError("PET ImageIndex contradicts increasing FrameReferenceTime per slice.")
    files = tuple(
        tuple(path for path, _ in ordered[t * slices : (t + 1) * slices]) for t in range(timepoints)
    )
    if all(len(set(block)) == 1 for block in blocks):
        if any(
            float(current[0]) <= float(previous[0])
            for previous, current in zip(blocks, blocks[1:], strict=False)
        ):
            raise ValueError("PET FrameReferenceTime values collapse in numeric sequence indices.")
        return PETFrameSet(
            files,
            tuple(str(headers[t * slices].FrameReferenceTime) for t in range(timepoints)),
            "FrameReferenceTime",
            "ms",
            "FrameReferenceTime",
        )
    return PETFrameSet(
        files, tuple(str(t + 1) for t in range(timepoints)), "TimeSlice", "count", "ImageIndex"
    )
