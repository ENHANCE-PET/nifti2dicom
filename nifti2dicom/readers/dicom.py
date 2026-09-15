"""Discover one coherent classic DICOM series and order its planes physically."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import UID

from nifti2dicom.errors import (
    AmbiguousReferenceError,
    GeometryError,
    ReferenceError,
    UnsupportedInputError,
)
from nifti2dicom.geometry import validate_geometry
from nifti2dicom.models import Geometry, ReferenceSeries


def _vector(ds: Dataset, name: str, length: int) -> np.ndarray:
    try:
        values = np.asarray(getattr(ds, name), dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise GeometryError(f"Reference DICOM lacks valid {name}.") from exc
    if values.shape != (length,) or not np.isfinite(values).all():
        raise GeometryError(f"Reference DICOM has invalid {name}.")
    return values


def _plane(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    position = _vector(ds, "ImagePositionPatient", 3)
    iop = _vector(ds, "ImageOrientationPatient", 6)
    spacing = _vector(ds, "PixelSpacing", 2)
    if np.any(spacing <= 0):
        raise GeometryError("Reference PixelSpacing must be positive.")
    orientation = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
    if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-4, rtol=0):
        raise GeometryError("Reference ImageOrientationPatient axes must be unit and orthogonal.")
    try:
        size = (int(ds.Columns), int(ds.Rows))
    except (AttributeError, TypeError, ValueError) as exc:
        raise GeometryError("Reference DICOM lacks valid Rows and Columns.") from exc
    if min(size) < 1:
        raise GeometryError("Reference Rows and Columns must be positive.")
    return position, orientation, spacing, size


def _position_rounding_tolerance(ds: Dataset, normal: np.ndarray) -> float:
    """Bound projected DS serialization error, never allowing over 5 microns.

    Scanner coordinates written to 0.01 mm can alternate neighboring slice
    gaps by 0.01 mm. Their individual plane errors remain at most 0.005 mm.
    Coarser coordinates must not justify larger departures from a regular grid.
    """
    uncertainty = []
    for value in ds.ImagePositionPatient:
        exponent = Decimal(str(value)).as_tuple().exponent
        assert isinstance(exponent, int)  # _plane already checked finite coordinates.
        uncertainty.append(0.5 * 10.0 ** min(exponent, -2))
    return max(1e-3, min(0.005, float(np.abs(normal) @ uncertainty)))


def _group_geometry(
    records: list[tuple[Dataset, Path]],
) -> tuple[Geometry, list[tuple[Dataset, Path]]]:
    first_position, direction, spacing, plane_size = _plane(records[0][0])
    normal = direction[:, 2] / np.linalg.norm(direction[:, 2])
    located = []
    for ds, path in records:
        position, candidate_direction, candidate_spacing, candidate_size = _plane(ds)
        if (
            candidate_size != plane_size
            or not np.allclose(candidate_spacing, spacing, atol=1e-5, rtol=1e-5)
            or not np.allclose(candidate_direction, direction, atol=1e-5, rtol=0)
        ):
            raise GeometryError(
                "Reference slices have inconsistent dimensions, spacing or orientation."
            )
        located.append((float(position @ normal), position, ds, path))
    located.sort(key=lambda entry: entry[0])
    projections = np.array([entry[0] for entry in located])
    if len(located) > 1:
        distances = np.diff(projections)
        if np.any(distances <= 1e-4):
            raise GeometryError(
                "Reference contains duplicate slice positions without distinct temporal groups.",
                hint="Choose a single acquisition or supply explicit temporal position metadata.",
            )
        # Anchor both ends rather than accumulating the median of rounded gaps.
        # Validate all original planes; small local gap differences must not
        # conceal a slowly drifting or otherwise nonuniform stack.
        slice_spacing = float((projections[-1] - projections[0]) / (len(located) - 1))
        ideal = projections[0] + np.arange(len(located)) * slice_spacing
        residual = np.abs(projections - ideal)
        tolerance = np.array([_position_rounding_tolerance(ds, normal) for _, _, ds, _ in located])
        if np.any(residual > tolerance + 1e-9):
            raise GeometryError(
                "Reference slice spacing is nonuniform.",
                details={
                    "max_plane_deviation_mm": float(residual.max()),
                    "max_allowed_deviation_mm": float(tolerance.max()),
                },
            )
    else:
        ds = records[0][0]
        declared_spacing = getattr(ds, "SpacingBetweenSlices", None)
        if declared_spacing is None or declared_spacing == "":
            declared_spacing = getattr(ds, "SliceThickness", None)
        if declared_spacing is None or declared_spacing == "":
            declared_spacing = 1.0
        try:
            slice_spacing = float(declared_spacing)
        except (TypeError, ValueError) as exc:
            raise GeometryError("Reference single-slice thickness is invalid.") from exc
        if not np.isfinite(slice_spacing) or slice_spacing <= 0:
            raise GeometryError("Reference single-slice spacing must be finite and positive.")
    first_position = located[0][1]
    for projection, position, _, _ in located:
        expected = first_position + normal * (projection - projections[0])
        if not np.allclose(position, expected, atol=1e-3, rtol=0):
            raise GeometryError("Reference slice origins are misaligned or contain in-plane shear.")
    affine = np.eye(4)
    affine[:3, 0] = direction[:, 0] * spacing[1]
    affine[:3, 1] = direction[:, 1] * spacing[0]
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = first_position
    geometry = Geometry(affine, (*plane_size, len(located)))
    validate_geometry(geometry)
    return geometry, [(ds, path) for _, _, ds, path in located]


def _read_header(path: Path) -> Dataset | None:
    """Accept raw DICOM datasets only when their identifying UIDs are valid."""
    try:
        try:
            return pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError:
            header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            for field in ("SOPClassUID", "SOPInstanceUID", "SeriesInstanceUID"):
                value = str(getattr(header, field, ""))
                if not value or not UID(value).is_valid:
                    return None
            return header
    except (OSError, ValueError, EOFError, TypeError, KeyError, NotImplementedError):
        return None


def _pet_integer(ds: Dataset, field: str) -> int:
    try:
        value = getattr(ds, field)
        integer = int(value)
        if integer < 1 or float(value) != integer:
            raise ValueError("Expected a positive integer")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ReferenceError(f"Reference PET {field} must be a positive integer.") from exc
    return integer


def _pet_temporal_groups(
    records: list[tuple[Dataset, Path]],
) -> dict[float, list[tuple[Dataset, Path]]] | None:
    """Use the native dynamic PET index, with slice index varying fastest.

    Defined by DICOM PS3.3 C.8.9.4.1.9; SeriesType is necessary to distinguish
    dynamic time blocks from gated cardiac dimensions and static stacks.
    """
    if str(getattr(records[0][0], "Modality", "")) != "PT":
        return None
    series_types = [tuple(getattr(ds, "SeriesType", ())) for ds, _ in records]
    if not any(value[:1] == ("DYNAMIC",) for value in series_types):
        return None
    if any(value != ("DYNAMIC", "IMAGE") for value in series_types):
        raise ReferenceError("Reference PET has inconsistent or unsupported SeriesType values.")
    dimensions = []
    for field in ("NumberOfSlices", "NumberOfTimeSlices"):
        values = {_pet_integer(ds, field) for ds, _ in records}
        if len(values) != 1:
            raise ReferenceError(f"Reference PET has inconsistent {field} values.")
        dimensions.append(values.pop())
    slices, timepoints = dimensions
    indices = [_pet_integer(ds, "ImageIndex") for ds, _ in records]
    if len(records) != slices * timepoints or sorted(indices) != list(range(1, len(records) + 1)):
        raise ReferenceError(
            "Reference PET ImageIndex must uniquely cover every declared timepoint and slice.",
            hint="Provide the complete dynamic PET series with coherent native temporal indices.",
        )
    frame_times = np.full((timepoints, slices), np.nan)
    for index, (ds, _) in zip(indices, records, strict=True):
        value = getattr(ds, "FrameReferenceTime", None)
        if value is None or value == "":
            continue
        try:
            frame_time = float(value)
        except (TypeError, ValueError) as exc:
            raise ReferenceError("Reference PET FrameReferenceTime is invalid.") from exc
        if not np.isfinite(frame_time):
            raise ReferenceError("Reference PET FrameReferenceTime must be finite.")
        frame_times.flat[index - 1] = frame_time
    for times in frame_times.T:
        if np.any(np.diff(times[np.isfinite(times)]) <= 0):
            raise ReferenceError("Reference PET ImageIndex contradicts FrameReferenceTime order.")
    groups: dict[float, list[tuple[Dataset, Path]]] = defaultdict(list)
    for index, record in zip(indices, records, strict=True):
        groups[float((index - 1) // slices)].append(record)
    return groups


def read_reference(path: str | Path, *, series_uid: str | None = None) -> ReferenceSeries:
    """Read headers recursively and select one geometrically coherent series."""
    source = Path(path).expanduser().resolve()
    if not source.exists() or not (source.is_file() or source.is_dir()):
        raise ReferenceError(f"DICOM reference path does not exist: {source}")
    candidates = (
        [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    )
    series: dict[str, list[tuple[Dataset, Path]]] = defaultdict(list)
    for candidate in candidates:
        header = _read_header(candidate)
        if header is not None and getattr(header, "SeriesInstanceUID", None):
            series[str(header.SeriesInstanceUID)].append((header, candidate))
    if not series:
        raise ReferenceError(
            "No DICOM reference series was found.",
            hint="Provide a folder containing classic DICOM image files.",
        )
    if series_uid is None:
        if len(series) != 1:
            raise AmbiguousReferenceError(
                f"Reference contains {len(series)} DICOM series; select one explicitly.",
                hint="Pass series_uid in Python or --series-uid on the command line.",
                details={"series_uids": sorted(series)},
            )
        series_uid = next(iter(series))
    if series_uid not in series:
        raise ReferenceError(
            f"Reference series {series_uid} was not found.", details={"series_uids": sorted(series)}
        )
    records = series[series_uid]
    warnings: list[str] = []
    for field in (
        "PatientID",
        "PatientName",
        "IssuerOfPatientID",
        "StudyInstanceUID",
        "FrameOfReferenceUID",
    ):
        values = {str(getattr(ds, field, "")).strip() for ds, _ in records}
        if len(values) != 1:
            raise ReferenceError(f"Reference series has mixed {field} values.")
        if values == {""}:
            if field == "StudyInstanceUID":
                raise ReferenceError("Reference series lacks StudyInstanceUID.")
            if field == "FrameOfReferenceUID":
                warnings.append(
                    "Reference lacks FrameOfReferenceUID; shared spatial identity is unknown."
                )
    seen_instances = set()
    for ds, _ in records:
        sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        if not sop_uid:
            raise ReferenceError("Reference image lacks SOPInstanceUID.")
        if sop_uid in seen_instances:
            raise ReferenceError(f"Reference contains duplicate SOPInstanceUID {sop_uid}.")
        seen_instances.add(sop_uid)
        try:
            multiframe = int(getattr(ds, "NumberOfFrames", 1)) != 1
        except (TypeError, ValueError) as exc:
            raise ReferenceError("Reference NumberOfFrames is invalid.") from exc
        sop_class_name = UID(str(getattr(ds, "SOPClassUID", ""))).name.lower()
        if (
            multiframe
            or "enhanced" in sop_class_name
            or "multi-frame" in sop_class_name
            or "SharedFunctionalGroupsSequence" in ds
            or "PerFrameFunctionalGroupsSequence" in ds
        ):
            raise UnsupportedInputError(
                "Enhanced or multiframe DICOM reference input is unsupported.",
                hint="Provide the corresponding classic single-frame DICOM series.",
            )
    for field in ("Modality", "SOPClassUID"):
        values = {str(getattr(ds, field, "")) for ds, _ in records}
        if len(values) != 1 or values == {""}:
            raise ReferenceError(f"Reference series has missing or inconsistent {field}.")

    time_field = next(
        (
            field
            for field in (
                "TemporalPositionIdentifier",
                "TemporalPositionIndex",
            )
            if any(getattr(ds, field, None) is not None for ds, _ in records)
        ),
        None,
    )
    pet_groups = _pet_temporal_groups(records) if time_field is None else None
    if (
        time_field is None
        and pet_groups is None
        and any("FrameReferenceTime" in ds for ds, _ in records)
    ):
        # PET reference times can differ from slice to slice within one volume.
        # Only repeated spatial planes establish that temporal grouping is needed.
        normal = _plane(records[0][0])[1][:, 2]
        positions = sorted(
            float(_vector(ds, "ImagePositionPatient", 3) @ normal) for ds, _ in records
        )
        if np.any(np.diff(positions) <= 1e-4):
            time_field = "FrameReferenceTime"
    groups: dict[float, list[tuple[Dataset, Path]]] = defaultdict(list)
    if pet_groups is not None:
        groups = pet_groups
    else:
        for ds, candidate in records:
            try:
                time = float(getattr(ds, time_field)) if time_field is not None else 0.0
            except (AttributeError, TypeError, ValueError) as exc:
                raise ReferenceError(
                    "Reference has incomplete or invalid temporal position metadata."
                ) from exc
            if not np.isfinite(time):
                raise ReferenceError("Reference temporal position metadata must be finite.")
            groups[time].append((ds, candidate))
    geometry = None
    ordered = []
    pet_slice_direction = None
    for time in sorted(groups):
        frame_geometry, frame_records = _group_geometry(groups[time])
        if pet_groups is not None:
            first_index = int(time) * frame_geometry.size[2] + 1
            expected = list(range(first_index, first_index + frame_geometry.size[2]))
            actual = [int(ds.ImageIndex) for ds, _ in frame_records]
            direction = 1 if actual == expected else -1 if actual == expected[::-1] else 0
            if not direction or pet_slice_direction not in (None, direction):
                raise GeometryError("Reference PET ImageIndex disagrees with physical slice order.")
            pet_slice_direction = direction
        if geometry is None:
            geometry = frame_geometry
        elif frame_geometry.size != geometry.size or not np.allclose(
            frame_geometry.affine,
            geometry.affine,
            atol=1e-3,
            rtol=0,
        ):
            raise GeometryError("Reference temporal groups do not share the same spatial geometry.")
        ordered.extend(frame_records)
    assert geometry is not None
    if pet_slice_direction == -1:
        # PS3.3 C.8.9.4.1.9 requires ascending image-plane-normal projection.
        # Recover only the same complete reversal in every native time block,
        # with complete temporal evidence. Keep source datasets/UIDs untouched.
        try:
            timing = np.asarray([float(ds.FrameReferenceTime) for ds, _ in ordered]).reshape(
                len(groups), geometry.size[2]
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ReferenceError(
                "Recovering descending PET ImageIndex requires complete FrameReferenceTime."
            ) from exc
        if not np.isfinite(timing).all() or np.any(np.diff(timing, axis=0) <= 0):
            raise ReferenceError("Descending PET ImageIndex contradicts FrameReferenceTime order.")
        warnings.append(
            "Reference PET ImageIndex uses nonconformant descending slice order; "
            "recovered the consistent reversal in every timepoint. "
            "Original source indices and SOP identities are retained."
        )
    deviation = max(
        float(
            np.linalg.norm(
                _vector(ds, "ImagePositionPatient", 3) - geometry.position(index % geometry.size[2])
            )
        )
        for index, (ds, _) in enumerate(ordered)
    )
    if deviation > 1e-3 + 1e-9:
        warnings.append(
            f"Reference positions are rounded (maximum grid deviation {deviation:.6g} mm); "
            "the resampling grid spans the original first and last planes."
        )
    if geometry.size[2] == 1 and any(
        getattr(ds, "SpacingBetweenSlices", None) in (None, "")
        and getattr(ds, "SliceThickness", None) in (None, "")
        for ds, _ in ordered
    ):
        warnings.append("Reference single-slice spacing is unknown; 1 mm was assumed.")
    return ReferenceSeries(
        slices=tuple(ds for ds, _ in ordered),
        paths=tuple(candidate for _, candidate in ordered),
        geometry=geometry,
        timepoints=len(groups),
        warnings=tuple(warnings),
    )
