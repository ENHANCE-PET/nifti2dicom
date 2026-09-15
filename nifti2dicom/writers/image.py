"""Classic derived CT, MR and PET encoding from a validated physical grid."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filewriter import dcmwrite
from pydicom.uid import (
    PYDICOM_IMPLEMENTATION_UID,
    CTImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    PositronEmissionTomographyImageStorage,
    generate_uid,
)
from pydicom.valuerep import DA, TM, format_number_as_ds

from nifti2dicom.errors import PixelEncodingError, ReferenceError, UnsupportedInputError
from nifti2dicom.models import ImageVolume, Progress, ProgressCallback, ReferenceSeries
from nifti2dicom.pixels import encode_pixels
from nifti2dicom.writers.codes import omit_incomplete_optional_codes

# Only identity and study context may be supplied by a separate header. UIDs
# belong to the selected reference and the new series, never to that header.
_CONTEXT = (
    "SpecificCharacterSet",
    "PatientName",
    "PatientID",
    "IssuerOfPatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientSize",
    "PatientWeight",
    "StudyDate",
    "StudyTime",
    "ReferringPhysicianName",
    "StudyID",
    "AccessionNumber",
    "StudyDescription",
)
_TYPE2 = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyDate",
    "StudyTime",
    "ReferringPhysicianName",
    "StudyID",
    "AccessionNumber",
    "PositionReferenceIndicator",
)
_PET_TIMING = ("FrameReferenceTime", "ActualFrameDuration")
_MR = (
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "RepetitionTime",
    "EchoTime",
    "EchoTrainLength",
    "InversionTime",
    "TriggerTime",
)
_PET = (
    "Units",
    "SUVType",
    "CountsSource",
    "SeriesDate",
    "SeriesTime",
    "DecayCorrection",
    "CorrectedImage",
    "CollimatorType",
    "RadiopharmaceuticalInformationSequence",
    "PatientOrientationCodeSequence",
    "PatientGantryRelationshipCodeSequence",
)
_STORAGE = {
    "CT": CTImageStorage,
    "MR": MRImageStorage,
    "PT": PositronEmissionTomographyImageStorage,
}


def _copy_fields(target: Dataset, source: Dataset, fields: tuple[str, ...]) -> None:
    for keyword in fields:
        element = source.data_element(keyword) if keyword in source else None
        if element is not None:
            target.add(deepcopy(element))


def _require(source: Dataset, fields: tuple[str, ...], modality: str) -> None:
    missing = []
    for key in fields:
        element = source.data_element(key) if key in source else None
        if element is None or element.is_empty:
            missing.append(key)
    if missing:
        raise ReferenceError(
            f"{modality} reference is missing required metadata: " + ", ".join(missing) + ".",
            hint="Use a complete reference series from the same acquisition.",
        )


def _validate_layout(image: ImageVolume, *, rgb: bool = False) -> None:
    shape = image.data.shape
    if (
        len(shape) != (5 if rgb else 4)
        or any(n == 0 for n in shape)
        or (rgb and shape[-1] != 3)
        or tuple(reversed(shape[1:4])) != tuple(image.geometry.size)
    ):
        raise PixelEncodingError("Pixel array dimensions do not match the image geometry.")
    if shape[2] > 65535 or shape[3] > 65535:
        raise PixelEncodingError("DICOM rows and columns must each be at most 65535.")
    if image.time_spacing is not None and (
        not np.isfinite(image.time_spacing) or image.time_spacing <= 0
    ):
        raise PixelEncodingError("Temporal spacing must be finite and positive.")


def _matching_spatial_axes(
    image: ImageVolume, reference: ReferenceSeries
) -> tuple[np.ndarray, np.ndarray] | None:
    """Establish exact voxel correspondence under signed spatial permutations.

    A signed permutation and equal dimensions establish discrete correspondence;
    all physical corners must agree within bounded NIfTI header roundoff.
    This does not establish correspondence between acquisition planes.
    """
    if (
        image.timepoints != reference.timepoints
        or len(reference.slices) != reference.timepoints * reference.geometry.size[2]
    ):
        return None
    transform = np.linalg.solve(reference.geometry.affine, image.geometry.affine)
    axes = np.rint(transform[:3, :3]).astype(int)
    if (
        not np.allclose(transform[:3, :3], axes, rtol=0, atol=1e-5)
        or not np.all(np.abs(axes).sum(axis=0) == 1)
        or not np.all(np.abs(axes).sum(axis=1) == 1)
        or not np.array_equal(np.abs(axes) @ image.geometry.size, reference.geometry.size)
    ):
        return None
    corners = np.array(list(product(*((0, size - 1) for size in image.geometry.size))))
    offset = np.where(axes.sum(axis=1) < 0, np.asarray(reference.geometry.size) - 1, 0)
    mapped = corners @ axes.T + offset
    image_affine, reference_affine = image.geometry.affine, reference.geometry.affine
    difference = (
        corners @ image_affine[:3, :3].T
        + image_affine[:3, 3]
        - mapped @ reference_affine[:3, :3].T
        - reference_affine[:3, 3]
    )
    # NIfTI-1 stores affine coefficients as float32. Bound accumulated
    # coefficient roundoff, including canonical flip arithmetic, at each
    # corner instead of applying a fixed voxel tolerance to a long volume.
    roundoff = np.finfo(np.float32).eps * (
        corners @ np.abs(image_affine[:3, :3]).T
        + np.abs(image_affine[:3, 3])
        + mapped @ np.abs(reference_affine[:3, :3]).T
        + np.abs(reference_affine[:3, 3])
    )
    if np.any(np.abs(difference) > np.maximum(1e-6, roundoff)) or np.any(
        np.linalg.norm(difference, axis=1) > 1e-3
    ):
        return None
    return axes, offset


def _matching_frames(image: ImageVolume, reference: ReferenceSeries) -> tuple[int, ...] | None:
    """Match acquisition planes, including in-plane axis swaps and flips."""
    spatial = _matching_spatial_axes(image, reference)
    if spatial is None:
        return None
    axes, offset = spatial
    if np.any(axes[2, :2]) or np.any(axes[:2, 2]):
        return None
    slice_indices = offset[2] + np.arange(image.geometry.size[2]) * axes[2, 2]
    frame_indices = tuple(
        t * reference.geometry.size[2] + int(z)
        for t in range(image.timepoints)
        for z in slice_indices
    )
    check_spacing = image.time_spacing is not None and image.timepoints > 1
    if str(getattr(reference.first, "Modality", "")).upper() == "PT":
        times = [getattr(reference.slices[i], "FrameReferenceTime", None) for i in frame_indices]
        if any(t is None or t == "" for t in times):
            return None
        try:
            timing = np.asarray(times, dtype=float).reshape(image.timepoints, -1)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(timing).all():
            return None
        if image.time_spacing is not None and check_spacing:
            # Each column is one physical slice across all timepoints. PET's
            # effective reference time may differ between slices in a volume.
            if not np.allclose(
                np.diff(timing, axis=0), image.time_spacing * 1000, rtol=1e-5, atol=1e-3
            ):
                return None
    elif image.timepoints > 1:
        resolutions = [
            getattr(reference.slices[i], "TemporalResolution", None) for i in frame_indices
        ]
        if all(value in (None, "") for value in resolutions):
            if check_spacing:
                return None
        else:
            if any(value in (None, "") for value in resolutions):
                return None
            try:
                spacing = np.asarray(resolutions, dtype=float)
            except (TypeError, ValueError):
                return None
            if (
                not np.isfinite(spacing).all()
                or np.any(spacing <= 0)
                or not np.allclose(spacing, spacing[0], rtol=1e-5, atol=1e-3)
            ):
                return None
            if image.time_spacing is not None and not np.allclose(
                spacing, image.time_spacing * 1000, rtol=1e-5, atol=1e-3
            ):
                return None
    return frame_indices


def _reformatted_mr_sources(
    image: ImageVolume, reference: ReferenceSeries
) -> tuple[tuple[Dataset, ...] | None, tuple[str, ...]]:
    """Retain only volume-uniform facts when an exact reorientation mixes planes."""
    spatial = _matching_spatial_axes(image, reference)
    unavailable = (
        "MR acquisition metadata omitted: exact spatial and temporal correspondence "
        "with the reference could not be established."
    )
    if spatial is None:
        return None, (unavailable,)
    axes, _ = spatial
    if not np.any(axes[2, :2]):
        # A plane match rejected by _matching_frames must not bypass its
        # temporal consistency checks through this volume-only path.
        return None, (unavailable,)
    count = reference.geometry.size[2]
    groups = [reference.slices[t * count : (t + 1) * count] for t in range(image.timepoints)]
    if image.timepoints > 1:
        time_field = next(
            (
                key
                for key in ("TemporalPositionIdentifier", "TemporalPositionIndex")
                if any(key in ds for ds in reference.slices)
            ),
            None,
        )
        try:
            positions = np.array(
                [[float(getattr(ds, time_field or "")) for ds in group] for group in groups]
            )
        except (AttributeError, TypeError, ValueError):
            return None, (unavailable,)
        if (
            not np.isfinite(positions).all()
            or np.any(positions <= 0)
            or np.any(positions != np.rint(positions))
            or np.any(positions != positions[:, :1])
            or np.any(np.diff(positions[:, 0]) <= 0)
        ):
            return None, (unavailable,)
    declared_counts = [getattr(ds, "NumberOfTemporalPositions", None) for ds in reference.slices]
    if any(value is not None for value in declared_counts):
        try:
            if any(value is None or float(value) != image.timepoints for value in declared_counts):
                return None, (unavailable,)
        except (TypeError, ValueError):
            return None, (unavailable,)
    summaries = []
    warnings = []
    fields = (*_MR, "AcquisitionDate", "AcquisitionTime", "TemporalResolution")
    for group in groups:
        summary = Dataset()
        for key in fields:
            elements = [ds.data_element(key) if key in ds else None for ds in group]
            if all(element == elements[0] for element in elements):
                _copy_fields(summary, group[0], (key,))
            elif any(element is not None and not element.is_empty for element in elements):
                warnings.append(
                    f"MR {key} omitted from reformatted planes: it is not uniform "
                    "within the contributing source volume."
                )
        _require(summary, _MR[:2], "Reformatted MR")
        summaries.append(summary)
    resolutions = [getattr(ds, "TemporalResolution", None) for ds in reference.slices]
    if any(value not in (None, "") for value in resolutions):
        try:
            values = np.asarray(resolutions, dtype=float)
        except (TypeError, ValueError):
            return None, (unavailable,)
        if (
            not np.isfinite(values).all()
            or np.any(values <= 0)
            or not np.allclose(values, values[0], rtol=1e-5, atol=1e-3)
            or (
                image.time_spacing is not None
                and not np.allclose(values, image.time_spacing * 1000, rtol=1e-5, atol=1e-3)
            )
        ):
            return None, (unavailable,)
    if image.timepoints > 1 and image.time_spacing is not None:
        # Sample spacing cannot replace irregular or incomplete acquisition
        # timestamps. Dates, when present, also disambiguate midnight rollover.
        try:
            dates = [getattr(ds, "AcquisitionDate", "") for ds in summaries]
            if any(dates) and not all(dates):
                return None, (unavailable,)
            stamps = [
                datetime.combine(DA(date or "20000101"), TM(ds.AcquisitionTime))
                for date, ds in zip(dates, summaries, strict=True)
            ]
        except (AttributeError, TypeError, ValueError):
            return None, (unavailable,)
        intervals = [(b - a).total_seconds() for a, b in zip(stamps[:-1], stamps[1:], strict=True)]
        if not np.allclose(intervals, image.time_spacing, rtol=1e-5, atol=1e-6):
            return None, (unavailable,)
    return tuple(summaries), tuple(dict.fromkeys(warnings))


def _base_dataset(
    reference: ReferenceSeries,
    *,
    sop_class: str,
    modality: str,
    description: str | None,
    header_source: Dataset | None,
) -> Dataset:
    if not reference.slices:
        raise ReferenceError("The reference series is empty.")
    first = reference.first
    ds = Dataset()
    _copy_fields(ds, first, _CONTEXT)
    if header_source is not None:
        for keyword in ("PatientID", "IssuerOfPatientID", "StudyInstanceUID"):
            expected = str(getattr(first, keyword, "") or "").strip()
            supplied = str(getattr(header_source, keyword, "") or "").strip()
            if expected and supplied and expected != supplied:
                raise ReferenceError(
                    f"Header source conflicts with reference {keyword}.",
                    hint="Use header metadata from the same patient and study.",
                )
        _copy_fields(ds, header_source, _CONTEXT)
    # Output uses Unicode regardless of either input's character encoding.
    ds.SpecificCharacterSet = "ISO_IR 192"
    for keyword in _TYPE2:
        if keyword not in ds:
            setattr(ds, keyword, "")
    ds.StudyInstanceUID = getattr(first, "StudyInstanceUID", None) or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = getattr(first, "FrameOfReferenceUID", None) or generate_uid()
    ds.SOPClassUID = sop_class
    ds.Modality = modality
    ds.SeriesNumber = int(getattr(first, "SeriesNumber", None) or 0) + 1000
    ds.SeriesDescription = (description if description is not None else "NIfTI conversion")[:64]
    ds.ImageType = ["DERIVED", "SECONDARY", "OTHER"]
    ds.Manufacturer = "nifti2dicom"
    now = datetime.now(timezone.utc)
    ds.SeriesDate = ds.ContentDate = ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.SeriesTime = ds.ContentTime = ds.InstanceCreationTime = now.strftime("%H%M%S.%f")
    return ds


def _frame_dataset(
    base: Dataset,
    image: ImageVolume,
    reference: ReferenceSeries,
    index: int,
    source_index: int | None,
) -> Dataset:
    ds = deepcopy(base)
    z_count = image.geometry.size[2]
    t, z = divmod(index, z_count)
    source = reference.slices[source_index] if source_index is not None else reference.first
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    ds.InstanceNumber = index + 1
    ds.Rows, ds.Columns = image.data.shape[2:4]
    ds.PixelSpacing = [format_number_as_ds(v) for v in image.geometry.pixel_spacing]
    ds.ImageOrientationPatient = [format_number_as_ds(v) for v in image.geometry.iop]
    ds.ImagePositionPatient = [format_number_as_ds(v) for v in image.geometry.position(z)]
    ds.SliceThickness = format_number_as_ds(image.geometry.spacing[2])
    # CT positions already establish spacing; older IOD validators reject the
    # optional Spacing Between Slices attribute in the CT Image Plane module.
    if ds.Modality != "CT":
        ds.SpacingBetweenSlices = ds.SliceThickness
    if image.timepoints > 1 and ds.Modality != "PT":
        ds.TemporalPositionIdentifier = t + 1
        ds.NumberOfTemporalPositions = image.timepoints
    if source_index is not None:
        _copy_fields(ds, source, ("AcquisitionDate", "AcquisitionTime"))
        if ds.Modality == "PT":
            _copy_fields(ds, source, _PET_TIMING)
        elif ds.Modality in {"CT", "MR"} and image.timepoints > 1:
            _copy_fields(ds, source, ("TemporalResolution",))
        _copy_fields(
            ds,
            source,
            ("LossyImageCompression", "LossyImageCompressionRatio", "LossyImageCompressionMethod"),
        )
    if image.time_spacing is not None and image.timepoints > 1 and ds.Modality != "PT":
        ds.TemporalResolution = format_number_as_ds(image.time_spacing * 1000)
    return ds


def _series_profile(ds: Dataset, reference: ReferenceSeries, matched: bool) -> tuple[str, ...]:
    """Keep series acquisition context uniform and conditional on correspondence."""
    # General Series requires this for paired anatomy; unknown remains empty.
    ds.Laterality = ""
    if ds.Modality in {"CT", "MR"}:
        ds.PatientPosition = ""
    if ds.Modality == "CT":
        ds.ImageType = ["DERIVED", "SECONDARY"]
    if not matched:
        return ()
    warnings = []
    if ds.Modality in {"CT", "MR"}:
        positions = {
            str(getattr(source, "PatientPosition", "") or "") for source in reference.slices
        }
        if len(positions) == 1:
            ds.PatientPosition = next(iter(positions))
        elif any(positions):
            warnings.append(
                "Reference PatientPosition is incomplete or inconsistent; "
                "output series patient position is unknown."
            )
    lateralities = {str(getattr(source, "Laterality", "") or "") for source in reference.slices}
    if len(lateralities) == 1 and lateralities <= {"L", "R"}:
        ds.Laterality = next(iter(lateralities))
    if ds.Modality == "CT":
        classifications = set()
        for source in reference.slices:
            image_type = getattr(source, "ImageType", None)
            if image_type is None or isinstance(image_type, str) or len(image_type) < 3:
                classifications.add("")
            else:
                classifications.add(str(image_type[2]))
        # AXIAL denotes cross-sectional CT, including sagittal/coronal/oblique.
        # Missing, mixed or unknown classes must not become a guessed value.
        if len(classifications) == 1 and classifications <= {"AXIAL", "LOCALIZER"}:
            ds.ImageType.append(next(iter(classifications)))
    if any(lateralities) and not ds.Laterality:
        warnings.append(
            "Reference Laterality is incomplete, inconsistent or unsupported; "
            "output series laterality is unknown."
        )
    return tuple(warnings)


def _profile(ds: Dataset, source: Dataset, image: ImageVolume, matched: bool) -> tuple[str, ...]:
    modality = ds.Modality
    if modality == "CT":
        ds.KVP = getattr(source, "KVP", "") if matched else ""
        ds.AcquisitionNumber = getattr(source, "AcquisitionNumber", "") if matched else ""
        ds.RescaleType = getattr(source, "RescaleType", None) or "US"
    elif modality == "MR":
        _require(source, ("ScanningSequence", "SequenceVariant"), "MR")
        _copy_fields(ds, source, _MR[:2])
        for keyword in _MR[2:7]:
            setattr(ds, keyword, getattr(source, keyword, "") if matched else "")
        if "IR" in source.ScanningSequence:
            ds.InversionTime = getattr(source, "InversionTime", "") if matched else ""
        options = ds.ScanOptions
        options = (options,) if isinstance(options, str) else options or ()
        if any(value in {"CG", "PPG"} for value in options):
            ds.TriggerTime = getattr(source, "TriggerTime", "") if matched else ""
    elif modality == "PT":
        # The PET Image Module specializes Image Type value 2 as PRIMARY.
        ds.ImageType = ["DERIVED", "PRIMARY", "OTHER"]
        _require(
            source,
            ("Units", "CountsSource", "SeriesDate", "SeriesTime", "DecayCorrection", "SeriesType"),
            "PT",
        )
        if (
            isinstance(source.SeriesType, str)
            or len(source.SeriesType) != 2
            or source.SeriesType[0] not in ("STATIC", "DYNAMIC", "WHOLE BODY", "GATED")
            or source.SeriesType[1] not in ("IMAGE", "REPROJECTION")
        ):
            raise ReferenceError(
                "PT reference has an invalid SeriesType.",
                hint="Use a complete classic PET image reference series.",
            )
        if source.SeriesType[0] == "GATED" or source.SeriesType[1] != "IMAGE":
            raise UnsupportedInputError("Gated and reprojection PET references are not supported.")
        _copy_fields(ds, source, _PET)
        ds.SeriesType = ["DYNAMIC" if image.timepoints > 1 else "STATIC", "IMAGE"]
        ds.NumberOfSlices = image.geometry.size[2]
        if image.timepoints > 1:
            ds.NumberOfTimeSlices = image.timepoints
        t, z = divmod(int(ds.InstanceNumber) - 1, image.geometry.size[2])
        normal = np.cross(image.geometry.iop[:3], image.geometry.iop[3:])
        if np.dot(normal, image.geometry.affine[:3, 2]) < 0:
            z = image.geometry.size[2] - 1 - z
        ds.ImageIndex = t * image.geometry.size[2] + z + 1
        for keyword in (
            "AcquisitionDate",
            "AcquisitionTime",
            "ActualFrameDuration",
            "CorrectedImage",
            "CollimatorType",
        ):
            if keyword not in ds:
                setattr(ds, keyword, "")
        for keyword in (
            "RadiopharmaceuticalInformationSequence",
            "PatientOrientationCodeSequence",
            "PatientGantryRelationshipCodeSequence",
        ):
            if keyword not in ds:
                setattr(ds, keyword, [])
        _require(ds, ("FrameReferenceTime",), "PT output")
        if ds.DecayCorrection != "NONE":
            if not matched:
                raise ReferenceError(
                    "Decay-corrected PET needs matching reference frame geometry "
                    "and timing to preserve its decay factors.",
                    hint="Use the matching NIfTI grid and reference acquisition.",
                )
            _require(source, ("DecayFactor",), "PT")
            ds.DecayFactor = source.DecayFactor
        warnings: list[str] = []
        for isotope in ds.RadiopharmaceuticalInformationSequence:
            # Only the optional tracer code, not the required isotope sequence
            # or its dose/half-life/timing facts (PS3.3 C.8.9.2).
            warnings.extend(
                omit_incomplete_optional_codes(isotope, ("RadiopharmaceuticalCodeSequence",))
            )
        return tuple(warnings)
    return ()


def _save_frames(
    datasets: Iterable[Dataset], output: Path, on_progress: ProgressCallback | None, *, total: int
) -> tuple[Path, ...]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, ds in enumerate(datasets, start=1):
        path = output / f"IM_{index:06d}.dcm"
        dcmwrite(path, ds, enforce_file_format=True, little_endian=True, implicit_vr=False)
        paths.append(path)
        if on_progress is not None:
            on_progress(Progress("encode", index, total, "Writing DICOM images"))
    return tuple(paths)


def iter_image_datasets(
    image: ImageVolume,
    reference: ReferenceSeries,
    *,
    description: str | None = None,
    header_source: Dataset | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> Iterator[Dataset]:
    """Yield one series with per-image PET scales or a shared CT/MR scale.

    Reference acquisition metadata is required for MR/PET facts that cannot be
    reconstructed from NIfTI. Validation occurs when iteration starts; callers
    writing files must use staging to handle a failure in any later frame.
    Optional-code omissions are sent to ``on_warning`` once per distinct warning.
    """
    _validate_layout(image)
    if not reference.slices:
        raise ReferenceError("The reference series is empty.")
    modality = str(getattr(reference.first, "Modality", "")).upper()
    if modality not in _STORAGE:
        raise UnsupportedInputError(
            f"Unsupported scalar DICOM modality: {modality or 'missing'}.",
            hint="Use a classic CT, MR or PT reference series.",
        )
    shared_pixels = encode_pixels(image.data) if modality != "PT" else None
    # PET permits per-image slopes, but PixelRepresentation is series-wide.
    # Unsupported dtypes are diagnosed by the encoder, not by this comparison.
    pet_signed = False
    if modality == "PT" and image.data.dtype.kind in "buif":
        pet_signed = bool(np.any(image.data < 0))
    frame_indices = _matching_frames(image, reference)
    volume_sources = None
    if modality == "MR" and frame_indices is None:
        volume_sources, metadata_warnings = _reformatted_mr_sources(image, reference)
        if on_warning is not None:
            for message in metadata_warnings:
                on_warning(message)
    if modality == "PT" and frame_indices is None:
        raise ReferenceError(
            "PET FrameReferenceTime requires matching geometry and coherent reference timing.",
            hint="Use a matching reference acquisition and grid. NIfTI sample spacing alone "
            "does not establish effective PET reference times.",
        )
    base = _base_dataset(
        reference,
        sop_class=_STORAGE[modality],
        modality=modality,
        description=description,
        header_source=header_source,
    )
    for message in _series_profile(
        base, reference, frame_indices is not None or volume_sources is not None
    ):
        if on_warning is not None:
            on_warning(message)
    base.SamplesPerPixel = 1
    base.PhotometricInterpretation = "MONOCHROME2"
    base.BitsAllocated = base.BitsStored = 16
    base.HighBit = 15
    reported_warnings: set[str] = set()
    for index in range(image.timepoints * image.geometry.size[2]):
        t, z = divmod(index, image.geometry.size[2])
        if shared_pixels is None:
            # PET's scale belongs to the image instance. A bright slice or
            # timepoint must not reduce another plane's quantitative precision.
            encoded = encode_pixels(image.data[t, z], zero_intercept=True, force_signed=pet_signed)
            values = encoded.values
        else:
            encoded = shared_pixels
            values = encoded.values[t, z]
        source_index = frame_indices[index] if frame_indices is not None else None
        ds = _frame_dataset(base, image, reference, index, source_index)
        ds.PixelRepresentation = int(encoded.signed)
        ds.RescaleSlope = format_number_as_ds(encoded.slope)
        ds.RescaleIntercept = format_number_as_ds(encoded.intercept)
        ds.DerivationDescription = (
            "Converted from NIfTI real values. Reference supplies study "
            f"context. Maximum pixel encoding error: {encoded.max_error:.8g}."
        )
        source = reference.slices[source_index] if source_index is not None else reference.first
        if volume_sources is not None:
            source = volume_sources[t]
            _copy_fields(ds, source, ("AcquisitionDate", "AcquisitionTime"))
            if image.time_spacing is None or image.timepoints == 1:
                _copy_fields(ds, source, ("TemporalResolution",))
        for message in _profile(
            ds, source, image, source_index is not None or volume_sources is not None
        ):
            if on_warning is not None and message not in reported_warnings:
                on_warning(message)
                reported_warnings.add(message)
        ds.PixelData = values.tobytes(order="C")
        ds["PixelData"].VR = "OW"
        yield ds


def write_images(
    image: ImageVolume,
    reference: ReferenceSeries,
    output: Path,
    *,
    description: str | None = None,
    header_source: Dataset | None = None,
    on_progress: ProgressCallback | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    """Write one classic instance per plane; publication belongs to the pipeline."""
    _validate_layout(image)
    datasets = iter_image_datasets(
        image,
        reference,
        description=description,
        header_source=header_source,
        on_warning=on_warning,
    )
    return _save_frames(
        datasets, output, on_progress, total=image.timepoints * image.geometry.size[2]
    )
