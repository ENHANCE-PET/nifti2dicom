"""Binary and multilabel DICOM SEG with verified source-frame correspondence."""

from __future__ import annotations

import copy
from pathlib import Path

import highdicom as hd
import numpy as np
import pydicom
from pydicom.sr.coding import Code

from nifti2dicom.errors import GeometryError, LabelError, ReferenceError
from nifti2dicom.labels import SegmentationInfo
from nifti2dicom.models import ImageVolume, Progress, ProgressCallback, ReferenceSeries
from nifti2dicom.writers.codes import omit_incomplete_optional_codes


def check_foreground_extent(image: ImageVolume, reference: ReferenceSeries) -> None:
    """Reject masks whose foreground would be cropped by the target grid."""
    indices = np.argwhere(image.data[0] != 0)  # ZYX
    transform = np.linalg.inv(reference.geometry.affine) @ image.geometry.affine
    radius = np.abs(transform[:3, :3]).sum(axis=1) * 0.5
    for start in range(0, len(indices), 65536):
        xyz = indices[start : start + 65536, ::-1]
        target = xyz @ transform[:3, :3].T + transform[:3, 3]
        if np.any(target - radius < -0.5001) or np.any(
            target + radius > np.asarray(reference.geometry.size) - 0.4999
        ):
            raise GeometryError(
                "Some segmented voxels lie outside the reference scan.",
                hint=(
                    "Choose the DICOM series used to create this mask, "
                    "or correct its spatial alignment."
                ),
            )


def write_seg(
    image: ImageVolume,
    reference: ReferenceSeries,
    output: Path,
    *,
    segments: SegmentationInfo,
    description: str | None = None,
    on_progress: ProgressCallback | None = None,
    manufacturer: str = "nifti2dicom",
    manufacturer_model_name: str = "nifti2dicom",
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    from nifti2dicom import __version__

    if image.timepoints != 1 or reference.timepoints != 1:
        raise LabelError(
            "SEG currently requires one spatial volume.",
            hint="Convert one timepoint at a time with its matching reference series.",
        )
    if not getattr(reference.first, "FrameOfReferenceUID", None):
        raise ReferenceError(
            "The reference has no Frame of Reference UID for linking the segmentation.",
            hint="Use the original spatial DICOM image series.",
        )
    mask = image.data[0]
    encoded = np.zeros(mask.shape, dtype=np.uint16)
    for segment in segments.segments:
        selected = mask == segment.value
        if not selected.any():
            raise GeometryError(
                f"Segment {segment.value} disappeared when resampling to the reference grid.",
                hint="Use a finer reference grid or a mask aligned to this scan.",
            )
        encoded[selected] = segment.number

    algorithm = None
    if segments.algorithm_type != "MANUAL":
        if segments.algorithm_name is None or segments.algorithm_version is None:
            raise LabelError("Automatic segmentation requires an algorithm name and version.")
        algorithm = hd.AlgorithmIdentificationSequence(
            name=segments.algorithm_name,
            version=segments.algorithm_version,
            family=Code("SEGMENTATION", "99NIFTI2DICOM", "Segmentation algorithm"),
        )
    colors = [(232, 116, 97), (81, 160, 160), (224, 180, 82), (147, 119, 184), (111, 166, 110)]
    descriptions = [
        hd.seg.SegmentDescription(
            segment_number=s.number,
            segment_label=s.name,
            segmented_property_category=s.category,
            segmented_property_type=s.property_type,
            algorithm_type=segments.algorithm_type,
            algorithm_identification=algorithm,
            display_color=hd.color.CIELabColor.from_rgb(*colors[(s.number - 1) % len(colors)]),
        )
        for s in segments.segments
    ]
    sources = [copy.deepcopy(ds) for ds in reference.slices]
    # Empty type-2 attributes mean unknown, not fabricated patient facts.
    for source in sources:
        for keyword in (
            "PatientName",
            "PatientID",
            "PatientBirthDate",
            "PatientSex",
            "AccessionNumber",
            "StudyID",
            "StudyDate",
            "StudyTime",
        ):
            if keyword not in source:
                setattr(source, keyword, "")
    try:
        seg = hd.seg.Segmentation(
            source_images=sources,
            pixel_array=encoded,
            segmentation_type="BINARY",
            segment_descriptions=descriptions,
            series_instance_uid=hd.UID(),
            series_number=900,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=__version__,
            device_serial_number="nifti2dicom",
            series_description=description or "NIfTI segmentation",
            transfer_syntax_uid=pydicom.uid.ExplicitVRLittleEndian,
        )
    except (ValueError, AttributeError, TypeError) as exc:
        raise ReferenceError(
            f"Could not construct DICOM SEG from the supplied reference: {exc}",
            hint="Check that the reference is a complete, coherent original CT, MR or PET series.",
        ) from exc
    path = output / "segmentation.dcm"
    # These copied General Study sequences are optional (PS3.3 C.7.2.1).
    warnings = omit_incomplete_optional_codes(
        seg,
        (
            "ProcedureCodeSequence",
            "RequestingServiceCodeSequence",
            "ReasonForPerformedProcedureCodeSequence",
        ),
    )
    seg.SpecificCharacterSet = "ISO_IR 192"
    for item, segment in zip(seg.SegmentSequence, segments.segments, strict=True):
        item.SegmentDescription = f"Original NIfTI label value: {segment.value}"
    pydicom.dcmwrite(path, seg, enforce_file_format=True)
    # Validate the serialized object, including empty source planes and label identity.
    restored = hd.seg.segread(path)
    recovered = restored.get_pixels_by_source_instance(
        [ds.SOPInstanceUID for ds in sources],
        combine_segments=True,
        assert_missing_frames_are_empty=True,
    )
    if not np.array_equal(recovered, encoded):
        raise GeometryError("The written SEG failed its source-frame and label round-trip check.")
    if on_progress:
        on_progress(Progress("writing", 1, 1, "DICOM SEG"))
    return (path,), warnings
