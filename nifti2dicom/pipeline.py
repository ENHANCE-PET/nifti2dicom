"""One conversion workflow shared by the Python API and command line."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from nifti2dicom.errors import AmbiguousReferenceError, InputError, OutputError
from nifti2dicom.inference import default_output, infer_kind
from nifti2dicom.labels import prepare_segments, read_labels
from nifti2dicom.models import (
    ConversionResult,
    ImageVolume,
    InspectionResult,
    Progress,
    ProgressCallback,
    ReferenceSeries,
)
from nifti2dicom.pixels import encode_rgb_pixels
from nifti2dicom.publication import staged_output, validate_output


@dataclass(frozen=True)
class Inputs:
    image: ImageVolume
    reference: ReferenceSeries
    labels: dict[str, Any]
    warnings: tuple[str, ...]


def _read_inputs(
    nifti: Path,
    reference: Path,
    *,
    kind: str,
    series_uid: str | None,
    labels: str | Path | dict[str, Any] | None,
) -> Inputs:
    from nifti2dicom.readers.dicom import read_reference
    from nifti2dicom.readers.nifti import read_nifti

    if not nifti.is_file():
        raise InputError(
            f"NIfTI file does not exist or is not a file: {nifti}",
            hint="Choose a readable .nii or .nii.gz image.",
        )
    definitions = read_labels(labels)
    resolved = infer_kind(nifti, kind, has_labels=labels is not None)
    image = read_nifti(nifti, kind=resolved)
    if resolved == "rgb":
        encode_rgb_pixels(image.data)
    source = read_reference(reference, series_uid=series_uid)
    warnings = list(image.warnings + source.warnings)
    if image.timepoints > 1 and image.time_spacing is None:
        warnings.append(
            "Timepoint order is preserved, but acquisition timing is unknown in the NIfTI header."
        )
    modality = str(getattr(source.first, "Modality", ""))
    if modality == "MR" and resolved == "image":
        warnings.append(
            "MR output uses a standard-extended rescale mapping to preserve values. "
            "The receiving viewer must honor RescaleSlope and RescaleIntercept."
        )
    if modality == "CT" and resolved == "image" and image.timepoints > 1:
        warnings.append(
            "4D CT uses standard-extended temporal attributes. "
            "Confirm that the receiving viewer groups timepoints correctly."
        )
    if modality == "PT" and resolved == "image":
        warnings.append(
            "PET values are interpreted in the reference Units; "
            "NIfTI cannot independently verify activity/SUV units."
        )
    return Inputs(image, source, definitions, tuple(warnings))


def inspect_inputs(
    nifti: Path,
    reference: Path,
    *,
    kind: str,
    series_uid: str | None,
    labels: str | Path | dict[str, Any] | None,
) -> InspectionResult:
    inputs = _read_inputs(nifti, reference, kind=kind, series_uid=series_uid, labels=labels)
    warnings = list(inputs.warnings)
    if inputs.image.kind == "seg":
        if "algorithm" not in inputs.labels:
            warnings.append(
                "SEG export needs --algorithm-type and, for model output, "
                "algorithm name and version."
            )
    return InspectionResult(
        inputs.image.kind,
        tuple(inputs.image.data.shape),
        inputs.image.geometry.spacing,
        inputs.image.timepoints,
        str(getattr(inputs.reference.first, "Modality", "OT")),
        str(inputs.reference.first.SeriesInstanceUID),
        tuple(warnings),
    )


def _verify_files(files: tuple[Path, ...]) -> None:
    if not files:
        raise OutputError("The converter produced no DICOM files.")
    uids = set()
    series = set()
    for path in files:
        try:
            ds = pydicom.dcmread(path)
            pixels = ds.pixel_array
            if pixels.size == 0 or str(ds.SOPInstanceUID) in uids:
                raise ValueError("Empty pixels or duplicate instance identity")
            if ds.SOPInstanceUID != ds.file_meta.MediaStorageSOPInstanceUID:
                raise ValueError("File and dataset identities differ")
            uids.add(str(ds.SOPInstanceUID))
            series.add(str(ds.SeriesInstanceUID))
        except Exception as exc:
            raise OutputError(
                f"Output validation failed for {path.name}: {exc}",
                hint="No output was published. Report this error with the input metadata.",
            ) from exc
    if len(series) != 1:
        raise OutputError("Output validation found inconsistent series identities.")


def run_conversion(
    nifti: Path,
    reference: Path,
    output: Path | None,
    *,
    kind: str,
    geometry: str,
    series_uid: str | None,
    labels: str | Path | dict[str, Any] | None,
    description: str | None,
    overwrite: bool,
    on_progress: ProgressCallback | None,
    header_source: Path | None,
    algorithm_type: str | None,
    algorithm_name: str | None,
    algorithm_version: str | None,
    manufacturer: str,
    manufacturer_model_name: str,
) -> ConversionResult:
    from nifti2dicom.geometry import resample_to_reference
    from nifti2dicom.readers.dicom import read_reference
    from nifti2dicom.writers.image import write_images
    from nifti2dicom.writers.rgb import write_rgb
    from nifti2dicom.writers.seg import check_foreground_extent, write_seg

    if geometry not in {"native", "reference"}:
        raise InputError(
            f"Unknown geometry policy '{geometry}'.", hint="Choose native or reference."
        )
    if description is not None and len(description) > 64:
        raise InputError("Series descriptions must be at most 64 characters.")
    target = output if output is not None else default_output(nifti)
    protected = [nifti, reference]
    if header_source is not None:
        protected.append(header_source)
    if isinstance(labels, (str, Path)):
        protected.append(Path(labels))
    validate_output(target, protected, overwrite=overwrite)
    if on_progress:
        on_progress(Progress("reading", message="Inspecting NIfTI and reference series"))
    inputs = _read_inputs(nifti, reference, kind=kind, series_uid=series_uid, labels=labels)
    image, source = inputs.image, inputs.reference
    warnings = list(inputs.warnings)
    header_reference = None
    try:
        if header_source is not None:
            header_reference = read_reference(header_source)
    except AmbiguousReferenceError as exc:
        raise AmbiguousReferenceError(
            "The separate header source contains several DICOM series.",
            hint=(
                "Pass a single header DICOM file or a folder containing one series to --header-dir."
            ),
            details=exc.details,
        ) from exc
    # Discovery may traverse symbolic links into the output directory. Protect
    # the actual selected files, not just the directory supplied by the caller.
    protected.extend(source.paths)
    if header_reference is not None:
        protected.extend(header_reference.paths)
    validate_output(target, protected, overwrite=overwrite)
    header = header_reference.first if header_reference is not None else None
    segments = None
    if image.kind == "seg":
        if header is not None:
            raise InputError(
                "A separate header source is not supported for SEG.",
                hint="Use the original reference series to preserve source identity.",
            )
        segments = prepare_segments(
            image.data,
            inputs.labels,
            algorithm_type=algorithm_type,
            algorithm_name=algorithm_name,
            algorithm_version=algorithm_version,
        )
        warnings.extend(segments.warnings)
        check_foreground_extent(image, source)
    if geometry == "reference" or image.kind == "seg":
        if on_progress:
            on_progress(
                Progress("resampling", message="Matching the reference grid in patient space")
            )
        image = resample_to_reference(image, source.geometry, labels=image.kind == "seg")
        if image.kind == "rgb":
            image = replace(image, data=np.rint(image.data))
            warnings.append("Interpolated RGB channels were rounded to the nearest 8-bit value.")
        warnings.extend(w for w in image.warnings if w not in warnings)
    if on_progress:
        on_progress(Progress("writing", message="Encoding DICOM"))
    with staged_output(target, overwrite=overwrite) as staging:
        if segments is not None:
            files, writer_warnings = write_seg(
                image,
                source,
                staging,
                segments=segments,
                description=description,
                on_progress=on_progress,
                manufacturer=manufacturer,
                manufacturer_model_name=manufacturer_model_name,
            )
            warnings.extend(writer_warnings)
        else:
            writer = (
                write_rgb
                if image.kind == "rgb"
                else partial(write_images, on_warning=warnings.append)
            )
            files = writer(
                image,
                source,
                staging,
                description=description,
                header_source=header,
                on_progress=on_progress,
            )
        if on_progress:
            on_progress(Progress("validating", message="Reading back the written DICOM files"))
        _verify_files(files)
        output_files = tuple(target / file.relative_to(staging) for file in files)
        result = ConversionResult(
            target,
            output_files,
            image.kind,
            tuple(dict.fromkeys(warnings)),
            segments.mapping if segments else {},
        )
        manifest = result.to_dict()
        manifest["reference_series_uid"] = str(source.first.SeriesInstanceUID)
        manifest["geometry"] = {
            "policy": "reference" if segments else geometry,
            "affine_lps_mm": image.geometry.affine.tolist(),
        }
        manifest["shape_tzyx"] = list(image.data.shape)
        with (staging / "conversion.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
    if on_progress:
        on_progress(Progress("complete", len(result.files), len(result.files), str(target)))
    return result
