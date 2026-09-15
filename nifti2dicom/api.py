"""Public conversion API. All paths accept strings or pathlib.Path objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nifti2dicom.models import ConversionResult, InspectionResult, ProgressCallback


def convert(
    nifti: str | Path,
    reference: str | Path,
    output: str | Path | None = None,
    *,
    kind: str = "auto",
    geometry: str = "native",
    series_uid: str | None = None,
    labels: str | Path | dict[str, Any] | None = None,
    description: str | None = None,
    overwrite: bool = False,
    on_progress: ProgressCallback | None = None,
    header_source: str | Path | None = None,
    algorithm_type: str | None = None,
    algorithm_name: str | None = None,
    algorithm_version: str | None = None,
    manufacturer: str = "nifti2dicom",
    manufacturer_model_name: str = "nifti2dicom",
) -> ConversionResult:
    """Convert scalar, label or RGB NIfTI data using a DICOM reference.

    The default preserves the NIfTI grid. ``geometry='reference'`` resamples
    into the reference grid; segmentations always use the reference grid.
    For SEG, describe how the mask was created with ``algorithm_type`` and,
    for automatic/semi-automatic masks, ``algorithm_name`` and ``algorithm_version``.
    Expected problems raise :class:`nifti2dicom.errors.ConversionError`.
    The library prints nothing; use ``on_progress`` to receive progress events.
    """
    from nifti2dicom.pipeline import run_conversion

    return run_conversion(
        Path(nifti),
        Path(reference),
        Path(output) if output is not None else None,
        kind=kind,
        geometry=geometry,
        series_uid=series_uid,
        labels=labels,
        description=description,
        overwrite=overwrite,
        on_progress=on_progress,
        header_source=Path(header_source) if header_source is not None else None,
        algorithm_type=algorithm_type,
        algorithm_name=algorithm_name,
        algorithm_version=algorithm_version,
        manufacturer=manufacturer,
        manufacturer_model_name=manufacturer_model_name,
    )


def inspect(
    nifti: str | Path,
    reference: str | Path,
    *,
    kind: str = "auto",
    series_uid: str | None = None,
    labels: str | Path | dict[str, Any] | None = None,
) -> InspectionResult:
    """Inspect supported input meaning and reference geometry without writing files."""
    from nifti2dicom.pipeline import inspect_inputs

    return inspect_inputs(
        Path(nifti), Path(reference), kind=kind, series_uid=series_uid, labels=labels
    )
