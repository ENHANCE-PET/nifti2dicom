"""Interleaved RGB Secondary Capture with explicit image geometry."""

from __future__ import annotations

from pathlib import Path

from pydicom.dataset import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

from nifti2dicom.models import ImageVolume, ProgressCallback, ReferenceSeries
from nifti2dicom.pixels import encode_rgb_pixels
from nifti2dicom.writers.image import (
    _base_dataset,
    _frame_dataset,
    _matching_frames,
    _save_frames,
    _validate_layout,
)


def write_rgb(
    image: ImageVolume,
    reference: ReferenceSeries,
    output: Path,
    *,
    description: str | None = None,
    header_source: Dataset | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Path, ...]:
    """Write exact 8-bit RGB; callers must explicitly scale other color ranges."""
    _validate_layout(image, rgb=True)
    pixels = encode_rgb_pixels(image.data)
    base = _base_dataset(
        reference,
        sop_class=SecondaryCaptureImageStorage,
        modality="OT",
        description=description,
        header_source=header_source,
    )
    base.ConversionType = "WSD"
    base.SecondaryCaptureDeviceManufacturer = "nifti2dicom"
    base.DerivationDescription = (
        "RGB NIfTI converted to Secondary Capture; reference study context."
    )
    base.PhotometricInterpretation = "RGB"
    base.SamplesPerPixel = 3
    base.PlanarConfiguration = 0
    base.BitsAllocated = base.BitsStored = 8
    base.HighBit = 7
    base.PixelRepresentation = 0
    frame_indices = _matching_frames(image, reference)
    datasets = []
    for index in range(image.timepoints * image.geometry.size[2]):
        source_index = frame_indices[index] if frame_indices is not None else None
        ds = _frame_dataset(base, image, reference, index, source_index)
        t, z = divmod(index, image.geometry.size[2])
        ds.PixelData = pixels[t, z].tobytes(order="C")
        ds["PixelData"].VR = "OB"
        datasets.append(ds)
    return _save_frames(datasets, output, on_progress, total=len(datasets))
