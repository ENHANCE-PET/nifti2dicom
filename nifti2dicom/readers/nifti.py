"""Read NIfTI arrays without losing spatial units, color, or temporal axes."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from nifti2dicom.errors import InputError, UnsupportedInputError
from nifti2dicom.geometry import validate_geometry
from nifti2dicom.models import Geometry, ImageKind, ImageVolume

_SPATIAL_SCALE = {"mm": 1.0, "meter": 1000.0, "micron": 0.001, "unknown": 1.0}
_TIME_SCALE = {"sec": 1.0, "msec": 0.001, "usec": 0.000001}


def read_nifti(path: str | Path, *, kind: ImageKind = "image") -> ImageVolume:
    """Load a NIfTI file and convert it to the shared physical-volume contract."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputError(f"NIfTI input does not exist or is not a file: {source}")
    try:
        loaded = nib.load(source)
    except (OSError, ValueError, TypeError, nib.filebasedimages.ImageFileError) as exc:
        raise InputError(
            f"Cannot read NIfTI input: {source.name}", details={"reason": str(exc)}
        ) from exc
    if not isinstance(loaded, (nib.Nifti1Image, nib.Nifti2Image)):
        raise UnsupportedInputError("The input must be a NIfTI-1 or NIfTI-2 image.")
    return nifti_to_volume(loaded, kind=kind, source=source)


def nifti_to_volume(
    loaded: nib.Nifti1Image | nib.Nifti2Image,
    *,
    kind: ImageKind = "image",
    source: Path | None = None,
) -> ImageVolume:
    """Return TZYX or TZYXC pixels and an XYZ-to-LPS-mm spatial affine.

    Three-dimensional spatial axes are reoriented by permutations and flips,
    without interpolation. A true 2D image keeps its original plane so that a
    sagittal or coronal image remains a single DICOM frame. The supplied image
    is not mutated, and in-memory images do not require temporary files.
    """
    if kind not in ("image", "seg", "rgb"):
        raise InputError(f"Unknown image kind: {kind!r}.")
    if not isinstance(loaded, (nib.Nifti1Image, nib.Nifti2Image)):
        raise UnsupportedInputError("The input must be a NIfTI-1 or NIfTI-2 image.")
    try:
        data = np.asanyarray(loaded.dataobj)
        affine = np.array(loaded.affine, dtype=np.float64, copy=True)
        spatial_unit, temporal_unit = loaded.header.get_xyzt_units()
    except (OSError, ValueError, TypeError, nib.filebasedimages.ImageFileError) as exc:
        name = source.name if source is not None else "in-memory image"
        raise InputError(f"Cannot read NIfTI input: {name}", details={"reason": str(exc)}) from exc

    messages: list[str] = []
    if spatial_unit not in _SPATIAL_SCALE:
        raise UnsupportedInputError(f"Unsupported NIfTI spatial unit: {spatial_unit}.")
    if spatial_unit == "unknown":
        messages.append("NIfTI spatial units are unknown; millimeters were assumed.")
    affine[:3] *= _SPATIAL_SCALE[spatial_unit]

    intent_code = int(loaded.header["intent_code"])
    if intent_code in range(1004, 1012) or (intent_code in (2003, 2004) and kind != "rgb"):
        raise UnsupportedInputError(
            "NIfTI vector, tensor, or geometric intent is unsupported for this conversion kind.",
            hint="Provide scalar image volumes or explicitly select RGB for three color channels.",
        )

    if data.dtype.fields is not None:
        if kind != "rgb" or data.dtype.names != ("R", "G", "B"):
            raise UnsupportedInputError("Structured NIfTI pixels require explicit RGB conversion.")
        data = np.stack([data[name] for name in ("R", "G", "B")], axis=-1)
    if data.dtype.kind not in "buif":
        raise UnsupportedInputError("Complex, vector, and tensor NIfTI pixels are unsupported.")
    if not np.isfinite(data).all():
        raise InputError("NIfTI pixels must be finite; NaN and infinity cannot be encoded.")
    if any(n <= 0 for n in data.shape):
        raise InputError("NIfTI input has an empty dimension.")

    if kind == "rgb":
        if data.ndim not in (3, 4) or data.shape[-1] != 3:
            raise UnsupportedInputError(
                "RGB requires a 2D or 3D spatial image with exactly three final channels.",
                hint="Use an array shaped (X, Y, 3) or (X, Y, Z, 3), or NIfTI RGB datatype.",
            )
        spatial_dimensions = data.ndim - 1
    else:
        if data.ndim not in (2, 3, 4):
            raise UnsupportedInputError("Scalar images require 2D, 3D, or 4D NIfTI data.")
        if kind == "seg" and data.ndim == 4:
            raise UnsupportedInputError("4D SEG input is unsupported; provide one 3D label map.")
        spatial_dimensions = min(data.ndim, 3)

    shape = (
        int(data.shape[0]),
        int(data.shape[1]),
        1 if spatial_dimensions == 2 else int(data.shape[2]),
    )
    validate_geometry(Geometry(affine, shape))
    if spatial_dimensions == 2:
        data = np.expand_dims(data, axis=2)
    else:
        orientation = nib.orientations.io_orientation(affine)
        transform = nib.orientations.ornt_transform(
            orientation,
            nib.orientations.axcodes2ornt(("R", "A", "S")),
        )
        data = nib.orientations.apply_orientation(data, transform)
        affine = affine @ nib.orientations.inv_ornt_aff(transform, shape)

    # NIfTI coordinates use RAS; DICOM and the shared Geometry contract use LPS.
    affine[:2] *= -1
    geometry = Geometry(affine, (int(data.shape[0]), int(data.shape[1]), int(data.shape[2])))
    validate_geometry(geometry)
    time_spacing = None
    if kind == "rgb":
        data = data.transpose(2, 1, 0, 3)[np.newaxis]
    elif data.ndim == 3:
        data = data.transpose(2, 1, 0)[np.newaxis]
    else:
        data = data.transpose(3, 2, 1, 0)
        if temporal_unit in _TIME_SCALE:
            time_spacing = float(loaded.header.get_zooms()[3]) * _TIME_SCALE[temporal_unit]
            if not np.isfinite(time_spacing) or time_spacing <= 0:
                raise InputError("Known NIfTI temporal spacing must be finite and positive.")
        elif temporal_unit != "unknown":
            raise UnsupportedInputError(
                f"NIfTI temporal unit {temporal_unit!r} does not describe time in seconds.",
                hint="Convert frequency or spectral axes to a supported scalar image first.",
            )
    return ImageVolume(
        np.ascontiguousarray(data),
        geometry,
        kind=kind,
        time_spacing=time_spacing,
        warnings=tuple(messages),
        source=source,
    )
