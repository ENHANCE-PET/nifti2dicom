"""Geometry validation and interpolation in the common LPS millimeter frame."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import SimpleITK as sitk

from nifti2dicom.errors import GeometryError
from nifti2dicom.models import Geometry, ImageVolume


def validate_geometry(geometry: Geometry) -> None:
    """Reject nonphysical or sheared grids; oblique orthogonal grids are valid."""
    affine = np.asarray(geometry.affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise GeometryError("The spatial affine must be a finite 4 × 4 matrix.")
    if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise GeometryError("The spatial affine has an invalid homogeneous last row.")
    if len(geometry.size) != 3 or any(int(n) != n or n <= 0 for n in geometry.size):
        raise GeometryError("The spatial grid must have three positive integer dimensions.")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise GeometryError("The spatial affine has zero voxel spacing or is singular.")
    direction = affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4, rtol=0):
        raise GeometryError(
            "The spatial affine contains shear or nonorthogonal axes.",
            hint="Resample the source to an orthogonal voxel grid before conversion.",
        )


def _corners(geometry: Geometry) -> np.ndarray:
    """Physical corners of voxel extents (including half a voxel at each edge)."""
    indices = np.array(list(product(*[(-0.5, n - 0.5) for n in geometry.size])))
    return np.asarray(indices @ geometry.affine[:3, :3].T + geometry.affine[:3, 3])


def _overlaps(first: Geometry, second: Geometry) -> bool:
    # Separating-axis test for two oriented boxes also handles oblique thin slabs,
    # for which an axis-aligned bounding-box test can give a false intersection.
    a = first.affine[:3, :3] / np.asarray(first.spacing)
    b = second.affine[:3, :3] / np.asarray(second.spacing)
    axes = [*a.T, *b.T, *(np.cross(x, y) for x in a.T for y in b.T)]
    corners_a, corners_b = _corners(first), _corners(second)
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-8:
            continue
        projection_a, projection_b = corners_a @ (axis / length), corners_b @ (axis / length)
        if (
            projection_a.max() <= projection_b.min() + 1e-8
            or projection_b.max() <= projection_a.min() + 1e-8
        ):
            return False
    return True


def resample_to_reference(
    image: ImageVolume,
    geometry: Geometry,
    *,
    labels: bool = False,
) -> ImageVolume:
    """Interpolate each timepoint onto a grid using its physical coordinates.

    Nearest-neighbor interpolation retains label identities. Other inputs use
    floating-point linear interpolation. Space outside the source is zero.
    """
    validate_geometry(image.geometry)
    validate_geometry(geometry)
    expected = tuple(reversed(image.geometry.size))
    if image.data.ndim not in (4, 5) or image.data.shape[1:4] != expected:
        raise GeometryError("The image data dimensions disagree with its spatial geometry.")
    if image.data.shape[0] == 0 or not np.isfinite(image.data).all():
        raise GeometryError("Resampling requires nonempty, finite image data.")
    if image.geometry.size == geometry.size and np.allclose(
        image.geometry.affine,
        geometry.affine,
        atol=1e-8,
        rtol=0,
    ):
        return replace(image, geometry=geometry, data=image.data.copy())
    if not _overlaps(image.geometry, geometry):
        raise GeometryError(
            "The NIfTI and reference spatial fields of view do not overlap.",
            hint="Check that both inputs describe the same anatomy in the same physical frame.",
        )
    source_direction = image.geometry.affine[:3, :3] / np.asarray(image.geometry.spacing)
    target_direction = geometry.affine[:3, :3] / np.asarray(geometry.spacing)
    frames = []
    try:
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(list(geometry.size))
        resampler.SetOutputOrigin(tuple(geometry.affine[:3, 3]))
        resampler.SetOutputSpacing(geometry.spacing)
        resampler.SetOutputDirection(tuple(target_direction.ravel()))
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if labels else sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        for frame in image.data:
            dtype = frame.dtype.newbyteorder("=") if labels else np.dtype(np.float64)
            pixels = frame.astype(dtype, copy=False)
            source = sitk.GetImageFromArray(np.ascontiguousarray(pixels), isVector=frame.ndim == 4)
            source.SetOrigin(tuple(image.geometry.affine[:3, 3]))
            source.SetSpacing(image.geometry.spacing)
            source.SetDirection(tuple(source_direction.ravel()))
            resampler.SetOutputPixelType(source.GetPixelID())
            output = resampler.Execute(source)
            frames.append(sitk.GetArrayFromImage(output))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise GeometryError(
            "The image could not be resampled onto the reference grid.",
            details={"reason": str(exc)},
        ) from exc
    return replace(image, data=np.stack(frames), geometry=geometry)
