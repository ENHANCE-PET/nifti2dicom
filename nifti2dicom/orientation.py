"""Compatibility view of the shared NIfTI geometry reader."""

from __future__ import annotations

import warnings

import nibabel as nib
import numpy as np

from nifti2dicom.readers.nifti import nifti_to_volume


def orient_nifti(
    img: nib.Nifti1Image | nib.Nifti2Image,
    *,
    vendor: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened slices, LPS positions and orientation for legacy callers."""
    if vendor is not None:
        warnings.warn(
            "The vendor parameter is ignored; the affine determines orientation.",
            DeprecationWarning,
            stacklevel=2,
        )
    volume = nifti_to_volume(img)
    _, nz, rows, cols = volume.data.shape
    pixels = volume.data.reshape(-1, rows, cols)
    positions = np.tile(
        np.array([volume.geometry.position(z) for z in range(nz)]),
        (volume.timepoints, 1),
    )
    return pixels, positions, volume.geometry.iop
