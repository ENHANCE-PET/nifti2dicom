"""Affine-based NIfTI → DICOM orientation.

Replaces the old vendor-specific ``np.flip`` hacks with deterministic
affine decomposition:

1. ``nib.as_closest_canonical(img)`` → data in RAS+ order
2. ``np.flip(data, axis=(0, 1))`` → RAS → LPS
3. ``data.transpose(2, 1, 0)`` → (slice, row, col)
4. Per-slice IPP from ``affine @ [0, 0, k, 1]`` with RAS→LPS negation
5. IOP from affine column vectors with RAS→LPS negation

The ``vendor`` parameter is accepted but ignored (deprecation warning).
"""

from __future__ import annotations

import warnings

import nibabel as nib
import numpy as np

# RAS→LPS sign flip: negate R→L (axis 0) and A→P (axis 1)
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def orient_nifti(
    img: nib.Nifti1Image,
    *,
    vendor: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorient a NIfTI image for DICOM slice writing.

    Parameters
    ----------
    img : nib.Nifti1Image
        Loaded NIfTI image.
    vendor : str, optional
        Accepted but **ignored**. Orientation is now computed from the
        affine; the vendor-specific flips are gone. Passing this triggers
        a deprecation warning.

    Returns
    -------
    data : np.ndarray
        Pixel data shaped ``(num_slices, rows, cols)`` (3-D) or
        ``(num_slices, rows, cols)`` for each volume concatenated along
        the slice axis (4-D images).
    ipp_list : np.ndarray
        Array of shape ``(num_slices, 3)`` with Image Position (Patient)
        values in LPS for each slice.
    iop : np.ndarray
        Flat 6-element array for Image Orientation (Patient) in LPS.
    """
    if vendor is not None:
        warnings.warn(
            "The 'vendor' parameter is deprecated and ignored. "
            "Orientation is now computed from the NIfTI affine.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Step 1 — canonical RAS+ ordering
    canonical = nib.as_closest_canonical(img)
    data = np.asarray(canonical.dataobj)
    affine = canonical.affine

    # Step 2 — RAS → LPS (flip first two spatial axes)
    data = np.flip(data, axis=(0, 1))

    # Step 3 — transpose to (slice, row, col, ...) for DICOM
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)
    elif data.ndim == 4:
        # (X, Y, Z, T) → (Z, Y, X, T) → collapse time into slice axis
        data = data.transpose(2, 1, 0, 3)
        nz, ny, nx, nt = data.shape
        data = data.reshape(nz * nt, ny, nx)
    else:
        raise ValueError(f"Unsupported NIfTI dimensionality: {data.ndim}")

    # Step 4 — compute LPS affine for IPP / IOP
    lps_affine = _RAS_TO_LPS @ affine

    # Per-slice Image Position (Patient)
    num_slices = data.shape[0]
    k_indices = np.arange(num_slices)
    # Build homogeneous coordinates for (0, 0, k) voxels
    coords = np.zeros((num_slices, 4))
    coords[:, 2] = k_indices
    coords[:, 3] = 1.0
    ipp_list = (lps_affine @ coords.T).T[:, :3]

    # Step 5 — Image Orientation (Patient) from column vectors
    row_cosine = lps_affine[:3, 0]
    row_cosine = row_cosine / np.linalg.norm(row_cosine)
    col_cosine = lps_affine[:3, 1]
    col_cosine = col_cosine / np.linalg.norm(col_cosine)
    iop = np.concatenate([row_cosine, col_cosine])

    return data, ipp_list, iop
