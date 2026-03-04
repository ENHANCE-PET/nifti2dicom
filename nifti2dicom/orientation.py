"""Affine-based NIfTI → DICOM orientation.

Replaces the old vendor-specific ``np.flip`` hacks with deterministic
affine decomposition.  **No data flips** — only a transpose to
(slice, row, col).  IPP and IOP are derived from a single composed
affine so data and geometry can never diverge.

1. ``nib.as_closest_canonical(img)`` → data in RAS+ order
2. ``data.transpose(2, 1, 0)`` → (slice, row, col)
3. Compose ``_RAS_TO_LPS @ canonical_affine @ _TRANSPOSE`` once
4. IPP / IOP read directly from that composed affine

The ``vendor`` parameter is accepted but ignored (deprecation warning).
"""

from __future__ import annotations

import warnings

import nibabel as nib
import numpy as np

# RAS→LPS sign flip: negate R→L (axis 0) and A→P (axis 1)
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])

# Maps output indices (s, r, c) back to canonical (i, j, k):
#   i = c,  j = r,  k = s   (inverse of transpose(2,1,0))
_TRANSPOSE = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
], dtype=float)


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

    # Step 2 — transpose to (slice, row, col).  No flip needed;
    # the composed affine handles the RAS→LPS sign change.
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)
        nz_slices = data.shape[0]
        n_timepoints = 1
    elif data.ndim == 4:
        # (X, Y, Z, T) → (T, Z, Y, X) → collapse to (T*Z, Y, X)
        data = data.transpose(3, 2, 1, 0)
        n_timepoints, nz_slices = data.shape[0], data.shape[1]
        data = data.reshape(
            n_timepoints * nz_slices, data.shape[2], data.shape[3],
        )
    else:
        raise ValueError(f"Unsupported NIfTI dimensionality: {data.ndim}")

    # Step 3 — single composed LPS affine for the output index space
    #   output (s,r,c) → canonical (i,j,k) → RAS mm → LPS mm
    lps_affine = _RAS_TO_LPS @ affine @ _TRANSPOSE

    # Per-slice Image Position (Patient)
    num_slices = data.shape[0]
    z_indices = np.tile(np.arange(nz_slices), n_timepoints)
    coords = np.zeros((num_slices, 4))
    coords[:, 0] = z_indices
    coords[:, 3] = 1.0
    ipp_list = (lps_affine @ coords.T).T[:, :3]

    # Image Orientation (Patient)
    # Row direction = along increasing column (col 2 of lps_affine)
    # Col direction = along increasing row    (col 1 of lps_affine)
    row_cosine = lps_affine[:3, 2]
    row_cosine = row_cosine / np.linalg.norm(row_cosine)
    col_cosine = lps_affine[:3, 1]
    col_cosine = col_cosine / np.linalg.norm(col_cosine)
    iop = np.concatenate([row_cosine, col_cosine])

    return data, ipp_list, iop
