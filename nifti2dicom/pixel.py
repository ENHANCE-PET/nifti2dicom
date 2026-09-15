"""Historical pixel helpers, delegated to the shared quantitative encoder."""

from __future__ import annotations

import numpy as np

from nifti2dicom.pixels import encode_fixed_pixels, encode_pixels


def normalize_for_dicom(
    data: np.ndarray,
    rescale_slope: float,
    rescale_intercept: float,
    pixel_representation: int,
) -> np.ndarray:
    """Invert a specified rescale, round nearest, and reject overflow."""
    return encode_fixed_pixels(data, rescale_slope, rescale_intercept, pixel_representation)


def encode_pixel_data(arr: np.ndarray) -> bytes:
    """Encode native or nonnative integer arrays explicitly as little endian."""
    from nifti2dicom.errors import PixelEncodingError

    arr = np.asarray(arr)
    if arr.dtype.kind not in "ui":
        raise PixelEncodingError("Stored pixel arrays must have an integer dtype.")
    little = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(little).tobytes()


def normalize_pt_dynamic_range(
    data: np.ndarray,
    rescale_slope: float,
    rescale_intercept: float,
) -> tuple[np.ndarray, float, float]:
    """Encode already-real PET values; reference scaling must not be applied twice."""
    encoded = encode_pixels(data, zero_intercept=True)
    return encoded.values, encoded.slope, encoded.intercept
