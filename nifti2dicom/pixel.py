"""Pixel data normalization and encoding to DICOM-compatible bytes.

Fixes the old bug where ``PixelData = numpy_array`` was assigned directly
(pydicom needs raw bytes, not a numpy array).
"""

from __future__ import annotations

import numpy as np

from nifti2dicom.exceptions import PixelEncodingError


def normalize_for_dicom(
    data: np.ndarray,
    rescale_slope: float,
    rescale_intercept: float,
    pixel_representation: int,
) -> np.ndarray:
    """Apply inverse rescale transform and cast to DICOM integer type.

    DICOM stored values are related to real values by::

        real_value = stored_value * slope + intercept

    So to store we invert::

        stored_value = (real_value - intercept) / slope

    Parameters
    ----------
    data : np.ndarray
        Real-valued pixel data (typically float64 from nibabel).
    rescale_slope, rescale_intercept : float
        DICOM rescale parameters from the reference slice.
    pixel_representation : int
        0 = unsigned (uint16), 1 = signed (int16).

    Returns
    -------
    np.ndarray
        Integer array ready for byte encoding.
    """
    dtype = np.int16 if pixel_representation == 1 else np.uint16

    if rescale_slope == 0:
        raise PixelEncodingError("RescaleSlope is zero — cannot encode pixel data.")

    stored = (data - rescale_intercept) / rescale_slope
    return np.clip(stored, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def encode_pixel_data(arr: np.ndarray) -> bytes:
    """Encode a 2-D integer array to raw bytes for DICOM PixelData.

    Ensures the array is contiguous and returns ``.tobytes()``.
    """
    return np.ascontiguousarray(arr).tobytes()


def normalize_pt_dynamic_range(
    data: np.ndarray,
    rescale_slope: float,
    rescale_intercept: float,
) -> tuple[np.ndarray, float, float]:
    """Handle PET images whose dynamic range exceeds uint16.

    If ``max(stored_value) > 65535``, rescale so it fits in uint16 and
    return updated slope/intercept.

    Returns
    -------
    (stored_array, new_slope, new_intercept)
    """
    max_val = np.max(data)
    if max_val <= 0:
        return data.astype(np.uint16), rescale_slope, rescale_intercept

    if max_val > 65535:
        scale = max_val / 65535.0
        new_slope = rescale_slope * scale
        stored = (data / scale).astype(np.uint16)
        return stored, new_slope, 0.0

    stored = (data - rescale_intercept) / rescale_slope if rescale_slope != 0 else data
    return np.clip(stored, 0, 65535).astype(np.uint16), rescale_slope, rescale_intercept
