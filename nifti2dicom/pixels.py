"""Quantitative 16-bit pixels with explicit, measurable reconstruction error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydicom.valuerep import format_number_as_ds

from nifti2dicom.errors import PixelEncodingError


@dataclass(frozen=True)
class EncodedPixels:
    values: np.ndarray
    slope: float
    intercept: float
    max_error: float
    signed: bool


def encode_rgb_pixels(data: np.ndarray) -> np.ndarray:
    """Validate exact 8-bit color without silently scaling or wrapping channels."""
    data = np.asarray(data)
    if (
        data.size == 0
        or data.dtype.kind not in "buif"
        or not np.isfinite(data).all()
        or data.min() < 0
        or data.max() > 255
        or not np.equal(data, np.rint(data)).all()
    ):
        raise PixelEncodingError(
            "RGB pixels must be finite integer values from 0 to 255.",
            hint="Explicitly scale color channels to 8-bit RGB first.",
        )
    return data.astype(np.uint8, copy=False)


def encode_fixed_pixels(
    data: np.ndarray, slope: float, intercept: float, pixel_representation: int
) -> np.ndarray:
    """Invert an explicitly supplied scale without clipping or integer wrap.

    This is intended for legacy callers that keep their own rescale tags.
    New image writers should use ``encode_pixels`` to choose a fitting scale.
    """
    data = np.asarray(data)
    if data.size == 0 or data.dtype.kind not in "buif" or not np.isfinite(data).all():
        raise PixelEncodingError("Pixels must be a nonempty array of finite real numbers.")
    try:
        slope, intercept = float(slope), float(intercept)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PixelEncodingError("Rescale slope and intercept must be finite numbers.") from exc
    if not np.isfinite(slope) or slope == 0 or not np.isfinite(intercept):
        raise PixelEncodingError("Rescale parameters must be finite, with a nonzero slope.")
    if pixel_representation not in (0, 1):
        raise PixelEncodingError("Pixel representation must be 0 (unsigned) or 1 (signed).")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.rint((data.astype(np.float64) - intercept) / slope)
    low, high = (-32768, 32767) if pixel_representation else (0, 65535)
    if not np.isfinite(rounded).all() or rounded.min() < low or rounded.max() > high:
        raise PixelEncodingError(
            "The fixed rescale parameters overflow 16-bit pixel storage.",
            hint="Use automatic pixel encoding to choose a fitting scale.",
        )
    return rounded.astype("<i2" if pixel_representation else "<u2")


def encode_pixels(
    data: np.ndarray, *, zero_intercept: bool = False, force_signed: bool = False
) -> EncodedPixels:
    """Encode real values, never reference stored values, without clipping.

    Integers fitting the selected 16-bit storage are exact. Otherwise a
    linear transform uses nearest rounding. ``max_error`` is measured after
    rounding rescale parameters to DICOM's 16-character decimal strings.
    PET uses ``zero_intercept=True`` to satisfy its Image Module and
    ``force_signed=True`` on every plane if any value in its series is negative.
    """
    data = np.asarray(data)
    if data.size == 0 or data.dtype.kind not in "buif" or not np.isfinite(data).all():
        raise PixelEncodingError(
            "Pixels must be a nonempty array of finite real numbers.",
            hint="Remove NaN/Inf values and unsupported pixel types.",
        )
    real = data.astype(np.float64)
    if not np.isfinite(real).all():
        raise PixelEncodingError("Pixel values exceed the supported numeric range.")
    low, high = float(real.min()), float(real.max())
    integer = np.equal(real, np.rint(real)).all()
    signed = force_signed or low < 0
    lower, upper = (-32768, 32767) if signed else (0, 65535)
    intercept, slope = 0.0, 1.0
    if integer and lower <= low <= high <= upper:
        pass
    elif zero_intercept:
        # Zero-only inputs already took the exact-integer path. A zero scale
        # here is underflow, not a reason to silently replace the scale by one.
        slope = max(high / (32767 if signed else 65535), -low / 32768)
    else:
        signed = force_signed
        intercept = low
        if low != high:
            # Division before subtraction avoids overflowing high - low.
            slope = high / 65535 - low / 65535
            if integer and slope <= 1:
                slope = 1.0
            if signed:
                intercept += 32768 * slope
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
        raise PixelEncodingError(
            "Pixel magnitudes are too small or too large for finite DICOM rescale parameters.",
            hint="Rescale extreme input magnitudes before conversion, keeping reference Units "
            "consistent with the rescaled values.",
        )
    slope = float(format_number_as_ds(slope))
    intercept = float(format_number_as_ds(intercept))
    # Divide before subtraction to avoid overflowing a wide signed range.
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = (real - intercept) / slope
        if not np.isfinite(normalized).all():
            normalized = real / slope - intercept / slope
        rounded = np.rint(normalized)
    lower, upper = (-32768, 32767) if signed else (0, 65535)
    if not np.isfinite(rounded).all() or rounded.min() < lower or rounded.max() > upper:
        raise PixelEncodingError(
            "Pixel range exceeds 16-bit storage after decimal scaling.",
            hint="Rescale extreme input magnitudes before conversion.",
        )
    values = rounded.astype("<i2" if signed else "<u2")
    with np.errstate(over="ignore", invalid="ignore"):
        reconstructed = values.astype(np.float64) * slope + intercept
    if not np.isfinite(reconstructed).all():
        raise PixelEncodingError("Reconstructed pixel values exceed the supported numeric range.")
    # Long-double comparison also includes precision lost when accepting int64 input.
    error = np.max(np.abs(reconstructed.astype(np.longdouble) - data.astype(np.longdouble)))
    return EncodedPixels(values, slope, intercept, float(error), signed)
