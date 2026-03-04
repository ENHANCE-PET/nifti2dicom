"""Tests for nifti2dicom.pixel."""

from __future__ import annotations

import numpy as np
import pytest

from nifti2dicom.exceptions import PixelEncodingError
from nifti2dicom.pixel import encode_pixel_data, normalize_for_dicom, normalize_pt_dynamic_range


class TestNormalizeForDicom:
    def test_identity_transform(self) -> None:
        data = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = normalize_for_dicom(data, rescale_slope=1.0, rescale_intercept=0.0, pixel_representation=1)
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, data.astype(np.int16))

    def test_with_slope_intercept(self) -> None:
        data = np.array([[10.0, 20.0]])
        result = normalize_for_dicom(data, rescale_slope=2.0, rescale_intercept=5.0, pixel_representation=0)
        # stored = (data - 5) / 2 = [2.5, 7.5] → clipped and cast to uint16
        assert result.dtype == np.uint16
        expected = np.array([[2, 7]], dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)

    def test_zero_slope_raises(self) -> None:
        data = np.array([[1.0]])
        with pytest.raises(PixelEncodingError, match="zero"):
            normalize_for_dicom(data, rescale_slope=0.0, rescale_intercept=0.0, pixel_representation=1)

    def test_clipping_negative_to_unsigned(self) -> None:
        data = np.array([[-100.0]])
        result = normalize_for_dicom(data, rescale_slope=1.0, rescale_intercept=0.0, pixel_representation=0)
        assert result[0, 0] == 0  # clipped to uint16 min


class TestEncodePixelData:
    def test_returns_bytes(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.int16)
        result = encode_pixel_data(arr)
        assert isinstance(result, bytes)
        assert len(result) == 4 * 2  # 4 elements × 2 bytes each

    def test_roundtrip(self) -> None:
        arr = np.array([[100, 200], [300, 400]], dtype=np.int16)
        raw = encode_pixel_data(arr)
        recovered = np.frombuffer(raw, dtype=np.int16).reshape(2, 2)
        np.testing.assert_array_equal(arr, recovered)


class TestNormalizePtDynamicRange:
    def test_no_rescale_needed(self) -> None:
        data = np.array([[100.0, 200.0]])
        stored, slope, intercept = normalize_pt_dynamic_range(data, 1.0, 0.0)
        assert stored.dtype == np.uint16
        assert slope == 1.0

    def test_rescale_when_exceeds_uint16(self) -> None:
        data = np.array([[0.0, 100000.0]])
        stored, slope, intercept = normalize_pt_dynamic_range(data, 1.0, 0.0)
        assert stored.max() <= 65535
        assert slope > 1.0
        assert intercept == 0.0
