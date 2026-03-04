"""Tests for nifti2dicom.orientation."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nifti2dicom.orientation import orient_nifti


class TestOrientNifti:
    def test_3d_output_shape(self, sample_nifti_3d: Path) -> None:
        img = nib.load(str(sample_nifti_3d))
        data, ipp_list, iop = orient_nifti(img)
        # Original shape (4, 4, 3) → oriented (3, 4, 4) = (slices, rows, cols)
        assert data.ndim == 3
        assert data.shape[0] == 3  # num_slices = Z dimension
        assert ipp_list.shape == (3, 3)
        assert iop.shape == (6,)

    def test_4d_output_shape(self, sample_nifti_4d: Path) -> None:
        img = nib.load(str(sample_nifti_4d))
        data, ipp_list, iop = orient_nifti(img)
        # Original (4, 4, 3, 2) → 3*2=6 slices after time collapse
        assert data.ndim == 3
        assert data.shape[0] == 6
        assert ipp_list.shape == (6, 3)

    def test_iop_unit_vectors(self, sample_nifti_3d: Path) -> None:
        img = nib.load(str(sample_nifti_3d))
        _, _, iop = orient_nifti(img)
        row_cos = iop[:3]
        col_cos = iop[3:]
        assert abs(np.linalg.norm(row_cos) - 1.0) < 1e-10
        assert abs(np.linalg.norm(col_cos) - 1.0) < 1e-10

    def test_ipp_varies_per_slice(self, sample_nifti_3d: Path) -> None:
        img = nib.load(str(sample_nifti_3d))
        _, ipp_list, _ = orient_nifti(img)
        # Each slice should have a different position
        assert not np.allclose(ipp_list[0], ipp_list[1])

    def test_vendor_param_triggers_warning(self, sample_nifti_3d: Path) -> None:
        img = nib.load(str(sample_nifti_3d))
        with pytest.warns(DeprecationWarning, match="vendor"):
            orient_nifti(img, vendor="ux")

    def test_identity_affine_lps(self, sample_nifti_3d: Path) -> None:
        """With identity affine (4x4x3), verify IPP and IOP in LPS."""
        img = nib.load(str(sample_nifti_3d))
        _, ipp_list, iop = orient_nifti(img)
        # No flip: output (s,0,0) maps to canonical (0,0,s) →
        # RAS (0,0,s) → LPS (0,0,s).
        for k in range(ipp_list.shape[0]):
            assert ipp_list[k, 0] == pytest.approx(0.0)
            assert ipp_list[k, 1] == pytest.approx(0.0)
            assert ipp_list[k, 2] == pytest.approx(float(k))
        # IOP: row (along c) = RAS +X → LPS -L = [-1,0,0]
        #      col (along r) = RAS +Y → LPS -P = [0,-1,0]
        np.testing.assert_allclose(iop, [-1, 0, 0, 0, -1, 0], atol=1e-10)
