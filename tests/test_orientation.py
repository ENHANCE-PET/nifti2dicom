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
        """With identity affine, RAS→LPS should negate first two IPP components."""
        img = nib.load(str(sample_nifti_3d))
        _, ipp_list, _ = orient_nifti(img)
        # Identity affine: voxel (0,0,k) → RAS (0,0,k) → LPS (0,0,k)
        # The first two components should be negated (but 0 negated is still 0)
        # The Z (S→S) component should equal k
        for k in range(ipp_list.shape[0]):
            assert ipp_list[k, 0] == pytest.approx(0.0)
            assert ipp_list[k, 1] == pytest.approx(0.0)
            assert ipp_list[k, 2] == pytest.approx(float(k))
