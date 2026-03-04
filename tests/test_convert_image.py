"""Tests for nifti2dicom.convert_image end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest

from nifti2dicom.exceptions import ShapeMismatchError


class TestConvertNiftiToDicom:
    def test_shape_mismatch_raises(self, sample_dicom_dir: Path, tmp_path: Path) -> None:
        """A NIfTI whose oriented shape doesn't match the DICOM series should raise."""
        import nibabel as nib
        import numpy as np

        # Create a NIfTI with wrong dimensions (10x10x10 vs expected 4x4x3)
        data = np.zeros((10, 10, 10), dtype=np.float64)
        img = nib.Nifti1Image(data, np.eye(4))
        nifti_path = tmp_path / "wrong_shape.nii.gz"
        nib.save(img, str(nifti_path))

        from nifti2dicom.convert_image import convert_nifti_to_dicom

        with pytest.raises(ShapeMismatchError):
            convert_nifti_to_dicom(
                ref_dir=sample_dicom_dir,
                nifti_path=nifti_path,
                output_dir=tmp_path / "output",
            )

    def test_successful_conversion(self, sample_dicom_dir: Path, sample_nifti_3d: Path, tmp_path: Path) -> None:
        """End-to-end: convert a matching NIfTI and verify output DICOM files."""
        from nifti2dicom.convert_image import convert_nifti_to_dicom

        output = tmp_path / "output"
        convert_nifti_to_dicom(
            ref_dir=sample_dicom_dir,
            nifti_path=sample_nifti_3d,
            output_dir=output,
            series_description="test_conversion",
        )

        assert output.exists()
        dcm_files = list(output.glob("*"))
        assert len(dcm_files) == 3

        # Verify each is valid DICOM
        import pydicom
        for f in dcm_files:
            ds = pydicom.dcmread(str(f))
            assert ds.Rows == 4
            assert ds.Columns == 4
            assert "test_conversion" in ds.SeriesDescription

    def test_skip_existing_without_force(self, sample_dicom_dir: Path, sample_nifti_3d: Path, tmp_path: Path) -> None:
        """If output dir exists and force=False, should skip without error."""
        from nifti2dicom.convert_image import convert_nifti_to_dicom

        output = tmp_path / "existing_output"
        output.mkdir()
        (output / "marker.txt").write_text("exists")

        convert_nifti_to_dicom(
            ref_dir=sample_dicom_dir,
            nifti_path=sample_nifti_3d,
            output_dir=output,
            force_overwrite=False,
        )

        # Should not have written DICOM files since it skipped
        assert (output / "marker.txt").exists()

    def test_force_overwrite(self, sample_dicom_dir: Path, sample_nifti_3d: Path, tmp_path: Path) -> None:
        """With force=True, should overwrite existing output."""
        from nifti2dicom.convert_image import convert_nifti_to_dicom

        output = tmp_path / "overwrite_test"
        output.mkdir()
        (output / "old_file.txt").write_text("old")

        convert_nifti_to_dicom(
            ref_dir=sample_dicom_dir,
            nifti_path=sample_nifti_3d,
            output_dir=output,
            force_overwrite=True,
            series_description="overwritten",
        )

        assert not (output / "old_file.txt").exists()
        assert len(list(output.glob("*"))) == 3
