"""Existing entry points must reach the same corrected conversion engine."""

import nibabel as nib
import numpy as np
import pydicom
import pytest


def test_legacy_resampling_does_not_flip_aligned_pixels(tmp_path, sample_dicom_dir):
    from nifti2dicom.converter import nifti_to_dicom_with_resampling

    values = np.broadcast_to(np.arange(4, dtype=np.int16)[None, :, None], (4, 4, 3)).copy()
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[2, 3] = 1.0
    nifti = tmp_path / "aligned.nii.gz"
    nib.save(nib.Nifti1Image(values, affine), nifti)
    output = tmp_path / "result"
    nifti_to_dicom_with_resampling(
        str(nifti), str(sample_dicom_dir), str(output), str(sample_dicom_dir)
    )
    ds = pydicom.dcmread(sorted(output.glob("*.dcm"))[0])
    np.testing.assert_array_equal(
        ds.pixel_array[:, 0] * float(ds.RescaleSlope) + float(ds.RescaleIntercept), [0, 1, 2, 3]
    )
    source = pydicom.dcmread(next(sample_dicom_dir.glob("*.dcm")))
    assert ds.SeriesInstanceUID != source.SeriesInstanceUID


def test_old_positional_image_alias_accepts_vendor_and_verbose(
    tmp_path, sample_nifti_3d, sample_dicom_dir
):
    from nifti2dicom.converter import save_dicom_from_nifti_image

    with pytest.warns(DeprecationWarning):
        save_dicom_from_nifti_image(
            str(sample_dicom_dir),
            str(sample_nifti_3d),
            str(tmp_path / "out"),
            "ux",
            "converted",
            None,
            False,
            False,
        )
    assert len(list((tmp_path / "out").glob("*.dcm"))) == 3


def test_legacy_save_slice_replaces_inconsistent_pixel_tags(tmp_path):
    from nifti2dicom.writer import save_slice
    from tests.conftest import _make_dicom_slice

    ds = _make_dicom_slice(1)
    ds.BitsStored, ds.HighBit = 12, 11
    save_slice(ds, np.full((4, 4), 3000.0), "test", "slice.dcm", tmp_path, "CT")
    restored = pydicom.dcmread(tmp_path / "slice.dcm")
    assert restored.pixel_array[0, 0] == 3000
    assert restored.BitsStored == 16
    assert ds.BitsStored == 12


def test_header_copy_preserves_pixels_and_encoding():
    from nifti2dicom.tags import copy_tags
    from tests.conftest import _make_dicom_slice

    target, source = _make_dicom_slice(1), _make_dicom_slice(1)
    del source.PixelData
    source.RescaleSlope = 2
    original_bytes = target.PixelData
    original_uid = target.SeriesInstanceUID
    copy_tags(target, source)
    assert target.PixelData == original_bytes
    assert float(target.RescaleSlope) == 1
    assert target.SeriesInstanceUID == original_uid
