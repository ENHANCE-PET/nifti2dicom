"""Compatibility writers must never follow output symlinks or replace files."""

import numpy as np
import pydicom
import pytest

from nifti2dicom.errors import OutputError
from nifti2dicom.writer import save_slice, write_slices_parallel
from tests.test_writers import make_reference


@pytest.mark.parametrize("batch", [False, True])
def test_legacy_writer_rejects_dangling_output_symlink(tmp_path, batch):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside.dcm"
    destination = output / "image.dcm"
    destination.symlink_to(outside)
    reference = make_reference(slices=1)
    pixels = np.arange(6).reshape(2, 3)
    with pytest.raises(OutputError):
        if batch:
            write_slices_parallel(
                list(reference.slices), ["image.dcm"], pixels[None], "Derived", output, "CT"
            )
        else:
            save_slice(reference.first, pixels, "Derived", "image.dcm", output, "CT")
    assert destination.is_symlink()
    assert not outside.exists()


@pytest.mark.parametrize("batch", [False, True])
def test_legacy_writer_exclusively_creates_output_after_validation(tmp_path, monkeypatch, batch):
    reference = make_reference(slices=1)
    pixels = np.arange(6).reshape(2, 3)
    destination = tmp_path / "image.dcm"
    from nifti2dicom import writer

    validate_destination = writer._destination

    def replace_after_validation(output, filename):
        path = validate_destination(output, filename)
        path.write_bytes(b"existing user file")
        return path

    monkeypatch.setattr(writer, "_destination", replace_after_validation)
    with pytest.raises(OutputError):
        if batch:
            write_slices_parallel(
                list(reference.slices), ["image.dcm"], pixels[None], "Derived", tmp_path, "CT"
            )
        else:
            save_slice(reference.first, pixels, "Derived", "image.dcm", tmp_path, "CT")
    assert destination.read_bytes() == b"existing user file"


def test_legacy_writer_new_output_has_decodable_pixels(tmp_path):
    reference = make_reference(slices=1)
    pixels = np.arange(6).reshape(2, 3)
    save_slice(reference.first, pixels, "Derived", "image.dcm", tmp_path, "CT")
    restored = pydicom.dcmread(tmp_path / "image.dcm")
    np.testing.assert_array_equal(restored.pixel_array, pixels)
