"""Public conversion behavior, with real files and independent read-back."""

import nibabel as nib
import numpy as np
import pydicom
import pytest

from nifti2dicom.errors import ConversionError, OutputError


@pytest.mark.parametrize("shape,count", [((4, 4), 1), ((4, 4, 3), 3), ((4, 4, 3, 2), 6)])
def test_public_api_preserves_dimensions_and_values(tmp_path, sample_dicom_dir, shape, count):
    import nifti2dicom

    assert hasattr(nifti2dicom, "convert"), "A single public conversion API is required"
    path = tmp_path / "input.nii.gz"
    nib.save(nib.Nifti1Image(np.full(shape, 123, dtype=np.int16), np.eye(4)), path)
    result = nifti2dicom.convert(path, sample_dicom_dir, tmp_path / "result")
    assert len(result.files) == count
    for file in result.files:
        ds = pydicom.dcmread(file)
        np.testing.assert_allclose(
            ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept), 123
        )


def test_public_api_is_quiet(tmp_path, sample_dicom_dir, sample_nifti_3d, capsys):
    import nifti2dicom

    assert hasattr(nifti2dicom, "convert")
    nifti2dicom.convert(sample_nifti_3d, sample_dicom_dir, tmp_path / "result")
    assert capsys.readouterr().out == ""


def test_existing_output_is_preserved(tmp_path, sample_dicom_dir, sample_nifti_3d):
    import nifti2dicom

    assert hasattr(nifti2dicom, "convert")
    output = tmp_path / "result"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(OutputError, match="already exists"):
        nifti2dicom.convert(sample_nifti_3d, sample_dicom_dir, output)
    assert marker.read_text() == "keep"
    assert list(output.iterdir()) == [marker]


def test_inspect_does_not_write(tmp_path, sample_dicom_dir, sample_nifti_3d):
    import nifti2dicom

    assert hasattr(nifti2dicom, "inspect")
    before = set(tmp_path.rglob("*"))
    result = nifti2dicom.inspect(sample_nifti_3d, sample_dicom_dir)
    assert result.kind == "image"
    assert set(tmp_path.rglob("*")) == before


def test_failed_overwrite_restores_original_and_cleans_staging(
    tmp_path, sample_dicom_dir, sample_nifti_3d, monkeypatch
):
    import nifti2dicom
    from nifti2dicom.writers import image

    output = tmp_path / "result"
    output.mkdir()
    (output / "keep.txt").write_text("original")
    original_writer = image.write_images

    def fail_after_writing(*args, **kwargs):
        original_writer(*args, **kwargs)
        raise OSError("simulated full disk")

    monkeypatch.setattr(image, "write_images", fail_after_writing)
    with pytest.raises(OutputError, match="write"):
        nifti2dicom.convert(sample_nifti_3d, sample_dicom_dir, output, overwrite=True)
    assert (output / "keep.txt").read_text() == "original"
    assert not list(tmp_path.glob(".result.*"))


def test_input_output_overlap_rejected_even_with_overwrite(sample_dicom_dir, sample_nifti_3d):
    import nifti2dicom

    with pytest.raises(OutputError, match="overlaps"):
        nifti2dicom.convert(sample_nifti_3d, sample_dicom_dir, sample_dicom_dir, overwrite=True)
    assert len(list(sample_dicom_dir.glob("*.dcm"))) == 3


def test_label_intent_infers_segmentation(tmp_path, sample_dicom_dir):
    import nifti2dicom

    img = nib.Nifti1Image(np.ones((4, 4, 3), dtype=np.uint8), np.eye(4))
    img.header.set_intent("label")
    path = tmp_path / "label.nii.gz"
    nib.save(img, path)
    result = nifti2dicom.inspect(path, sample_dicom_dir)
    assert result.kind == "seg"


def test_integer_pixels_do_not_infer_segmentation(sample_nifti_3d, sample_dicom_dir):
    import nifti2dicom

    assert nifti2dicom.inspect(sample_nifti_3d, sample_dicom_dir).kind == "image"


def test_invalid_labels_json_has_actionable_error(tmp_path, sample_nifti_3d, sample_dicom_dir):
    import nifti2dicom

    path = tmp_path / "labels.json"
    path.write_text("{broken")
    with pytest.raises(ConversionError) as caught:
        nifti2dicom.convert(sample_nifti_3d, sample_dicom_dir, labels=path)
    assert caught.value.code == "invalid_labels"
    assert caught.value.hint


def test_rgb_resampling_quantizes_interpolated_channels(tmp_path, sample_dicom_dir):
    import nifti2dicom

    colors = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    colors[0] = [0, 10, 100]
    colors[1] = [11, 21, 201]
    affine = np.diag([-2.0, -3.0, 2.0, 1.0])
    affine[2, 3] = 1.0
    path = tmp_path / "rgb.nii.gz"
    nib.save(nib.Nifti1Image(colors, affine), path)
    result = nifti2dicom.convert(
        path, sample_dicom_dir, tmp_path / "out", kind="rgb", geometry="reference"
    )
    ds = pydicom.dcmread(result.files[0])
    np.testing.assert_array_equal(ds.pixel_array[0, 1], [6, 16, 150])


@pytest.mark.parametrize("separate_header", [False, True])
def test_overwrite_protects_symlinked_reference_files(
    tmp_path, sample_dicom_dir, sample_nifti_3d, separate_header
):
    from nifti2dicom import convert

    output = tmp_path / "out"
    output.mkdir()
    source = next(sample_dicom_dir.glob("*.dcm"))
    protected = output / "original.dcm"
    protected.write_bytes(source.read_bytes())
    links = tmp_path / "links"
    links.mkdir()
    link = links / "reference.dcm"
    link.symlink_to(protected)
    reference = sample_dicom_dir if separate_header else links
    kwargs = {"header_source": links} if separate_header else {}

    with pytest.raises(OutputError, match="overlaps"):
        convert(sample_nifti_3d, reference, output, overwrite=True, **kwargs)

    assert protected.read_bytes() == source.read_bytes()
    assert link.is_file()


@pytest.mark.parametrize("modality", ["MR", "CT"])
def test_inspection_explains_standard_extended_profiles(
    sample_dicom_dir, sample_nifti_3d, sample_nifti_4d, modality
):
    from pydicom.uid import MRImageStorage

    from nifti2dicom import inspect

    if modality == "MR":
        for file in sample_dicom_dir.glob("*.dcm"):
            ds = pydicom.dcmread(file)
            ds.Modality = "MR"
            ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
            pydicom.dcmwrite(file, ds, enforce_file_format=True)
    path = sample_nifti_3d if modality == "MR" else sample_nifti_4d
    result = inspect(path, sample_dicom_dir)
    assert any("standard-extended" in warning.lower() for warning in result.warnings)
