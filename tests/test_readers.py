"""Readers must retain voxel locations and reject incoherent references."""

from __future__ import annotations

import itertools

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    PositronEmissionTomographyImageStorage,
    generate_uid,
)

from nifti2dicom.errors import (
    AmbiguousReferenceError,
    GeometryError,
    InputError,
    ReferenceError,
    UnsupportedInputError,
)


def save_nifti(tmp_path, data, affine=None, *, spatial="mm", temporal="unknown", dt=1):
    image = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units(spatial, temporal)
    if data.ndim >= 4:
        image.header["pixdim"][4] = dt
    path = tmp_path / "input.nii"
    nib.save(image, path)
    return path


def save_reference(tmp_path, positions=(0, 2, 4), *, times=(None,), series=None, **tags):
    directory = tmp_path / f"ref-{len(list(tmp_path.iterdir()))}"
    directory.mkdir()
    study, frame = generate_uid(), generate_uid()
    series = series or generate_uid()
    paths = []
    for t_index, time in enumerate(times):
        for z_index, z in enumerate(positions):
            ds = Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = generate_uid()
            ds.PatientID = "PATIENT"
            ds.StudyInstanceUID = study
            ds.SeriesInstanceUID = series
            ds.FrameOfReferenceUID = frame
            ds.Modality = "CT"
            ds.Rows = 3
            ds.Columns = 4
            ds.PixelSpacing = [2, 1]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [10, 20, z]
            ds.SliceThickness = 2
            ds.InstanceNumber = 100 - z_index
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = np.full((3, 4), z_index, dtype=np.uint16).tobytes()
            if time is not None:
                ds.TemporalPositionIdentifier = time
                ds.FrameReferenceTime = t_index * 1250
            for key, value in tags.items():
                setattr(ds, key, value)
            path = directory / f"{t_index}-{z_index}.dcm"
            ds.save_as(path, enforce_file_format=True)
            paths.append(path)
    return directory, paths


def rewrite(path, **tags):
    ds = pydicom.dcmread(path)
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)


def test_scalar_2d_preserves_the_source_plane(tmp_path):
    from nifti2dicom.readers import read_nifti

    data = np.arange(12, dtype=np.int16).reshape(3, 4)
    # A sagittal source plane: canonical axis reordering must not turn it into slices.
    affine = np.array([[0, 0, 5, 11], [2, 0, 0, 13], [0, 3, 0, 17], [0, 0, 0, 1]])
    image = read_nifti(save_nifti(tmp_path, data, affine))
    assert image.data.shape == (1, 1, 4, 3)
    np.testing.assert_array_equal(image.data[0, 0], data.T)
    np.testing.assert_allclose(image.geometry.affine @ [2, 3, 0, 1], [-11, -17, 26, 1])


def test_in_memory_nifti_conversion_preserves_2d_plane_and_converts_units():
    from nifti2dicom.readers.nifti import nifti_to_volume

    data = np.arange(12, dtype=np.int16).reshape(3, 4)
    affine = np.array(
        [[0, 0, 0.005, 0.011], [0.002, 0, 0, 0.013], [0, 0.003, 0, 0.017], [0, 0, 0, 1]]
    )
    loaded = nib.Nifti1Image(data, affine)
    loaded.header.set_xyzt_units("meter")
    image = nifti_to_volume(loaded)
    assert image.source is None
    assert image.data.shape == (1, 1, 4, 3)
    np.testing.assert_array_equal(image.data[0, 0], data.T)
    np.testing.assert_allclose(image.geometry.spacing, [2, 3, 5])
    np.testing.assert_allclose(image.geometry.affine @ [2, 3, 0, 1], [-11, -17, 26, 1])
    np.testing.assert_array_equal(loaded.affine, affine)


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("signs", list(itertools.product((-1, 1), repeat=3)))
def test_signed_spatial_permutations_preserve_physical_landmarks(tmp_path, permutation, signs):
    from nifti2dicom.readers import read_nifti

    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    angle = np.deg2rad(17)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation[:, permutation] * np.array(signs) * [2, 3, 5]
    affine[:3, 3] = [11, -7, 13]
    image = read_nifti(save_nifti(tmp_path, data, affine))
    for xyz in ((0, 0, 0), (1, 2, 3), (1, 0, 2)):
        z, y, x = np.argwhere(image.data[0] == data[xyz])[0]
        expected = affine @ [*xyz, 1]
        expected[:2] *= -1
        np.testing.assert_allclose(image.geometry.affine @ [x, y, z, 1], expected, atol=1e-5)


@pytest.mark.parametrize(("unit", "scale"), [("meter", 1000), ("micron", 0.001)])
def test_spatial_units_scale_positions_and_spacing(tmp_path, unit, scale):
    from nifti2dicom.readers import read_nifti

    affine = np.diag([2.0, 3.0, 5.0, 1.0])
    affine[:3, 3] = [7, 11, 13]
    image = read_nifti(save_nifti(tmp_path, np.zeros((2, 3, 4)), affine, spatial=unit))
    np.testing.assert_allclose(image.geometry.spacing, np.array([2, 3, 5]) * scale)
    np.testing.assert_allclose(image.geometry.position(0), np.array([-7, -11, 13]) * scale)


def test_4d_keeps_temporal_axis_and_converts_milliseconds(tmp_path):
    from nifti2dicom.readers import read_nifti

    data = np.arange(48, dtype=np.int16).reshape(2, 3, 4, 2)
    image = read_nifti(save_nifti(tmp_path, data, temporal="msec", dt=1250))
    assert image.data.shape == (2, 4, 3, 2)
    assert image.time_spacing == 1.25
    np.testing.assert_array_equal(image.data[1], data[..., 1].transpose(2, 1, 0))


def test_three_timepoints_are_not_automatically_rgb(tmp_path):
    from nifti2dicom.readers import read_nifti

    path = save_nifti(tmp_path, np.zeros((2, 4, 5, 3), dtype=np.uint8))
    image = read_nifti(path)
    assert image.kind == "image"
    assert image.data.shape == (3, 5, 4, 2)
    assert image.time_spacing is None


@pytest.mark.parametrize("structured", [False, True])
def test_explicit_rgb_preserves_channel_values(tmp_path, structured):
    from nifti2dicom.readers import read_nifti

    data = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    data[1, 2, 3] = [13, 71, 229]
    encoded = data
    if structured:
        encoded = np.zeros((2, 4, 5), dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
        for i, name in enumerate(encoded.dtype.names):
            encoded[name] = data[..., i]
    image = read_nifti(save_nifti(tmp_path, encoded), kind="rgb")
    assert image.data.shape == (1, 5, 4, 2, 3)
    np.testing.assert_array_equal(image.data[0, 3, 2, 1], [13, 71, 229])


def test_seg_rejects_4d_input_even_with_one_timepoint(tmp_path):
    from nifti2dicom.readers import read_nifti

    with pytest.raises(UnsupportedInputError, match="4D"):
        read_nifti(save_nifti(tmp_path, np.zeros((2, 3, 4, 1))), kind="seg")


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4, 2, 2)])
def test_unsupported_scalar_dimensions_have_an_actionable_error(tmp_path, shape):
    from nifti2dicom.readers import read_nifti

    with pytest.raises(UnsupportedInputError):
        read_nifti(save_nifti(tmp_path, np.zeros(shape)))


def test_nonfinite_data_and_sheared_affine_are_rejected(tmp_path):
    from nifti2dicom.readers import read_nifti

    data = np.zeros((2, 3, 4))
    data[0, 0, 0] = np.nan
    with pytest.raises(InputError, match="finite|NaN"):
        read_nifti(save_nifti(tmp_path, data))
    affine = np.eye(4)
    affine[0, 1] = 0.1
    with pytest.raises(GeometryError, match="shear|orthogonal"):
        read_nifti(save_nifti(tmp_path, np.zeros((2, 3, 4)), affine))


def test_unknown_spatial_units_are_recorded_without_inventing_timing(tmp_path):
    from nifti2dicom.readers import read_nifti

    image = read_nifti(save_nifti(tmp_path, np.zeros((2, 3, 4, 2)), spatial="unknown"))
    assert any("millimeter" in warning.lower() for warning in image.warnings)
    assert image.time_spacing is None


@pytest.mark.parametrize("intent", ["vector", "displacement vector", "symmetric matrix"])
def test_vector_and_tensor_intents_cannot_be_misread_as_scalar_timepoints(tmp_path, intent):
    from nifti2dicom.readers import read_nifti

    data = np.zeros((2, 3, 4, 3), dtype=np.float32)
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_intent(intent)
    path = tmp_path / "vector.nii"
    nib.save(image, path)
    with pytest.raises(UnsupportedInputError, match="vector|tensor|intent"):
        read_nifti(path)


@pytest.mark.parametrize("corruption", ["singular", "nonfinite"])
def test_invalid_affine_in_nifti_header_is_rejected(tmp_path, corruption):
    from nifti2dicom.readers import read_nifti

    image = nib.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    affine = np.eye(4)
    affine[0, 0] = 0 if corruption == "singular" else np.nan
    image.header.set_sform(affine, code=1)
    path = tmp_path / "invalid.nii"
    # Use no image affine so NiBabel does not overwrite the malformed test header.
    nib.save(nib.Nifti1Image(image.dataobj, None, image.header), path)
    with pytest.raises(GeometryError):
        read_nifti(path)


def test_reference_recurses_and_sorts_by_geometry_without_reading_pixels(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, _ = save_reference(tmp_path, positions=(4, 0, 2))
    image = read_reference(tmp_path)
    assert image.geometry.size == (4, 3, 3)
    np.testing.assert_allclose(image.geometry.spacing, [1, 2, 2])
    assert [float(ds.ImagePositionPatient[2]) for ds in image.slices] == [0, 2, 4]
    assert all("PixelData" not in ds for ds in image.slices)
    assert all(path.parent == directory for path in image.paths)


@pytest.mark.parametrize("implicit", [False, True])
def test_no_preamble_reference_keeps_physical_order_and_ignores_unrelated_files(tmp_path, implicit):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path, positions=(4, 0, 2))
    for path in paths:
        ds = pydicom.dcmread(path)
        ds.preamble = None
        ds.file_meta.TransferSyntaxUID = (
            ImplicitVRLittleEndian if implicit else ExplicitVRLittleEndian
        )
        pydicom.dcmwrite(path, ds, enforce_file_format=False)
    (directory / "notes.txt").write_text("These are notes, not medical image data.")
    (directory / "random.bin").write_bytes(bytes(range(255)))
    (directory / "empty.dcm").write_bytes(b"")
    image = read_reference(directory)
    assert image.geometry.size == (4, 3, 3)
    np.testing.assert_allclose(image.geometry.spacing, [1, 2, 2])
    assert [float(ds.ImagePositionPatient[2]) for ds in image.slices] == [0, 2, 4]
    assert all("PixelData" not in ds for ds in image.slices)


@pytest.mark.parametrize(
    ("changes", "error"),
    [({"PatientID": "OTHER"}, ReferenceError), ({"ImagePositionPatient": [10, 20]}, GeometryError)],
)
def test_no_preamble_reference_still_validates_patient_and_geometry(tmp_path, changes, error):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path)
    ds = pydicom.dcmread(paths[1])
    for key, value in changes.items():
        setattr(ds, key, value)
    ds.preamble = None
    pydicom.dcmwrite(paths[1], ds, enforce_file_format=False)
    with pytest.raises(error):
        read_reference(directory)


def test_non_dicom_files_alone_never_form_a_reference(tmp_path):
    from nifti2dicom.readers import read_reference

    (tmp_path / "notes.dcm").write_text("A renamed text file is not a DICOM image.")
    (tmp_path / "random.dcm").write_bytes(bytes(range(255)))
    with pytest.raises(ReferenceError, match="No DICOM"):
        read_reference(tmp_path)


def test_multiple_series_require_an_explicit_selection(tmp_path):
    from nifti2dicom.readers import read_reference

    series = generate_uid()
    save_reference(tmp_path, series=series)
    save_reference(tmp_path)
    with pytest.raises(AmbiguousReferenceError):
        read_reference(tmp_path)
    selected = read_reference(tmp_path, series_uid=series)
    assert str(selected.first.SeriesInstanceUID) == series
    with pytest.raises(ReferenceError, match="found|match"):
        read_reference(tmp_path, series_uid=generate_uid())


@pytest.mark.parametrize("field", ["PatientID", "StudyInstanceUID", "FrameOfReferenceUID"])
def test_mixed_reference_identity_is_rejected(tmp_path, field):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path)
    rewrite(paths[1], **{field: generate_uid() if field.endswith("UID") else "OTHER"})
    with pytest.raises(ReferenceError, match="patient|Patient|study|Study|frame|Frame"):
        read_reference(directory)


def test_duplicate_sop_instance_is_rejected(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path)
    rewrite(paths[1], SOPInstanceUID=pydicom.dcmread(paths[0]).SOPInstanceUID)
    with pytest.raises(ReferenceError, match="duplicate|Duplicate"):
        read_reference(directory)


@pytest.mark.parametrize(
    "tags",
    [
        {"NumberOfFrames": 2},
        {"SOPClassUID": "1.2.840.10008.5.1.4.1.1.2.1"},
        {"SOPClassUID": "1.2.840.10008.5.1.4.1.1.7.2", "NumberOfFrames": 1},
        {"SOPClassUID": "1.2.840.10008.5.1.4.1.1.6.2", "NumberOfFrames": 1},
    ],
)
def test_multiframe_or_enhanced_reference_is_rejected(tmp_path, tags):
    from nifti2dicom.readers import read_reference

    directory, _ = save_reference(tmp_path, **tags)
    with pytest.raises(UnsupportedInputError, match="multiframe|[Ee]nhanced"):
        read_reference(directory)


@pytest.mark.parametrize("positions", [(0, 2, 5), (0, 0, 2)])
def test_irregular_or_duplicate_planes_are_rejected(tmp_path, positions):
    from nifti2dicom.readers import read_reference

    directory, _ = save_reference(tmp_path, positions=positions)
    with pytest.raises(GeometryError):
        read_reference(directory)


def test_scanner_rounded_positions_form_a_grid_without_accumulated_drift(tmp_path):
    from nifti2dicom.readers import read_reference

    # Six significant digits serialize this regular scanner grid to 0.01 mm
    # below -1000 mm and 0.001 mm above it, producing unequal adjacent gaps.
    positions = tuple(f"{-1253.96 + index * 2.027:.6g}" for index in range(419))
    directory, _ = save_reference(tmp_path, positions=positions, SliceThickness="2.027")
    reference = read_reference(directory)
    assert reference.geometry.spacing[2] == pytest.approx(2.027, abs=1e-12)
    np.testing.assert_allclose(reference.geometry.position(418), [10, 20, -406.674], atol=1e-10)
    assert [str(ds.ImagePositionPatient[2]) for ds in reference.slices] == list(positions)
    assert any("rounded" in message and "0.005" in message for message in reference.warnings)
    residuals = [
        float(ds.ImagePositionPatient[2]) - reference.geometry.position(i)[2]
        for i, ds in enumerate(reference.slices)
    ]
    assert max(abs(value) for value in residuals) == pytest.approx(0.005, abs=1e-10)


def test_small_gap_variations_cannot_hide_accumulated_plane_drift(tmp_path):
    from nifti2dicom.readers import read_reference

    positions = tuple(f"{2 * index + 0.000004 * index**2:.6f}" for index in range(101))
    directory, _ = save_reference(tmp_path, positions=positions)
    with pytest.raises(GeometryError, match="nonuniform"):
        read_reference(directory)


@pytest.mark.parametrize(
    "positions",
    [
        ("0.00", "2.05", "4.10", "6.10"),
        ("0", "2", "5"),
        ("0.000000", "2.023000", "4.054000"),
    ],
)
def test_coordinate_precision_does_not_excuse_real_spacing_changes(tmp_path, positions):
    from nifti2dicom.readers import read_reference

    directory, _ = save_reference(tmp_path, positions=positions)
    with pytest.raises(GeometryError, match="nonuniform"):
        read_reference(directory)


def test_in_plane_shift_between_reference_slices_is_rejected(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path)
    rewrite(paths[1], ImagePositionPatient=[10.5, 20, 2])
    with pytest.raises(GeometryError, match="align|shear"):
        read_reference(directory)


def test_temporal_reference_is_sorted_with_metadata_retained(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, _ = save_reference(tmp_path, positions=(4, 0, 2), times=(2, 1))
    image = read_reference(directory)
    assert image.timepoints == 2
    assert image.geometry.size == (4, 3, 3)
    assert [int(ds.TemporalPositionIdentifier) for ds in image.slices] == [1, 1, 1, 2, 2, 2]
    assert [float(ds.ImagePositionPatient[2]) for ds in image.slices] == [0, 2, 4, 0, 2, 4]
    assert float(image.first.FrameReferenceTime) == 1250


def test_per_slice_reference_time_does_not_split_one_spatial_volume(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path)
    for path, frame_time in zip(paths, (100, 200, 300), strict=True):
        rewrite(path, FrameReferenceTime=frame_time)
    image = read_reference(directory)
    assert image.timepoints == 1
    assert image.geometry.size == (4, 3, 3)
    assert [float(ds.FrameReferenceTime) for ds in image.slices] == [100, 200, 300]


def test_frame_reference_time_groups_repeated_spatial_volumes(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path, positions=(4, 0, 2), times=(1, 2))
    for path in paths:
        ds = pydicom.dcmread(path)
        del ds.TemporalPositionIdentifier
        ds.save_as(path, enforce_file_format=True)
    image = read_reference(directory)
    assert image.timepoints == 2
    assert image.geometry.size == (4, 3, 3)
    assert [float(ds.FrameReferenceTime) for ds in image.slices] == [0, 0, 0, 1250, 1250, 1250]


def save_native_pet_reference(tmp_path):
    directory, paths = save_reference(
        tmp_path,
        positions=(2, 0),
        times=(2, 1),
        Modality="PT",
        SOPClassUID=PositronEmissionTomographyImageStorage,
    )
    for path, image_index, frame_time in zip(
        paths, (4, 3, 2, 1), (2700, 2600, 200, 100), strict=True
    ):
        ds = pydicom.dcmread(path)
        del ds.TemporalPositionIdentifier
        ds.SeriesType = ["DYNAMIC", "IMAGE"]
        ds.NumberOfSlices = 2
        ds.NumberOfTimeSlices = 2
        ds.ImageIndex = image_index
        ds.FrameReferenceTime = frame_time
        ds.save_as(path, enforce_file_format=True)
    return directory, paths


def test_native_pet_indices_group_slice_specific_reference_times(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, _ = save_native_pet_reference(tmp_path)
    image = read_reference(directory)
    assert image.timepoints == 2
    assert image.geometry.size == (4, 3, 2)
    assert [int(ds.ImageIndex) for ds in image.slices] == [1, 2, 3, 4]
    assert [float(ds.FrameReferenceTime) for ds in image.slices] == [100, 200, 2600, 2700]
    assert [float(ds.ImagePositionPatient[2]) for ds in image.slices] == [0, 2, 0, 2]


@pytest.mark.parametrize(
    "changes",
    [
        {"NumberOfSlices": 3},
        {"NumberOfTimeSlices": 3},
        {"ImageIndex": 1},
        {"SeriesType": ["STATIC", "IMAGE"]},
    ],
)
def test_incoherent_native_pet_temporal_indices_are_rejected(tmp_path, changes):
    from nifti2dicom.readers import read_reference

    directory, paths = save_native_pet_reference(tmp_path)
    rewrite(paths[0], **changes)
    with pytest.raises(ReferenceError, match="PET"):
        read_reference(directory)


def test_native_pet_image_index_must_agree_with_physical_slice_order(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_native_pet_reference(tmp_path)
    rewrite(paths[0], ImageIndex=3)
    rewrite(paths[1], ImageIndex=4)
    with pytest.raises(GeometryError, match="ImageIndex"):
        read_reference(directory)


@pytest.mark.parametrize("reverse_instance_numbers", [False, True])
def test_consistently_descending_native_pet_indices_recover_with_warning(
    tmp_path, reverse_instance_numbers
):
    from nifti2dicom.readers import read_reference

    directory, paths = save_native_pet_reference(tmp_path)
    original = {}
    for path, index in zip(paths, (3, 4, 1, 2), strict=True):
        ds = pydicom.dcmread(path)
        ds.ImageIndex = index
        ds.InstanceNumber = 5 - index if reverse_instance_numbers else index
        ds.save_as(path, enforce_file_format=True)
        original[str(ds.SOPInstanceUID)] = (ds.ImagePositionPatient, ds.pixel_array.copy())
    reference = read_reference(directory)
    assert [int(ds.ImageIndex) for ds in reference.slices] == [2, 1, 4, 3]
    assert [float(ds.ImagePositionPatient[2]) for ds in reference.slices] == [0, 2, 0, 2]
    assert any(
        "descending" in warning and "ImageIndex" in warning for warning in reference.warnings
    )
    for ds, path in zip(reference.slices, reference.paths, strict=True):
        position, pixels = original[str(ds.SOPInstanceUID)]
        assert ds.ImagePositionPatient == position
        np.testing.assert_array_equal(pydicom.dcmread(path).pixel_array, pixels)


def test_native_pet_arbitrary_slice_permutation_is_rejected(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(
        tmp_path,
        positions=(0, 2, 4),
        times=(1, 2),
        Modality="PT",
        SOPClassUID=PositronEmissionTomographyImageStorage,
    )
    for path, index in zip(paths, (1, 3, 2, 4, 6, 5), strict=True):
        ds = pydicom.dcmread(path)
        del ds.TemporalPositionIdentifier
        ds.SeriesType = ["DYNAMIC", "IMAGE"]
        ds.NumberOfSlices, ds.NumberOfTimeSlices, ds.ImageIndex = 3, 2, index
        ds.save_as(path, enforce_file_format=True)
    with pytest.raises(GeometryError, match="ImageIndex"):
        read_reference(directory)


def test_native_pet_missing_index_is_rejected(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_native_pet_reference(tmp_path)
    ds = pydicom.dcmread(paths[0])
    del ds.ImageIndex
    ds.save_as(paths[0], enforce_file_format=True)
    with pytest.raises(ReferenceError, match="ImageIndex"):
        read_reference(directory)


def test_descending_native_pet_indices_cannot_reverse_time(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_native_pet_reference(tmp_path)
    for path, index in zip(paths, (1, 2, 3, 4), strict=True):
        rewrite(path, ImageIndex=index)
    with pytest.raises(ReferenceError, match="FrameReferenceTime"):
        read_reference(directory)


def test_descending_native_pet_recovery_requires_complete_frame_times(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_native_pet_reference(tmp_path)
    for path, index in zip(paths, (3, 4, 1, 2), strict=True):
        rewrite(path, ImageIndex=index)
    ds = pydicom.dcmread(paths[0])
    del ds.FrameReferenceTime
    ds.save_as(paths[0], enforce_file_format=True)
    with pytest.raises(ReferenceError, match="FrameReferenceTime"):
        read_reference(directory)


def test_native_pet_indices_require_complete_explicit_dimensions(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_native_pet_reference(tmp_path)
    for path in paths:
        ds = pydicom.dcmread(path)
        del ds.NumberOfTimeSlices
        ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ReferenceError, match="NumberOfTimeSlices"):
        read_reference(directory)


def test_native_pet_temporal_index_cannot_reverse_known_frame_times(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_native_pet_reference(tmp_path)
    rewrite(paths[-1], FrameReferenceTime=3000)
    with pytest.raises(ReferenceError, match="FrameReferenceTime"):
        read_reference(directory)


def test_single_slice_reference_uses_declared_spacing(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, _ = save_reference(tmp_path, positions=(7,))
    image = read_reference(directory)
    assert image.geometry.size == (4, 3, 1)
    np.testing.assert_allclose(image.geometry.spacing, [1, 2, 2])
    np.testing.assert_allclose(image.geometry.position(0), [10, 20, 7])


def test_missing_frame_of_reference_produces_warning_for_ordinary_reference(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path)
    for path in paths:
        ds = pydicom.dcmread(path)
        del ds.FrameOfReferenceUID
        ds.save_as(path, enforce_file_format=True)
    assert any("FrameOfReferenceUID" in warning for warning in read_reference(directory).warnings)


def test_zero_declared_single_slice_spacing_is_rejected(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, _ = save_reference(tmp_path, positions=(7,), SpacingBetweenSlices=0)
    with pytest.raises(GeometryError):
        read_reference(directory)


def test_missing_single_slice_spacing_records_assumption(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path, positions=(7,))
    ds = pydicom.dcmread(paths[0])
    del ds.SliceThickness
    ds.save_as(paths[0], enforce_file_format=True)
    result = read_reference(directory)
    assert result.geometry.spacing[2] == 1
    assert any("1 mm" in warning for warning in result.warnings)


def test_same_patient_id_with_different_patient_names_is_rejected(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path, PatientName="First^Patient")
    rewrite(paths[1], PatientName="Other^Patient")
    with pytest.raises(ReferenceError, match="Patient"):
        read_reference(directory)


def test_reference_oblique_positions_use_slice_normal(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path, positions=(4, 0, 2))
    a = float(1 / np.sqrt(2))
    for path, distance in zip(paths, (4, 0, 2), strict=True):
        rewrite(
            path,
            ImageOrientationPatient=[a, 0, a, 0, 1, 0],
            ImagePositionPatient=[10 - a * distance, 20, 30 + a * distance],
        )
    image = read_reference(directory)
    np.testing.assert_allclose(image.geometry.position(2), [10 - a * 4, 20, 30 + a * 4])
    np.testing.assert_allclose(image.geometry.iop, [a, 0, a, 0, 1, 0])
    np.testing.assert_allclose(image.geometry.spacing, [1, 2, 2])


def test_temporal_groups_require_identical_complete_spatial_grids(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path, times=(1, 2))
    rewrite(paths[-1], ImagePositionPatient=[10, 20, 6])
    with pytest.raises(GeometryError):
        read_reference(directory)


def test_reference_without_temporal_tag_does_not_infer_time_from_instance_number(tmp_path):
    from nifti2dicom.readers import read_reference

    directory, paths = save_reference(tmp_path, times=(1, 2))
    for path in paths:
        ds = pydicom.dcmread(path)
        del ds.TemporalPositionIdentifier
        del ds.FrameReferenceTime
        ds.save_as(path, enforce_file_format=True)
    with pytest.raises(GeometryError, match="duplicate"):
        read_reference(directory)
