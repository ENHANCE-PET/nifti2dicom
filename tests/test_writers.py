"""Read-back regression tests for quantitative and RGB DICOM writing."""

from copy import deepcopy
from dataclasses import replace

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    MRImageStorage,
    PositronEmissionTomographyImageStorage,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from nifti2dicom.errors import PixelEncodingError, ReferenceError, UnsupportedInputError
from nifti2dicom.models import Geometry, ImageVolume, ReferenceSeries


def make_reference(modality="CT", *, timepoints=1, slices=2):
    geometry = Geometry(np.diag([2.0, 3.0, 4.0, 1.0]), (3, 2, slices))
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    datasets = []
    for t in range(timepoints):
        for z in range(slices):
            ds = Dataset()
            ds.PatientName, ds.PatientID = "Example^Patient", "P001"
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
            ds.FrameOfReferenceUID = frame
            ds.Modality = modality
            ds.SOPClassUID = {
                "CT": CTImageStorage,
                "MR": MRImageStorage,
                "PT": PositronEmissionTomographyImageStorage,
            }.get(modality, SecondaryCaptureImageStorage)
            ds.SOPInstanceUID = generate_uid()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRBigEndian
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            ds.Rows, ds.Columns = 2, 3
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
            ds.PixelRepresentation = 0
            ds.RescaleSlope, ds.RescaleIntercept = "2", "-1024"
            ds.PixelSpacing = [3, 2]
            ds.ImagePositionPatient = [0, 0, z * 4]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.TemporalPositionIdentifier = t + 1
            ds.NumberOfTemporalPositions = timepoints
            ds.FrameReferenceTime = t * 2500
            ds.ActualFrameDuration = 2000
            ds.AcquisitionTime = f"1200{t * 3:02d}"
            ds.TriggerTime = 123
            ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
            ds.SeriesDescription = "Reference"
            if modality == "MR":
                ds.ScanningSequence, ds.SequenceVariant = "GR", "SP"
                ds.EchoTime = 3 + t
            if modality == "PT":
                ds.Units, ds.CountsSource = "BQML", "EMISSION"
                ds.SeriesType = ["DYNAMIC" if timepoints > 1 else "STATIC", "IMAGE"]
                ds.SeriesDate, ds.SeriesTime = "20260101", "120000"
                ds.DecayCorrection = "NONE"
            datasets.append(ds)
    return ReferenceSeries(tuple(datasets), (), geometry, timepoints=timepoints)


def image_for(data, reference, *, time_spacing=None, geometry=None, kind="image"):
    return ImageVolume(
        np.asarray(data), geometry or reference.geometry, kind=kind, time_spacing=time_spacing
    )


def read_images(tmp_path, image, reference, **kwargs):
    from nifti2dicom.writers.image import write_images

    return [pydicom.dcmread(p) for p in write_images(image, reference, tmp_path, **kwargs)]


@pytest.mark.parametrize(
    "values",
    [
        [-32768, -100, -1, 0, 1, 32767],
        [0, 100, 3000, 4096, 32768, 65535],
    ],
)
def test_integer_values_survive_unsigned_12bit_big_endian_reference(tmp_path, values):
    reference = make_reference(slices=1)
    data = np.array(values).reshape(1, 1, 2, 3)
    ds = read_images(tmp_path, image_for(data, reference), reference)[0]
    real = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    np.testing.assert_array_equal(real, data[0, 0])
    assert (ds.BitsAllocated, ds.BitsStored, ds.HighBit) == (16, 16, 15)
    assert ds.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert ds.RescaleType


def test_pet_input_is_already_real_and_intercept_is_zero(tmp_path):
    reference = make_reference("PT", slices=1)
    data = np.array([0.0, 100.0, 100000.0, 123.5, 99999.0, 50000.0]).reshape(1, 1, 2, 3)
    ds = read_images(tmp_path, image_for(data, reference), reference)[0]
    assert float(ds.RescaleIntercept) == 0
    np.testing.assert_allclose(ds.pixel_array * float(ds.RescaleSlope), data[0, 0], atol=0.77)
    assert ds.Units == "BQML"
    assert ds.SOPClassUID == PositronEmissionTomographyImageStorage
    assert ds.ImageType[:2] == ["DERIVED", "PRIMARY"]


@pytest.mark.parametrize("timepoints", [1, 3], ids=["3d", "4d"])
def test_pet_precision_is_local_to_each_image_not_bright_neighbors(tmp_path, timepoints):
    reference = make_reference("PT", slices=2, timepoints=timepoints)
    pattern = np.array([0, 0.0123, 0.183, 0.547, 0.992, 1.23456789]).reshape(2, 3)
    data = np.stack([pattern * 10.0**i for i in range(timepoints * 2)])
    data = data.reshape(timepoints, 2, 2, 3)
    before = data.copy()
    output = read_images(tmp_path, image_for(data, reference), reference)
    for ds in output:
        t, z = divmod(int(ds.ImageIndex) - 1, 2)
        expected = data[t, z]
        decoded = ds.pixel_array.astype(float) * float(ds.RescaleSlope)
        error = np.max(np.abs(decoded - expected))
        # Independent bound from this plane's real range, not the writer's scale.
        bound = float(expected.max()) / (2 * 65535)
        assert error <= bound + float(expected.max()) * 1e-12
        assert float(ds.RescaleIntercept) == 0
        assert np.count_nonzero(decoded) == np.count_nonzero(expected)
        reported = float(ds.DerivationDescription.rsplit(": ", 1)[1].rstrip("."))
        assert reported == pytest.approx(error, rel=1e-7, abs=1e-15)
        assert ds.FrameReferenceTime == reference.slices[t * 2 + z].FrameReferenceTime
        assert ds.ImagePositionPatient == [0, 0, z * 4]
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("timepoints", [1, 3], ids=["3d", "4d"])
def test_pet_exact_planes_remain_exact_next_to_quantized_planes(tmp_path, timepoints):
    reference = make_reference("PT", slices=9 // timepoints, timepoints=timepoints)
    data = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 2, 123, 345, 32767],
            [-32768, -2, -1, 0, 1, 32767],
            [0.000125] * 6,
            [-0.000125] * 6,
            [-1.1e8, -2.3, 0, 1.2, 4.5, 2.2e8],
            [1.25e-200] * 6,
            [-1.25e-200] * 6,
            [0, 1, 2, 32768, 65534, 65535],
        ],
        dtype=float,
    )
    # The first timepoint is nonnegative: a later negative plane must still
    # select signed storage for the earlier planes in this same PET series.
    data = data[[0, 1, 8, 3, 6, 7, 2, 4, 5]].reshape(timepoints, 9 // timepoints, 2, 3)
    output = read_images(tmp_path, image_for(data, reference), reference)
    for z, ds in enumerate(output):
        expected = data.reshape(-1, 2, 3)[z]
        decoded = ds.pixel_array.astype(float) * float(ds.RescaleSlope)
        assert float(ds.RescaleIntercept) == 0
        # PET requires one signedness across the series, even for positive planes.
        assert int(ds.PixelRepresentation) == 1
        if z in (0, 1, 6):
            np.testing.assert_array_equal(decoded, expected)
        else:
            low, high = float(expected.min()), float(expected.max())
            bound = max(high / 32767, -low / 32768) / 2
            tolerance = bound + float(np.abs(expected).max()) * 1e-12
            np.testing.assert_allclose(decoded, expected, rtol=0, atol=tolerance)


def test_nonnegative_pet_keeps_unsigned_integer_precision(tmp_path):
    reference = make_reference("PT", slices=2)
    data = np.array([0, 1, 2, 32768, 65534, 65535, 0.1, 0.3, 0.5, 1.1, 2.2, 3.3])
    data = data.reshape(1, 2, 2, 3)
    output = read_images(tmp_path, image_for(data, reference), reference)
    assert {int(ds.PixelRepresentation) for ds in output} == {0}
    decoded = output[0].pixel_array.astype(float) * float(output[0].RescaleSlope)
    np.testing.assert_array_equal(decoded, data[0, 0])


@pytest.mark.parametrize("zero_intercept", [False, True])
def test_forced_signed_encoding_preserves_values_within_the_selected_storage_range(
    zero_intercept,
):
    from nifti2dicom.pixels import encode_pixels

    data = np.array([0, 1, 2, 32768, 65534, 65535])
    encoded = encode_pixels(data, zero_intercept=zero_intercept, force_signed=True)
    decoded = encoded.values.astype(float) * encoded.slope + encoded.intercept
    assert encoded.signed and encoded.values.dtype.kind == "i"
    if zero_intercept:
        assert encoded.intercept == 0
        np.testing.assert_allclose(decoded, data, rtol=0, atol=65535 / (2 * 32767) + 1e-8)
    else:
        np.testing.assert_array_equal(decoded, data)


def test_extreme_forced_signed_offset_fails_with_a_structured_error():
    from nifti2dicom.pixels import encode_pixels

    maximum = np.finfo(np.float64).max
    with pytest.raises(PixelEncodingError) as caught:
        encode_pixels(np.array([-maximum, maximum]), force_signed=True)
    assert caught.value.hint


@pytest.mark.parametrize("bad", ["text", 1j])
def test_pet_unsupported_pixel_type_has_a_clear_error(tmp_path, bad):
    reference = make_reference("PT", slices=1)
    data = np.full((1, 1, 2, 3), bad)
    with pytest.raises(PixelEncodingError):
        read_images(tmp_path, image_for(data, reference), reference)


@pytest.mark.parametrize("sign", [-1, 1])
def test_pet_scale_underflow_fails_instead_of_erasing_nonzero_input(sign):
    from nifti2dicom.pixels import encode_pixels

    data = np.array([0.0, sign * np.nextafter(0.0, 1.0)])
    with pytest.raises(PixelEncodingError) as caught:
        encode_pixels(data, zero_intercept=True)
    assert caught.value.hint


def test_pet_code_cleanup_preserves_in_memory_sources_and_notifies_once(tmp_path):
    reference = make_reference("PT", timepoints=2)
    for ds in reference.slices:
        complete, incomplete = Dataset(), Dataset()
        code = Dataset()
        code.URNCodeValue, code.CodeMeaning = "urn:example:tracer", "Synthetic tracer"
        complete.RadiopharmaceuticalCodeSequence = [code]
        complete.RadionuclideTotalDose = 12345
        incomplete.RadiopharmaceuticalCodeSequence = [Dataset()]
        incomplete.RadionuclideTotalDose = 67890
        ds.RadiopharmaceuticalInformationSequence = [complete, incomplete]
    before = deepcopy(reference.slices)
    messages = []
    output = read_images(
        tmp_path,
        image_for(np.arange(24).reshape(2, 2, 2, 3), reference),
        reference,
        on_warning=messages.append,
    )
    assert len(messages) == 1
    assert "RadiopharmaceuticalCodeSequence" in messages[0]
    assert reference.slices == before
    for ds in output:
        first, second = ds.RadiopharmaceuticalInformationSequence
        assert first == before[0].RadiopharmaceuticalInformationSequence[0]
        assert "RadiopharmaceuticalCodeSequence" not in second
        assert second.RadionuclideTotalDose == 67890


def test_encoding_reports_error_and_rounds_to_nearest():
    from nifti2dicom.pixels import encode_pixels

    data = np.array([-100.125, -0.2, 0.1, 99.4, 100000.75])
    encoded = encode_pixels(data)
    reconstructed = encoded.values.astype(float) * encoded.slope + encoded.intercept
    error = np.abs(reconstructed - data).max()
    assert error <= encoded.max_error + 1e-10
    assert error <= encoded.slope / 2 + 1e-8
    assert encoded.values.dtype.itemsize == 2
    assert encoded.values.dtype.kind == ("i" if encoded.signed else "u")


def test_fixed_scaling_inverts_real_values_with_nearest_rounding():
    from nifti2dicom.pixels import encode_fixed_pixels

    encoded = encode_fixed_pixels(np.array([-10.0, -8.98, -8.02, 100.0]), 2.0, -10.0, 1)
    np.testing.assert_array_equal(encoded, [0, 1, 1, 55])
    assert encoded.dtype == np.dtype("<i2")


@pytest.mark.parametrize(
    ("values", "slope", "intercept", "representation"),
    [
        ([65536], 1.0, 0.0, 0),
        ([-1], 1.0, 0.0, 0),
        ([32768], 1.0, 0.0, 1),
        ([1], 0.0, 0.0, 0),
        ([1], np.inf, 0.0, 0),
        ([1], 1.0, np.nan, 0),
        ([np.nan], 1.0, 0.0, 0),
        ([1], 1.0, 0.0, 2),
    ],
)
def test_fixed_scaling_rejects_invalid_parameters_and_overflow(
    values,
    slope,
    intercept,
    representation,
):
    from nifti2dicom.pixels import encode_fixed_pixels

    with pytest.raises(PixelEncodingError):
        encode_fixed_pixels(np.asarray(values), slope, intercept, representation)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_pixels_fail_before_writing(tmp_path, bad):
    reference = make_reference(slices=1)
    data = np.full((1, 1, 2, 3), bad)
    with pytest.raises(PixelEncodingError):
        read_images(tmp_path, image_for(data, reference), reference)
    assert not list(tmp_path.glob("*.dcm"))


def test_geometry_and_4d_identity_are_generated_consistently(tmp_path):
    reference = make_reference(timepoints=2)
    data = np.arange(24).reshape(2, 2, 2, 3)
    output = read_images(tmp_path, image_for(data, reference), reference)
    assert len({ds.SeriesInstanceUID for ds in output}) == 1
    assert output[0].SeriesInstanceUID != reference.first.SeriesInstanceUID
    assert len({ds.SOPInstanceUID for ds in output}) == 4
    assert [int(ds.InstanceNumber) for ds in output] == [1, 2, 3, 4]
    assert [int(ds.TemporalPositionIdentifier) for ds in output] == [1, 1, 2, 2]
    assert [int(ds.NumberOfTemporalPositions) for ds in output] == [2] * 4
    assert [str(ds.AcquisitionTime) for ds in output] == ["120000", "120000", "120003", "120003"]
    for i, ds in enumerate(output):
        assert ds.ImagePositionPatient == [0, 0, (i % 2) * 4]
        assert ds.ImageOrientationPatient == [1, 0, 0, 0, 1, 0]
        assert ds.PixelSpacing == [3, 2]
        assert float(ds.SliceThickness) == 4
        assert ds.ImageType[:2] == ["DERIVED", "SECONDARY"]
        assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
        assert ds.FrameOfReferenceUID == reference.first.FrameOfReferenceUID
        np.testing.assert_array_equal(ds.pixel_array, data[i // 2, i % 2])


def test_image_dataset_iterator_shares_series_and_encodes_each_plane():
    from nifti2dicom.writers.image import iter_image_datasets

    reference = make_reference()
    data = np.arange(12).reshape(1, 2, 2, 3)
    datasets = iter_image_datasets(image_for(data, reference), reference, description="Derived")
    first = next(datasets)
    second = next(datasets)
    np.testing.assert_array_equal(first.pixel_array, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(second.pixel_array, [[6, 7, 8], [9, 10, 11]])
    assert first.SeriesInstanceUID == second.SeriesInstanceUID
    assert first.SOPInstanceUID != second.SOPInstanceUID
    assert second.SeriesDescription == "Derived"
    with pytest.raises(StopIteration):
        next(datasets)


def test_image_writer_reports_total_while_consuming_dataset_iterator(tmp_path):
    from nifti2dicom.writers.image import write_images

    reference = make_reference()
    events = []
    paths = write_images(
        image_for(np.zeros((1, 2, 2, 3)), reference), reference, tmp_path, on_progress=events.append
    )
    assert len(paths) == 2
    assert [(event.completed, event.total) for event in events] == [(1, 2), (2, 2)]


def test_geometry_mismatch_scrubs_stale_frame_metadata_and_uses_input_timing(tmp_path):
    reference = make_reference(timepoints=2)
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    affine[:3, 3] = [10, 20, 30]
    geometry = Geometry(affine, (3, 2, 2))
    data = np.zeros((3, 2, 2, 3))
    output = read_images(
        tmp_path, image_for(data, reference, geometry=geometry, time_spacing=1.5), reference
    )
    assert [float(ds.TemporalResolution) for ds in output] == [1500] * 6
    for ds in output:
        assert "TriggerTime" not in ds
        assert not getattr(ds, "AcquisitionTime", None)
        assert "SourceImageSequence" not in ds
        assert "FrameReferenceTime" not in ds
        assert "ActualFrameDuration" not in ds  # spacing does not establish acquisition duration
        assert ds.PixelSpacing == [1, 1]
    assert output[1].ImagePositionPatient == [10, 20, 35]


def test_unknown_time_does_not_copy_stale_reference_timing(tmp_path):
    reference = make_reference(timepoints=2)
    output = read_images(tmp_path, image_for(np.zeros((1, 2, 2, 3)), reference), reference)
    for ds in output:
        assert "TemporalPositionIdentifier" not in ds
        assert "FrameReferenceTime" not in ds
        assert "ActualFrameDuration" not in ds
        assert "TriggerTime" not in ds


def test_header_source_cannot_override_encoding_geometry_or_generated_uids(tmp_path):
    reference = make_reference(slices=1)
    del reference.first.FrameOfReferenceUID
    original = deepcopy(reference.first)
    header = Dataset()
    header.PatientName = "Override^Name"
    header.SOPInstanceUID, header.SeriesInstanceUID = generate_uid(), generate_uid()
    header.FrameOfReferenceUID = generate_uid()
    header.StudyInstanceUID = reference.first.StudyInstanceUID
    header.BitsAllocated, header.PixelRepresentation = 8, 0
    header.RescaleSlope, header.RescaleIntercept = "100", "1000"
    header.Rows, header.Columns = 99, 99
    data = np.full((1, 1, 2, 3), -100)
    ds = read_images(tmp_path, image_for(data, reference), reference, header_source=header)[0]
    assert ds.PatientName == "Override^Name"
    assert ds.StudyInstanceUID == reference.first.StudyInstanceUID
    assert ds.SOPInstanceUID not in (header.SOPInstanceUID, reference.first.SOPInstanceUID)
    assert ds.SeriesInstanceUID not in (header.SeriesInstanceUID, reference.first.SeriesInstanceUID)
    assert ds.FrameOfReferenceUID != header.FrameOfReferenceUID
    assert ds.Rows == 2 and ds.Columns == 3 and ds.BitsAllocated == 16
    np.testing.assert_array_equal(
        ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept), -100
    )
    assert reference.first == original


@pytest.mark.parametrize("keyword", ["PatientID", "IssuerOfPatientID", "StudyInstanceUID"])
@pytest.mark.parametrize("rgb", [False, True])
def test_header_source_rejects_conflicting_patient_or_study_identity(tmp_path, keyword, rgb):
    from nifti2dicom.writers.image import write_images
    from nifti2dicom.writers.rgb import write_rgb

    reference = make_reference(slices=1)
    reference.first.IssuerOfPatientID = "HOSPITAL_A"
    header = Dataset()
    setattr(header, keyword, generate_uid() if keyword == "StudyInstanceUID" else "OTHER")
    data = np.zeros((1, 1, 2, 3, 3) if rgb else (1, 1, 2, 3))
    writer = write_rgb if rgb else write_images
    with pytest.raises(ReferenceError, match=keyword):
        writer(
            image_for(data, reference, kind="rgb" if rgb else "image"),
            reference,
            tmp_path,
            header_source=header,
        )
    assert not list(tmp_path.glob("*.dcm"))


def test_mr_profile_has_required_empty_type2_fields(tmp_path):
    reference = make_reference("MR", slices=1)
    ds = read_images(tmp_path, image_for(np.zeros((1, 1, 2, 3)), reference), reference)[0]
    assert ds.SOPClassUID == MRImageStorage
    assert ds.ScanningSequence == "GR"
    for keyword in (
        "ScanOptions",
        "MRAcquisitionType",
        "RepetitionTime",
        "EchoTime",
        "EchoTrainLength",
        "PatientBirthDate",
        "PatientSex",
        "StudyDate",
        "StudyTime",
        "ReferringPhysicianName",
        "StudyID",
        "AccessionNumber",
    ):
        assert keyword in ds


@pytest.mark.parametrize("modality", ["CT", "MR"])
@pytest.mark.parametrize("position", ["", "FFS"])
@pytest.mark.parametrize("matched", [False, True])
def test_ct_and_mr_include_known_or_empty_patient_position(tmp_path, modality, position, matched):
    reference = make_reference(modality, slices=1)
    if position:
        reference.first.PatientPosition = position
    affine = reference.geometry.affine.copy()
    if not matched:
        affine[0, 3] += 10
    image = image_for(
        np.zeros((1, 1, 2, 3)), reference, geometry=Geometry(affine, reference.geometry.size)
    )
    ds = read_images(tmp_path, image, reference)[0]
    assert "PatientPosition" in ds
    assert ds.PatientPosition == (position if matched else "")


@pytest.mark.parametrize("profile", ["CT", "MR", "reformatted_MR"])
@pytest.mark.parametrize(
    ("positions", "expected", "conflict"),
    [
        ((None, None, None), "", False),
        (("HFS", "HFS", "HFS"), "HFS", False),
        (("HFP", "HFP", "HFP"), "HFP", False),
        (("HFS", "FFS", "HFS"), "", True),
        (("HFS", None, "HFS"), "", True),
    ],
)
def test_patient_position_is_uniform_across_all_output_timepoints(
    tmp_path, profile, positions, expected, conflict
):
    if profile == "reformatted_MR":
        reference, image, _, _ = sagittal_dynamic_mr(tmp_path)
    else:
        reference = make_reference(profile, timepoints=3)
        image = image_for(np.arange(36).reshape(3, 2, 2, 3), reference)
    for i, source in enumerate(reference.slices):
        position = positions[i // reference.geometry.size[2]]
        if position is not None:
            source.PatientPosition = position
    before = deepcopy(reference.slices)
    warnings = []
    output = read_images(tmp_path / "out", image, reference, on_warning=warnings.append)
    assert all(ds.PatientPosition == expected for ds in output)
    assert any("PatientPosition" in warning for warning in warnings) == conflict
    assert reference.slices == before


@pytest.mark.parametrize("laterality", [None, "L", "R"])
@pytest.mark.parametrize(
    ("modality", "matched"),
    [("CT", False), ("CT", True), ("MR", False), ("MR", True), ("PT", True)],
)
def test_images_include_known_or_empty_series_laterality(tmp_path, modality, laterality, matched):
    reference = make_reference(modality, slices=1)
    if laterality is not None:
        reference.first.Laterality = laterality
    affine = reference.geometry.affine.copy()
    if not matched:
        affine[0, 3] += 10
    geometry = Geometry(affine, reference.geometry.size)
    ds = read_images(
        tmp_path,
        image_for(np.zeros((1, 1, 2, 3)), reference, geometry=geometry),
        reference,
    )[0]
    assert "Laterality" in ds
    assert ds.Laterality == (laterality if matched and laterality is not None else "")


@pytest.mark.parametrize("modality", ["CT", "MR", "PT"])
@pytest.mark.parametrize("other_laterality", [None, "R"])
def test_series_laterality_never_varies_between_output_images(tmp_path, modality, other_laterality):
    reference = make_reference(modality)
    reference.first.Laterality = "L"
    if other_laterality is not None:
        reference.slices[-1].Laterality = other_laterality
    before = deepcopy(reference.slices)
    warnings = []
    output = read_images(
        tmp_path,
        image_for(np.zeros((1, 2, 2, 3)), reference),
        reference,
        on_warning=warnings.append,
    )
    assert all(ds.Laterality == "" for ds in output)
    assert any("Laterality" in warning for warning in warnings)
    assert reference.slices == before


@pytest.mark.parametrize("slices", [1, 2])
def test_ct_omits_optional_spacing_tag_without_losing_physical_geometry(tmp_path, slices):
    from nifti2dicom.readers.dicom import read_reference

    reference = make_reference(slices=slices)
    image = image_for(np.arange(slices * 6).reshape(1, slices, 2, 3), reference)
    output = read_images(tmp_path, image, reference)
    for z, ds in enumerate(output):
        assert "SpacingBetweenSlices" not in ds
        assert ds.ImagePositionPatient == [0, 0, z * 4]
        assert ds.ImageOrientationPatient == [1, 0, 0, 0, 1, 0]
        assert ds.PixelSpacing == [3, 2]
        assert float(ds.SliceThickness) == 4
        np.testing.assert_array_equal(ds.pixel_array, image.data[0, z])
    restored = read_reference(tmp_path)
    np.testing.assert_allclose(restored.geometry.affine, reference.geometry.affine)


@pytest.mark.parametrize("sagittal", [False, True])
def test_ct_image_type_does_not_invent_optional_image_classification(tmp_path, sagittal):
    reference = make_reference(slices=1)
    reference.first.ImageType = ["ORIGINAL", "PRIMARY"]
    affine = reference.geometry.affine.copy()
    if sagittal:
        affine[:3, :3] = [[0, 0, 4], [2, 0, 0], [0, 3, 0]]
    image = image_for(np.zeros((1, 1, 2, 3)), reference, geometry=Geometry(affine, (3, 2, 1)))
    ds = read_images(tmp_path, image, reference)[0]
    assert ds.ImageType == ["DERIVED", "SECONDARY"]
    assert ds.ImageOrientationPatient == ([0, 1, 0, 0, 0, 1] if sagittal else [1, 0, 0, 0, 1, 0])


@pytest.mark.parametrize("classification", ["AXIAL", "LOCALIZER"])
@pytest.mark.parametrize("matched", [False, True])
def test_ct_preserves_known_classification_only_for_matching_frames(
    tmp_path, classification, matched
):
    reference = make_reference()
    for ds in reference.slices:
        ds.ImageType = ["ORIGINAL", "PRIMARY", classification]
    before = deepcopy(reference.slices)
    affine = reference.geometry.affine.copy()
    if not matched:
        affine[0, 3] += 10
    image = image_for(
        np.arange(12).reshape(1, 2, 2, 3),
        reference,
        geometry=Geometry(affine, reference.geometry.size),
    )
    output = read_images(tmp_path, image, reference)
    expected = ["DERIVED", "SECONDARY", classification] if matched else ["DERIVED", "SECONDARY"]
    assert all(ds.ImageType == expected for ds in output)
    np.testing.assert_array_equal(np.stack([ds.pixel_array for ds in output]), image.data[0])
    assert reference.slices == before


@pytest.mark.parametrize("other_classification", [None, "LOCALIZER", "OTHER"])
def test_ct_omits_incomplete_or_conflicting_source_classification(tmp_path, other_classification):
    reference = make_reference()
    reference.slices[-1].ImageType = ["ORIGINAL", "PRIMARY"]
    if other_classification is not None:
        reference.slices[-1].ImageType.append(other_classification)
    output = read_images(tmp_path, image_for(np.zeros((1, 2, 2, 3)), reference), reference)
    assert all(ds.ImageType == ["DERIVED", "SECONDARY"] for ds in output)


@pytest.mark.parametrize(("modality", "classification"), [("MR", "SECONDARY"), ("PT", "PRIMARY")])
def test_ct_compatibility_changes_preserve_mr_and_pet_profiles(tmp_path, modality, classification):
    reference = make_reference(modality, slices=1)
    ds = read_images(tmp_path, image_for(np.zeros((1, 1, 2, 3)), reference), reference)[0]
    assert float(ds.SpacingBetweenSlices) == 4
    assert ds.ImageType == ["DERIVED", classification, "OTHER"]


@pytest.mark.parametrize("modality", ["CT", "MR", "RGB"])
@pytest.mark.parametrize("input_timing", [None, 2.5])
def test_non_pet_profiles_exclude_pet_frame_timing(tmp_path, modality, input_timing):
    from nifti2dicom.writers.rgb import write_rgb

    reference = make_reference("CT" if modality == "RGB" else modality, timepoints=2, slices=1)
    rgb = modality == "RGB"
    image = image_for(
        np.zeros((2, 1, 2, 3, 3) if rgb else (2, 1, 2, 3)),
        reference,
        kind="rgb" if rgb else "image",
        time_spacing=input_timing,
    )
    output = (
        [pydicom.dcmread(p) for p in write_rgb(image, reference, tmp_path)]
        if rgb
        else read_images(tmp_path, image, reference)
    )
    for ds in output:
        assert "FrameReferenceTime" not in ds
        assert "ActualFrameDuration" not in ds


def test_changed_pet_grid_rejects_spacing_without_an_effective_time_origin(tmp_path):
    reference = make_reference("PT", timepoints=2)
    geometry = Geometry(np.diag([1.0, 1.0, 5.0, 1.0]), (3, 2, 2))
    image = image_for(np.zeros((3, 2, 2, 3)), reference, geometry=geometry, time_spacing=1.5)
    with pytest.raises(ReferenceError, match="FrameReferenceTime|timing"):
        read_images(tmp_path, image, reference)
    assert not list(tmp_path.glob("*.dcm"))


def test_pet_checks_explicit_time_spacing_on_every_mapped_slice(tmp_path):
    reference = make_reference("PT", timepoints=2)
    for ds, time in zip(reference.slices, [100, 200, 2600, 4200], strict=True):
        ds.FrameReferenceTime = time
    image = image_for(np.zeros((2, 2, 2, 3)), reference, time_spacing=2.5)
    with pytest.raises(ReferenceError, match="FrameReferenceTime|timing"):
        read_images(tmp_path, image, reference)
    assert not list(tmp_path.glob("*.dcm"))


@pytest.mark.parametrize("modality", ["CT", "MR"])
@pytest.mark.parametrize("input_timing", [None, 2.5])
def test_coherent_native_temporal_resolution_preserves_acquisition_facts(
    tmp_path,
    modality,
    input_timing,
):
    reference = make_reference(modality, timepoints=2)
    for source in reference.slices:
        source.TemporalResolution = 2500
        del source.FrameReferenceTime
    image = image_for(np.zeros((2, 2, 2, 3)), reference, time_spacing=input_timing)
    output = read_images(tmp_path, image, reference)
    assert [float(ds.TemporalResolution) for ds in output] == [2500] * 4
    assert [str(ds.AcquisitionTime) for ds in output] == ["120000", "120000", "120003", "120003"]
    assert all("FrameReferenceTime" not in ds for ds in output)
    if modality == "MR":
        assert [float(ds.EchoTime) for ds in output] == [3, 3, 4, 4]


@pytest.mark.parametrize("modality", ["CT", "MR"])
@pytest.mark.parametrize("conflict", ["input", "last_reference_slice"])
def test_conflicting_native_temporal_resolution_scrubs_acquisition_facts(
    tmp_path,
    modality,
    conflict,
):
    reference = make_reference(modality, timepoints=2)
    for source in reference.slices:
        source.TemporalResolution = 2500
        del source.FrameReferenceTime
    if conflict == "last_reference_slice":
        reference.slices[-1].TemporalResolution = 3000
    input_timing = 3.0 if conflict == "input" else None
    image = image_for(np.zeros((2, 2, 2, 3)), reference, time_spacing=input_timing)
    output = read_images(tmp_path, image, reference)
    for ds in output:
        assert not getattr(ds, "AcquisitionTime", None)
        if conflict == "input":
            assert float(ds.TemporalResolution) == 3000
        else:
            assert "TemporalResolution" not in ds
        if modality == "MR":
            assert ds.data_element("EchoTime").is_empty


def sagittal_dynamic_mr(tmp_path):
    from nifti2dicom.readers import read_nifti

    reference = make_reference("MR", timepoints=3, slices=4)
    affine = np.array([[0, 0, 4, 11], [2, 0, 0, 13], [0, 3, 0, 17], [0, 0, 0, 1]])
    reference = replace(reference, geometry=Geometry(affine, (3, 2, 4)))
    times = ("120000", "120017.49", "120126.94")
    for i, source in enumerate(reference.slices):
        t, z = divmod(i, 4)
        source.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
        source.ImagePositionPatient = [11 + 4 * z, 13, 17]
        source.AcquisitionTime = times[t]
        source.AcquisitionDate = "20260915"
        del source.FrameReferenceTime
        source.LossyImageCompression = "01"
        source.LossyImageCompressionRatio = 5 + z
    data = np.arange(72, dtype=np.int16).reshape(3, 2, 4, 3)
    nifti = nib.Nifti1Image(data, np.diag([-1, -1, 1, 1]) @ affine)
    nifti.header.set_xyzt_units("mm", "unknown")
    path = tmp_path / "sagittal.nii"
    nib.save(nifti, path)
    return reference, read_nifti(path), data, times


def test_canonical_sagittal_mr_preserves_irregular_volume_times_and_world_voxels(tmp_path):
    reference, image, data, times = sagittal_dynamic_mr(tmp_path)
    assert image.geometry.size == (4, 3, 2)
    output = read_images(tmp_path / "out", image, reference)
    seen = set()
    for ds in output:
        t = int(ds.TemporalPositionIdentifier) - 1
        assert str(ds.AcquisitionTime) == times[t]
        assert float(ds.EchoTime) == 3 + t
        assert "TemporalResolution" not in ds
        assert "LossyImageCompressionRatio" not in ds
        assert "SourceImageSequence" not in ds
        for row in range(ds.Rows):
            for col in range(ds.Columns):
                world = (
                    np.asarray(ds.ImagePositionPatient, float)
                    + np.asarray(ds.ImageOrientationPatient[:3], float)
                    * col
                    * float(ds.PixelSpacing[1])
                    + np.asarray(ds.ImageOrientationPatient[3:], float)
                    * row
                    * float(ds.PixelSpacing[0])
                )
                x, y, z = (
                    int(round((world[1] - 13) / 2)),
                    int(round((world[2] - 17) / 3)),
                    int(round((world[0] - 11) / 4)),
                )
                np.testing.assert_allclose(world, [11 + 4 * z, 13 + 2 * x, 17 + 3 * y], atol=1e-6)
                assert (
                    ds.pixel_array[row, col] * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                    == data[x, y, z, t]
                )
                seen.add((x, y, z, t))
    assert len(seen) == data.size


@pytest.mark.parametrize("field", ["EchoTime", "TriggerTime", "AcquisitionTime"])
def test_reformatted_mr_omits_nonuniform_plane_facts_with_warning(tmp_path, field):
    reference, image, _, times = sagittal_dynamic_mr(tmp_path)
    if field == "TriggerTime":
        for ds in reference.slices:
            ds.ScanOptions = "CG"
    setattr(reference.slices[1], field, "120001" if field == "AcquisitionTime" else 999)
    warnings = []
    output = read_images(tmp_path / "out", image, reference, on_warning=warnings.append)
    assert warnings
    assert all(not getattr(ds, field, None) for ds in output[:2])
    if field != "AcquisitionTime":
        assert [str(ds.AcquisitionTime) for ds in output] == [v for v in times for _ in range(2)]


@pytest.mark.parametrize(
    "conflict", ["cropped", "translated", "timing", "incomplete", "missing_volume", "only_volume"]
)
def test_reformatted_mr_does_not_assign_unproven_temporal_metadata(tmp_path, conflict):
    reference, image, _, _ = sagittal_dynamic_mr(tmp_path)
    if conflict == "cropped":
        image = replace(
            image,
            data=image.data[:, :, :, :-1],
            geometry=Geometry(image.geometry.affine, (3, 3, 2)),
        )
    elif conflict == "translated":
        affine = image.geometry.affine.copy()
        affine[0, 3] += 0.01
        image = replace(image, geometry=Geometry(affine, image.geometry.size))
    elif conflict == "timing":
        image = replace(image, time_spacing=17.49)
    elif conflict == "incomplete":
        del reference.slices[-1].TemporalPositionIdentifier
    else:
        available = 1 if conflict == "only_volume" else 2
        reference = replace(
            reference, slices=reference.slices[: available * 4], timepoints=available
        )
        image = replace(image, data=image.data[:available])
    warnings = []
    output = read_images(tmp_path / "out", image, reference, on_warning=warnings.append)
    assert warnings
    assert all(not getattr(ds, "AcquisitionTime", None) for ds in output)
    assert all(not getattr(ds, "EchoTime", None) for ds in output)
    if conflict == "timing":
        assert all(float(ds.TemporalResolution) == 17490 for ds in output)


@pytest.mark.parametrize("empty_source_resolution", [False, True])
def test_reformatted_mr_accepts_agreeing_explicit_sample_spacing(tmp_path, empty_source_resolution):
    reference, image, _, _ = sagittal_dynamic_mr(tmp_path)
    for i, ds in enumerate(reference.slices):
        ds.AcquisitionTime = ("120000", "120017.49", "120034.98")[i // 4]
        if empty_source_resolution:
            ds.TemporalResolution = ""
    image = replace(image, time_spacing=17.49)
    output = read_images(tmp_path / "out", image, reference)
    assert [str(ds.AcquisitionTime) for ds in output] == [
        "120000",
        "120000",
        "120017.49",
        "120017.49",
        "120034.98",
        "120034.98",
    ]
    assert all(float(ds.TemporalResolution) == 17490 for ds in output)


@pytest.mark.parametrize("conflict", [None, "slice", "volume"])
def test_reformatted_mr_preserves_only_series_uniform_laterality(tmp_path, conflict):
    reference, image, _, _ = sagittal_dynamic_mr(tmp_path)
    for ds in reference.slices:
        ds.Laterality = "L"
    if conflict == "slice":
        reference.slices[-1].Laterality = "R"
    elif conflict == "volume":
        for ds in reference.slices[-4:]:
            ds.Laterality = "R"
    before = deepcopy(reference.slices)
    warnings = []
    output = read_images(tmp_path / "out", image, reference, on_warning=warnings.append)
    assert all(ds.Laterality == ("L" if conflict is None else "") for ds in output)
    assert bool(warnings) == (conflict is not None)
    assert reference.slices == before


@pytest.mark.parametrize("reformatted", [False, True])
@pytest.mark.parametrize(
    ("scan_options", "trigger_time", "expected"),
    [
        (None, "", None),
        ("RG", 123, None),
        ("CG", None, ""),
        (["FS", "PPG"], 12.5, 12.5),
        ("CG", 0, 0),
    ],
)
def test_mr_trigger_time_follows_output_heart_gating(
    tmp_path, reformatted, scan_options, trigger_time, expected
):
    if reformatted:
        reference, image, _, _ = sagittal_dynamic_mr(tmp_path)
    else:
        reference = make_reference("MR")
        image = image_for(np.arange(12).reshape(1, 2, 2, 3), reference)
    for ds in reference.slices:
        if scan_options is not None:
            ds.ScanOptions = scan_options
        if trigger_time is None:
            del ds.TriggerTime
        else:
            ds.TriggerTime = trigger_time
    before = deepcopy(reference.slices)
    output = read_images(tmp_path / "out", image, reference)
    for ds in output:
        if expected is None:
            assert "TriggerTime" not in ds
        elif expected == "":
            assert ds.data_element("TriggerTime").is_empty
        else:
            assert float(ds.TriggerTime) == expected
    assert reference.slices == before


@pytest.mark.parametrize("bad_time", [np.nan, np.inf])
def test_pet_rejects_nonfinite_effective_reference_times(tmp_path, bad_time):
    reference = make_reference("PT")
    reference.slices[1].FrameReferenceTime = bad_time
    image = image_for(np.zeros((1, 2, 2, 3)), reference)
    with pytest.raises(ReferenceError, match="FrameReferenceTime|timing"):
        read_images(tmp_path, image, reference)
    assert not list(tmp_path.glob("*.dcm"))


@pytest.mark.parametrize("reverse_slices", [False, True])
def test_pet_native_timing_roundtrips_without_mr_temporal_tags(tmp_path, reverse_slices):
    from nifti2dicom.readers.dicom import read_reference
    from nifti2dicom.writers.image import write_images

    reference = make_reference("PT", timepoints=2)
    for ds, time in zip(reference.slices, [100, 200, 2600, 2700], strict=True):
        ds.FrameReferenceTime = time
    affine = reference.geometry.affine.copy()
    if reverse_slices:
        affine[:3, 2] *= -1
        affine[:3, 3] = [0, 0, 4]
    geometry = Geometry(affine, (3, 2, 2))
    image = image_for(
        np.arange(24).reshape(2, 2, 2, 3), reference, geometry=geometry, time_spacing=2.5
    )
    paths = write_images(image, reference, tmp_path)
    for path in paths:
        ds = pydicom.dcmread(path)
        assert "TemporalPositionIdentifier" not in ds
        assert "NumberOfTemporalPositions" not in ds
        assert "TemporalResolution" not in ds
        assert int(ds.NumberOfTimeSlices) == 2
        assert int(ds.NumberOfSlices) == 2
    restored = read_reference(tmp_path)
    assert restored.timepoints == 2
    assert restored.geometry.size == (3, 2, 2)
    assert [float(ds.FrameReferenceTime) for ds in restored.slices] == [100, 200, 2600, 2700]
    assert [int(ds.ImageIndex) for ds in restored.slices] == [1, 2, 3, 4]


def test_unsupported_modality_fails_clearly(tmp_path):
    reference = make_reference("US", slices=1)
    with pytest.raises(UnsupportedInputError, match="US"):
        read_images(tmp_path, image_for(np.zeros((1, 1, 2, 3)), reference), reference)


@pytest.mark.parametrize(
    ("modality", "missing"),
    [
        ("MR", "ScanningSequence"),
        ("MR", "SequenceVariant"),
        ("PT", "Units"),
        ("PT", "CountsSource"),
        ("PT", "FrameReferenceTime"),
        ("PT", "DecayCorrection"),
    ],
)
def test_incomplete_required_acquisition_metadata_fails_before_writing(tmp_path, modality, missing):
    reference = make_reference(modality, slices=1)
    delattr(reference.first, missing)
    with pytest.raises(ReferenceError, match=missing):
        read_images(tmp_path, image_for(np.zeros((1, 1, 2, 3)), reference), reference)
    assert not list(tmp_path.glob("*.dcm"))


def test_pet_series_type_must_be_a_supported_pair(tmp_path):
    reference = make_reference("PT", slices=1)
    reference.first.SeriesType = "INVALID"
    with pytest.raises(ReferenceError, match="SeriesType"):
        read_images(tmp_path, image_for(np.zeros((1, 1, 2, 3)), reference), reference)


@pytest.mark.parametrize("reverse_slices", [False, True])
def test_pet_uses_physical_frame_matching_after_inplane_flips(tmp_path, reverse_slices):
    reference = make_reference("PT", timepoints=2)
    for index, ds in enumerate(reference.slices):
        ds.FrameReferenceTime = [100, 200, 2600, 2700][index]
    affine = np.diag([-2.0, -3.0, -4.0 if reverse_slices else 4.0, 1.0])
    affine[:3, 3] = [4, 3, 4 if reverse_slices else 0]
    geometry = Geometry(affine, (3, 2, 2))
    data = np.arange(24).reshape(2, 2, 2, 3)
    output = read_images(tmp_path, image_for(data, reference, geometry=geometry), reference)
    expected_times = [200, 100, 2700, 2600] if reverse_slices else [100, 200, 2600, 2700]
    expected_indices = [2, 1, 4, 3] if reverse_slices else [1, 2, 3, 4]
    assert [float(ds.FrameReferenceTime) for ds in output] == expected_times
    assert [int(ds.ImageIndex) for ds in output] == expected_indices
    for i, ds in enumerate(output):
        np.testing.assert_array_equal(ds.pixel_array, data[i // 2, i % 2])


def test_pet_preserves_timing_after_nifti1_affine_roundtrip(tmp_path):
    import nibabel as nib

    from nifti2dicom.readers.nifti import nifti_to_volume

    reference = make_reference("PT", timepoints=2, slices=3)
    affine = np.diag([4.07283, 4.07283, 2.027, 1.0])
    affine[:3, 3] = [-407.037, -566.085, -1253.96]
    reference = ReferenceSeries(reference.slices, (), Geometry(affine, (3, 2, 3)), timepoints=2)
    for index, ds in enumerate(reference.slices):
        ds.ImagePositionPatient = reference.geometry.position(index % 3).tolist()
        ds.FrameReferenceTime = [100, 200, 300, 2600, 2700, 2800][index]
    ras_affine = affine.copy()
    ras_affine[:2] *= -1
    raw = np.arange(36, dtype=np.float32).reshape(3, 2, 3, 2)
    nifti = nib.Nifti1Image(raw, ras_affine)
    nifti.header.set_xyzt_units("mm", "sec")
    nifti.header.set_zooms((4.07283, 4.07283, 2.027, 2.5))
    # Serialization stores the affine coefficients as float32, unlike the
    # original in-memory Nifti1Image's float64 affine.
    image = nifti_to_volume(nib.Nifti1Image.from_bytes(nifti.to_bytes()))
    output = read_images(tmp_path, image, reference)
    assert [float(ds.FrameReferenceTime) for ds in output] == [100, 200, 300, 2600, 2700, 2800]
    assert [int(ds.ImageIndex) for ds in output] == [1, 2, 3, 4, 5, 6]
    for index, ds in enumerate(output):
        t, z = divmod(index, 3)
        np.testing.assert_array_equal(ds.pixel_array, raw[::-1, ::-1, z, t].T)
        np.testing.assert_allclose(ds.ImagePositionPatient, image.geometry.position(z), atol=1e-9)


@pytest.mark.parametrize("change", ["translation", "spacing_drift", "subtle_drift", "large_origin"])
def test_pet_roundoff_allowance_does_not_accept_changed_physical_grid(tmp_path, change):
    slices = 419 if change in {"spacing_drift", "subtle_drift"} else 3
    reference = make_reference("PT", slices=slices)
    affine = np.diag([4.07283, 4.07283, 2.027, 1.0])
    affine[:3, 3] = [-407.037, -566.085, 1e6 if change == "large_origin" else -1253.96]
    reference = ReferenceSeries(reference.slices, (), Geometry(affine, (3, 2, slices)))
    changed = affine.copy()
    if change == "spacing_drift":
        changed[2, 2] = 2.0299072265625
    elif change == "subtle_drift":
        changed[2, 2] += 2e-6  # Near-unit voxel scale still drifts beyond corner precision.
    else:
        changed[2, 3] += 0.002 if change == "large_origin" else 0.001
    image = image_for(
        np.zeros((1, slices, 2, 3)), reference, geometry=Geometry(changed, (3, 2, slices))
    )
    with pytest.raises(ReferenceError, match="FrameReferenceTime|timing"):
        read_images(tmp_path, image, reference)
    assert not list(tmp_path.glob("*.dcm"))


def test_empty_reference_has_actionable_error(tmp_path):
    reference = make_reference(slices=1)
    empty = ReferenceSeries((), (), reference.geometry)
    with pytest.raises(ReferenceError, match="empty"):
        read_images(tmp_path, image_for(np.zeros((1, 1, 2, 3)), reference), empty)


@pytest.mark.parametrize("slices", [1, 2])
def test_rgb_2d_and_3d_decodes_channel_order_and_geometry(tmp_path, slices):
    from nifti2dicom.writers.rgb import write_rgb

    reference = make_reference(slices=slices)
    data = np.arange(slices * 18, dtype=np.uint8).reshape(1, slices, 2, 3, 3)
    output = write_rgb(image_for(data, reference, kind="rgb"), reference, tmp_path)
    assert len(output) == slices
    for z, path in enumerate(output):
        ds = pydicom.dcmread(path)
        np.testing.assert_array_equal(ds.pixel_array, data[0, z])
        assert ds.SOPClassUID == SecondaryCaptureImageStorage
        assert ds.ConversionType == "WSD"
        assert ds.PhotometricInterpretation == "RGB"
        assert ds.SamplesPerPixel == 3 and ds.PlanarConfiguration == 0
        assert ds.PixelSpacing == [3, 2]
        assert ds.ImagePositionPatient == [0, 0, z * 4]
        assert "RescaleSlope" not in ds and "RescaleIntercept" not in ds


@pytest.mark.parametrize("bad", [-1, 256, 12.5, np.nan])
def test_rgb_rejects_values_that_would_wrap_or_truncate(tmp_path, bad):
    from nifti2dicom.writers.rgb import write_rgb

    reference = make_reference(slices=1)
    data = np.full((1, 1, 2, 3, 3), bad)
    with pytest.raises(PixelEncodingError):
        write_rgb(image_for(data, reference, kind="rgb"), reference, tmp_path)
    assert not list(tmp_path.glob("*.dcm"))
