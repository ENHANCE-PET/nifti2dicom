"""Pure parser regressions: real DICOM headers, without Slicer or converter imports."""

import sys
import traceback
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import pydicom
import pytest
from fixtures import pet_series
from pydicom.datadict import tag_for_keyword
from pydicom.dataelem import RawDataElement
from pydicom.tag import Tag
from pydicom.uid import generate_uid

sys.path.insert(0, str(Path(__file__).parents[1]))


def group(records):
    from Nifti2DicomPETLib.frames import group_frames

    return group_frames(records)


@pytest.fixture
def records(tmp_path):
    paths, _ = pet_series(tmp_path / "dicom")
    return [(str(path), pydicom.dcmread(path, stop_before_pixels=True)) for path in paths]


def test_native_indices_survive_filename_instance_and_selection_disorder(records):
    # InstanceNumber and filename order both contradict the authored spatial order.
    result = group([records[i] for i in [5, 2, 0, 4, 1, 3]])
    assert result.frame_files == (
        (records[0][0], records[1][0]),
        (records[2][0], records[3][0]),
        (records[4][0], records[5][0]),
    )
    assert tuple(map(float, result.index_values)) == (0.0, 2000.0, 17000.0)
    assert result.index_name == result.tag_name == "FrameReferenceTime"
    assert result.index_unit == "ms"


@pytest.mark.parametrize("varying_acquisition", [False, True])
def test_slice_specific_times_keep_native_time_blocks_and_original_headers(
    tmp_path, varying_acquisition
):
    paths, _ = pet_series(
        tmp_path / "dicom", varying_acquisition=varying_acquisition, slice_times=True
    )
    records = [(str(path), pydicom.dcmread(path, stop_before_pixels=True)) for path in paths]
    originals = [deepcopy(ds) for _, ds in records]
    result = group(records)
    assert tuple(map(len, result.frame_files)) == (2, 2, 2)
    assert result.index_values == ("1", "2", "3")
    assert result.index_name == "TimeSlice"
    assert result.index_unit == "count"
    assert result.tag_name == "ImageIndex"
    assert [ds for _, ds in records] == originals


def test_nearly_equal_slice_times_are_not_averaged(records):
    records[1][1].FrameReferenceTime = "0.000000001"
    result = group(records)
    assert result.index_values == ("1", "2", "3")
    assert result.index_unit == "count"


def test_uniform_fractional_times_retain_recorded_index_values(records):
    values = ["0.125", "2000.0625", "17000.875"]
    for i, (_, ds) in enumerate(records):
        ds.FrameReferenceTime = values[i // 2]
    assert group(records).index_values == tuple(values)


@pytest.mark.parametrize("selected", [[], [0], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 5]])
def test_incomplete_or_excess_selection_is_rejected(records, selected):
    with pytest.raises(ValueError, match="PET"):
        group([records[i] for i in selected])


@pytest.mark.parametrize("value", [0, 1, 7])
def test_image_index_must_cover_dense_unique_native_range(records, value):
    records[-1][1].ImageIndex = value
    with pytest.raises(ValueError, match="PET.*ImageIndex"):
        group(records)


@pytest.mark.parametrize(
    "field,value",
    [
        ("SOPClassUID", "1.2.840.10008.5.1.4.1.1.130"),
        ("Modality", "MR"),
        ("SeriesType", ["STATIC", "IMAGE"]),
        ("SeriesType", ["DYNAMIC", "REPROJECTION"]),
        ("NumberOfFrames", 2),
        ("NumberOfSlices", 0),
        ("NumberOfSlices", 3),
        ("NumberOfTimeSlices", 0),
        ("NumberOfTimeSlices", 2),
        ("Rows", 5),
        ("Columns", 7),
        ("PixelSpacing", [1.5, 3.5]),
        ("PixelSpacing", [0, 2.5]),
        ("ImageOrientationPatient", [0, 1, 0, 1, 0, 0]),
        ("ImageOrientationPatient", [1, 0, 0, 1, 0, 0]),
        ("ImagePositionPatient", [-11, 23, -5]),
        ("ImagePositionPatient", [-12, 23, -4]),
        ("BitsAllocated", 8),
        ("BitsStored", 12),
        ("HighBit", 14),
        ("PixelRepresentation", 1),
        ("PixelRepresentation", 2),
        ("SamplesPerPixel", 3),
        ("PhotometricInterpretation", "MONOCHROME1"),
        ("Units", "CNTS"),
        ("DecayCorrection", "START"),
        ("PatientID", "ANOTHER-PATIENT"),
        ("PatientName", "Other^Patient"),
        ("StudyInstanceUID", "1.2.3.4"),
        ("SeriesInstanceUID", "1.2.3.4"),
        ("FrameOfReferenceUID", "1.2.3.4"),
        ("RescaleSlope", "0"),
        ("RescaleSlope", "-1"),
        ("RescaleIntercept", "0.1"),
    ],
)
def test_mixed_or_invalid_metadata_is_rejected(records, field, value):
    setattr(records[-1][1], field, value)
    with pytest.raises(ValueError, match="PET"):
        group(records)


@pytest.mark.parametrize(
    "field",
    [
        "SOPClassUID",
        "SOPInstanceUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "FrameOfReferenceUID",
        "SeriesType",
        "NumberOfSlices",
        "NumberOfTimeSlices",
        "ImageIndex",
        "FrameReferenceTime",
        "Rows",
        "Columns",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "Units",
        "RescaleSlope",
        "RescaleIntercept",
    ],
)
def test_required_metadata_cannot_be_missing(records, field):
    delattr(records[-1][1], field)
    with pytest.raises(ValueError, match="PET"):
        group(records)


@pytest.mark.parametrize("field", ["FrameReferenceTime", "RescaleSlope", "RescaleIntercept"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", ""])
def test_nonfinite_and_empty_numeric_metadata_is_rejected(records, field, value):
    # DS deliberately permits these malformed values in pydicom's default mode.
    with pytest.warns(UserWarning) if value else nullcontext():
        setattr(records[-1][1], field, value)
    with pytest.raises(ValueError, match="PET"):
        group(records)


@pytest.mark.parametrize("last_time", ["1999", "2000"])
def test_each_corresponding_slice_time_must_increase(records, last_time):
    records[-1][1].FrameReferenceTime = last_time
    with pytest.raises(ValueError, match="PET.*FrameReferenceTime"):
        group(records)


def test_different_slices_may_have_overlapping_time_ranges(records):
    for i, (_, ds) in enumerate(records):
        ds.FrameReferenceTime = ["0", "4000", "2000", "6000", "17000", "21000"][i]
    assert group(records).index_values == ("1", "2", "3")


def test_duplicate_sop_identity_is_rejected(records):
    records[-1][1].SOPInstanceUID = records[0][1].SOPInstanceUID
    with pytest.raises(ValueError, match="PET.*SOPInstanceUID"):
        group(records)


def test_duplicate_selected_path_is_rejected(records):
    records[-1] = (records[0][0], records[-1][1])
    with pytest.raises(ValueError, match="PET"):
        group(records)


@pytest.mark.parametrize("reversed_positions", [False, True])
def test_native_slice_indices_require_unique_ascending_normal_positions(
    records, reversed_positions
):
    for i, (_, ds) in enumerate(records):
        ds.ImagePositionPatient = [-12, 23, -5 - (4 * (i % 2) if reversed_positions else 0)]
    with pytest.raises(ValueError, match="PET.*(position|order|geometry)"):
        group(records)


def test_signed_series_with_per_image_slopes_is_supported(records):
    for _, ds in records:
        ds.PixelRepresentation = 1
    assert len(group(records).frame_files) == 3


def _three_slice_records(records, z_positions):
    expanded = []
    for t in range(3):
        for z, position in enumerate(z_positions):
            ds = deepcopy(records[t * 2][1])
            ds.SOPInstanceUID = generate_uid()
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.NumberOfSlices = len(z_positions)
            ds.ImageIndex = t * len(z_positions) + z + 1
            ds.ImagePositionPatient = [-12, 23, position]
            expanded.append((f"t{t}-z{z}.dcm", ds))
    return expanded


def test_irregular_slice_grid_is_rejected(records):
    with pytest.raises(ValueError, match="PET.*(regular|spacing|geometry)"):
        group(_three_slice_records(records, [-9, -5, 0]))


def test_rounded_regular_grid_is_supported_without_resampling(records):
    expanded = _three_slice_records(records, ["-9.000", "-5.723", "-2.445"])
    before = [tuple(ds.ImagePositionPatient) for _, ds in expanded]
    assert tuple(map(len, group(expanded).frame_files)) == (3, 3, 3)
    assert [tuple(ds.ImagePositionPatient) for _, ds in expanded] == before


def test_rounding_cannot_exceed_native_import_geometry_tolerance(records):
    with pytest.raises(ValueError, match="PET.*(regular|spacing|geometry)"):
        group(_three_slice_records(records, ["-9.00", "-5.73", "-2.45"]))


def test_oblique_positions_follow_cross_product_with_correct_pixel_axis_spacing(records):
    # Independently specified orthonormal directions and world positions.
    for i, (_, ds) in enumerate(records):
        ds.ImageOrientationPatient = [0, 1, 0, 0.6, 0, 0.8]
        ds.ImagePositionPatient = [-12 + 3.2 * (i % 2), 23, -9 - 2.4 * (i % 2)]
    result = group(records)
    headers = dict(records)
    for files in result.frame_files:
        np.testing.assert_allclose(headers[files[0]].ImagePositionPatient, [-12, 23, -9])
        np.testing.assert_allclose(headers[files[1]].ImagePositionPatient, [-8.8, 23, -11.4])


@pytest.mark.parametrize("field,value", [("SeriesDate", "20260915"), ("SeriesTime", "120001")])
def test_frame_reference_times_require_one_shared_series_time_origin(records, field, value):
    setattr(records[-1][1], field, value)
    with pytest.raises(ValueError, match="PET"):
        group(records)


def test_numeric_sequence_indices_cannot_collapse_distinct_recorded_times(records):
    # Deliberately malformed overlong DS values can differ yet collide in VTK double indices.
    with pytest.warns(UserWarning):
        for i, (_, ds) in enumerate(records):
            ds.FrameReferenceTime = ["1000000000000000", "1000000000000000.01", "1000000000000001"][
                i // 2
            ]
    with pytest.raises(ValueError, match="PET.*FrameReferenceTime"):
        group(records)


def test_empty_series_type_raises_a_pet_validation_error(records):
    records[-1][1].SeriesType = None
    with pytest.raises(ValueError, match="PET.*SeriesType"):
        group(records)


def test_intercept_underflow_cannot_hide_a_nonzero_value(records):
    records[-1][1].RescaleIntercept = "1e-999"
    with pytest.raises(ValueError, match="PET.*RescaleIntercept"):
        group(records)


@pytest.mark.parametrize("field", ["MediaStorageSOPClassUID", "MediaStorageSOPInstanceUID"])
def test_file_meta_identity_must_agree_with_dataset_identity(records, field):
    setattr(records[-1][1].file_meta, field, "1.2.3.4")
    with pytest.raises(ValueError, match="PET.*identity"):
        group(records)


@pytest.mark.parametrize("value", [None, ""])
def test_units_cannot_be_unknown_in_every_frame(records, value):
    for _, ds in records:
        ds.Units = value
    with pytest.raises(ValueError, match="PET.*Units"):
        group(records)


def test_nontext_series_type_raises_a_pet_validation_error(records):
    with pytest.warns(UserWarning):
        records[-1][1].SeriesType = 7
    with pytest.raises(ValueError, match="PET.*SeriesType"):
        group(records)


@pytest.mark.parametrize(
    "field",
    [
        "NumberOfSlices",
        "NumberOfTimeSlices",
        "Rows",
        "Columns",
        "StudyInstanceUID",
        "PatientID",
        "MediaStorageSOPInstanceUID",
    ],
)
def test_lazy_element_decode_failure_is_a_pet_error_without_raw_header_contents(records, field):
    # Raw elements model dcmread's delayed conversion. The declared US length
    # is invalid, including when corrupt VR bytes occur in an identity field.
    header = records[-1][1]
    target = header.file_meta if field.startswith("MediaStorage") else header
    tag = Tag(tag_for_keyword(field))
    raw_value = b"PRIVATE"
    target[tag] = RawDataElement(tag, "US", len(raw_value), raw_value, 0, False, True)
    with pytest.raises(ValueError, match="PET.*(header|metadata)") as error:
        group(records)
    assert raw_value.decode() not in "".join(traceback.format_exception(error.value))
