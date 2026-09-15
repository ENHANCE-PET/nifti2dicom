"""QA-oracle regressions, independent of Slicer and converter internals."""

import numpy as np
import pytest
from pydicom.dataset import Dataset


def test_pet_timing_follows_physical_source_slice_instead_of_first_slice():
    from public_oracle import check_plane_timing

    truth = {
        "frame_reference_times_ms": np.array([[1000.0, 1007.0]]),
        "frame_durations_ms": np.array([[120000.0, 120007.0]]),
        "times_ms": np.array([1000.0]),
        "durations_ms": np.array([120000.0]),
    }
    ds = Dataset()
    ds.Modality = "PT"
    ds.FrameReferenceTime, ds.ActualFrameDuration = 1007, 120007
    check_plane_timing(ds, truth, 0, np.array([1, 1, 1]), "PT")
    with pytest.raises(AssertionError):
        check_plane_timing(ds, truth, 0, np.array([0, 0, 0]), "PT")


def test_legacy_pet_truth_still_checks_time_and_duration():
    from public_oracle import check_plane_timing

    truth = {"times_ms": np.array([0, 17000]), "durations_ms": np.array([2000, 4000])}
    ds = Dataset()
    ds.Modality = "PT"
    ds.FrameReferenceTime, ds.ActualFrameDuration = 17000, 4000
    check_plane_timing(ds, truth, 1, np.array([0, 1]), "PT")
    ds.ActualFrameDuration = 2000
    with pytest.raises(AssertionError):
        check_plane_timing(ds, truth, 1, np.array([0, 1]), "PT")


def test_mr_time_survives_plane_axis_permutation_and_detects_time_reversal():
    from public_oracle import check_plane_timing

    truth = {"acquisition_times": np.array(["152306.48", "152323.97", "152433.42"])}
    ds = Dataset()
    ds.Modality = "MR"
    ds.AcquisitionTime, ds.TemporalPositionIdentifier = "152323.97", 2
    check_plane_timing(ds, truth, 1, np.array([0, 1, 2, 3]), "MR")
    ds.AcquisitionTime = "152433.42"
    with pytest.raises(AssertionError):
        check_plane_timing(ds, truth, 1, np.array([0, 1, 2, 3]), "MR")


def test_stock_mr_sequence_ordinals_map_to_irregular_acquisition_elapsed_times():
    from public_oracle import check_sequence_timing

    truth = {"acquisition_times": np.array(["152306.48", "152323.97", "152433.42"])}
    attributes = {
        "MultiVolume.FrameIdentifyingDICOMTagName": "AcquisitionTime",
        "MultiVolume.FrameIdentifyingDICOMTagUnits": "ms",
        "MultiVolume.FrameLabels": "0,17490,86940",
    }
    check_sequence_timing(["0", "1", "2"], "", attributes, truth, "MR")
    attributes["MultiVolume.FrameLabels"] = "0,17490,34980"
    with pytest.raises(AssertionError):
        check_sequence_timing(["0", "1", "2"], "", attributes, truth, "MR")


def test_pet_sequence_uses_reference_time_without_zeroing_it():
    from public_oracle import check_sequence_timing

    truth = {"times_ms": np.array([95996, 100996, 105996])}
    check_sequence_timing(["95996", "100996", "105996"], "ms", {}, truth, "PT")
    with pytest.raises(AssertionError):
        check_sequence_timing(["0", "5000", "10000"], "ms", {}, truth, "PT")


@pytest.mark.parametrize("field", ["loaded", "serialized", "expected"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_pixels_cannot_pass_error_or_quantization_checks(field, nonfinite):
    from public_oracle import check_frame_pixels

    arrays = {name: np.array([1.0, 2.0]) for name in ("loaded", "serialized", "expected")}
    arrays[field][1] = nonfinite
    with pytest.raises(AssertionError, match="finite"):
        check_frame_pixels(**arrays, rescale_slope=0.5)


def test_pixel_errors_keep_per_plane_quantization_bound():
    from public_oracle import check_frame_pixels

    result = check_frame_pixels(
        np.array([0.0, 2.0, 6.0]),
        np.array([0.0, 1.5, 6.0]),
        np.array([0.0, 1.4, 6.2]),
        rescale_slope=0.4,
    )
    assert result == (0.5, 1)


@pytest.mark.parametrize("flat", [[0, 1, 2, 3, 1], [0, 1, 1, 3], [0, 1, 3], []])
def test_physical_coverage_rejects_extra_duplicate_or_missing_voxels(flat):
    from public_oracle import check_physical_coverage

    with pytest.raises(AssertionError):
        check_physical_coverage(np.asarray(flat, dtype=int), 4)


def test_physical_coverage_accepts_complete_reordered_grid():
    from public_oracle import check_physical_coverage

    check_physical_coverage(np.array([2, 0, 3, 1]), 4)


@pytest.mark.parametrize(
    "requested,sequences,multivolumes",
    [("sequence", 0, 1), ("multivolume", 1, 0), ("sequence", 1, 1), ("multivolume", 0, 0)],
)
def test_requested_representation_cannot_pass_when_another_or_ambiguous_one_loads(
    requested, sequences, multivolumes
):
    from public_oracle import check_representation

    with pytest.raises(AssertionError):
        check_representation(requested, sequences, multivolumes)


@pytest.mark.parametrize(
    "requested,sequences,multivolumes", [("sequence", 1, 0), ("multivolume", 0, 1)]
)
def test_requested_representation_accepts_one_matching_loaded_node(
    requested, sequences, multivolumes
):
    from public_oracle import check_representation

    check_representation(requested, sequences, multivolumes)
