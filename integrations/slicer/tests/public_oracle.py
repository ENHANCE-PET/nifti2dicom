"""Independent public-data QA assertions; no Slicer or converter imports."""

from decimal import Decimal

import numpy as np


def check_frame_pixels(loaded, serialized, expected, rescale_slope):
    for name, array in (("loaded", loaded), ("serialized", serialized), ("expected", expected)):
        assert np.isfinite(array).all(), f"{name} pixels must be finite"
    error = float(np.max(np.abs(loaded.astype(float) - serialized)))
    violations = int(
        np.count_nonzero(
            np.abs(loaded.astype(float) - expected) > abs(float(rescale_slope)) / 2 + 1e-7
        )
    )
    return error, violations


def check_physical_coverage(flat, truth_size):
    assert flat.size == truth_size, "Missing or extra physical locations"
    assert np.unique(flat).size == flat.size, "Duplicate physical locations"


def check_representation(requested, sequences, multivolumes):
    expected = {"sequence": (1, 0), "multivolume": (0, 1)}[requested]
    assert (sequences, multivolumes) == expected, (
        f"Requested {requested}, loaded {sequences} sequences and {multivolumes} multivolumes"
    )


def acquisition_elapsed_ms(truth):
    """Decode full DICOM TM values independently of the receiver's parser."""
    absolute = []
    for value in truth["acquisition_times"]:
        text = str(value)
        absolute.append((3600 * int(text[:2]) + 60 * int(text[2:4]) + Decimal(text[4:])) * 1000)
    elapsed = np.array([float(value - absolute[0]) for value in absolute])
    assert np.all(np.diff(elapsed) > 0), "QA acquisition times must increase within one day"
    return elapsed


def check_plane_timing(ds, truth, t, source_z, modality):
    assert str(ds.Modality) == modality
    if modality == "PT":
        for field, grid_key, legacy_key in (
            ("FrameReferenceTime", "frame_reference_times_ms", "times_ms"),
            ("ActualFrameDuration", "frame_durations_ms", "durations_ms"),
        ):
            expected = truth[grid_key][t, source_z] if grid_key in truth else truth[legacy_key][t]
            assert np.all(float(getattr(ds, field)) == expected), (field, t, source_z, expected)
    elif modality == "MR" and "acquisition_times" in truth:
        assert str(ds.AcquisitionTime) == str(truth["acquisition_times"][t])
        assert int(ds.TemporalPositionIdentifier) == t + 1
        intervals = np.diff(acquisition_elapsed_ms(truth))
        if intervals.size > 1 and not np.allclose(intervals, intervals[0], atol=1e-5, rtol=0):
            assert "TemporalResolution" not in ds, "Irregular acquisition has no uniform interval"


def check_multivolume_timing(attributes, truth, modality):
    field = "AcquisitionTime" if modality == "MR" else "FrameReferenceTime"
    assert attributes["MultiVolume.FrameIdentifyingDICOMTagName"] == field
    assert attributes["MultiVolume.FrameIdentifyingDICOMTagUnits"] == "ms"
    labels = np.array([float(value) for value in attributes["MultiVolume.FrameLabels"].split(",")])
    expected = acquisition_elapsed_ms(truth) if modality == "MR" else truth["times_ms"]
    np.testing.assert_allclose(labels, expected, atol=1e-5 if modality == "MR" else 0, rtol=0)


def check_sequence_timing(index_values, index_unit, attributes, truth, modality):
    if modality == "MR":
        # The stock importer stores ordinal sequence indices and its physical
        # acquisition-time axis in MultiVolume.FrameLabels, in milliseconds.
        assert index_unit == ""
        assert list(index_values) == [str(t) for t in range(len(truth["acquisition_times"]))]
        check_multivolume_timing(attributes, truth, modality)
    else:
        assert index_unit == "ms"
        np.testing.assert_array_equal([float(value) for value in index_values], truth["times_ms"])
