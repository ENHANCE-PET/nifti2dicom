"""End-to-end guards against spatial mirrors and reassigned timepoints.

The oracle uses a hand-built asymmetric phantom and DICOM world coordinates,
never the package's readers, geometry helpers, or orientation transforms.
"""

from copy import deepcopy
from itertools import permutations, product

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    PositronEmissionTomographyImageStorage,
    generate_uid,
)

from nifti2dicom import convert
from nifti2dicom.errors import PixelEncodingError

_SHAPE = (3, 4, 2)
_PET_TIMES = (0, 7500, 25000)
_DURATIONS = (7500, 17500, 35000)
_ORIENTATIONS = list(product(permutations(range(3)), product((-1, 1), repeat=3)))


def _make_case(tmp_path, modality, timepoints):
    x, y, z, t = np.indices((*_SHAPE, timepoints))
    pixels = (100 * x + 10 * y + z + 1000 * t).astype(np.int16)
    a, b = np.deg2rad([17, 11])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    directions = rz @ rx
    affine = np.eye(4)
    affine[:3, :3] = directions * [2, 3, 5]
    affine[:3, 3] = [31, -47, 19]
    directory = tmp_path / "reference"
    directory.mkdir()
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    for t in range(timepoints):
        for z in range(_SHAPE[2]):
            ds = Dataset()
            ds.SOPClassUID = {
                "CT": CTImageStorage,
                "MR": MRImageStorage,
                "PT": PositronEmissionTomographyImageStorage,
            }[modality]
            ds.SOPInstanceUID = generate_uid()
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
            ds.FrameOfReferenceUID = frame
            ds.PatientID, ds.PatientName = "PHANTOM", "Orientation^Phantom"
            ds.Modality = modality
            ds.Rows, ds.Columns = _SHAPE[1], _SHAPE[0]
            ds.ImageOrientationPatient = list(directions[:, 0]) + list(directions[:, 1])
            ds.ImagePositionPatient = list(affine[:3, 3] + z * affine[:3, 2])
            ds.PixelSpacing, ds.SliceThickness = [3, 2], 5
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 0
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.RescaleSlope, ds.RescaleIntercept = "1", "0"
            ds.PixelData = pixels[:, :, z, t].T.astype("<u2").tobytes()
            index = t * _SHAPE[2] + z
            # Filenames and InstanceNumber intentionally disagree with physical/time order.
            ds.InstanceNumber = timepoints * _SHAPE[2] - index
            if modality == "PT":
                ds.SeriesType = ["DYNAMIC" if timepoints > 1 else "STATIC", "IMAGE"]
                ds.SeriesDate, ds.SeriesTime = "20260913", "120000"
                ds.Units, ds.CountsSource = "BQML", "EMISSION"
                ds.DecayCorrection, ds.CorrectedImage = "START", ["DECY"]
                ds.NumberOfSlices, ds.ImageIndex = _SHAPE[2], index + 1
                ds.FrameReferenceTime = _PET_TIMES[t] + 37 * z
                ds.ActualFrameDuration = _DURATIONS[t]
                ds.DecayFactor = 1.125 + 0.25 * t + 0.001 * z
                if timepoints > 1:
                    ds.NumberOfTimeSlices = timepoints
            elif timepoints > 1:
                ds.TemporalPositionIdentifier = t + 1
                ds.NumberOfTemporalPositions = timepoints
                ds.TemporalResolution = 2500
            if modality == "MR":
                ds.ScanningSequence, ds.SequenceVariant = "GR", "SP"
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            pydicom.dcmwrite(
                directory / f"{ds.InstanceNumber:03d}.dcm", ds, enforce_file_format=True
            )
    return pixels, affine, directory


def _convert_orientation(tmp_path, modality, timepoints, permutation, signs, *, fractional=False):
    pixels, affine, reference = _make_case(tmp_path, modality, timepoints)
    if fractional:
        _, _, z, t = np.indices(pixels.shape)
        pixels = (pixels.astype(float) + 1) * (0.000013 + z * 17.131) * (t + 1) ** 5
    stored = pixels.transpose((*permutation, 3))
    stored_affine = affine.copy()
    for axis, original_axis in enumerate(permutation):
        stored_affine[:3, axis] = affine[:3, original_axis] * signs[axis]
        if signs[axis] < 0:
            stored = np.flip(stored, axis)
            stored_affine[:3, 3] += affine[:3, original_axis] * (_SHAPE[original_axis] - 1)
    ras_affine = np.diag([-1, -1, 1, 1]) @ stored_affine
    nifti = nib.Nifti1Image(stored if timepoints > 1 else stored[..., 0], ras_affine)
    nifti.header.set_xyzt_units("mm", "unknown" if modality == "PT" else "sec")
    if timepoints > 1 and modality != "PT":
        nifti.header["pixdim"][4] = 2.5
    path = tmp_path / "input.nii.gz"
    nib.save(nifti, path)
    result = convert(path, reference, tmp_path / "output", kind="image")
    return [pydicom.dcmread(path) for path in result.files], pixels, affine


def _assert_physical_identity(datasets, expected, affine, modality, *, fractional=False):
    occupied = np.zeros(expected.shape, dtype=bool)
    inverse = np.linalg.inv(affine)
    for ds in datasets:
        if modality == "PT":
            t = (int(ds.ImageIndex) - 1) // _SHAPE[2]
        else:
            t = int(getattr(ds, "TemporalPositionIdentifier", 1)) - 1
        yy, xx = np.indices((ds.Rows, ds.Columns))
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        world = (
            np.asarray(ds.ImagePositionPatient, dtype=float)[:, None]
            + iop[:3, None] * float(ds.PixelSpacing[1]) * xx.ravel()
            + iop[3:, None] * float(ds.PixelSpacing[0]) * yy.ravel()
        )
        coords = inverse[:3, :3] @ world + inverse[:3, 3, None]
        rounded = np.rint(coords).astype(int)
        np.testing.assert_allclose(coords, rounded, atol=2e-5, rtol=0)
        x, y, z = rounded
        assert 0 <= t < expected.shape[3]
        for indices, size in zip(rounded, _SHAPE, strict=True):
            assert np.all((indices >= 0) & (indices < size))
        assert np.unique(z).size == 1
        assert not occupied[x, y, z, t].any()
        occupied[x, y, z, t] = True
        decoded = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        real = expected[x, y, z, t]
        if fractional:
            # Positive PET values: independent per-plane 16-bit precision bound.
            bound = float(real.max()) / (2 * 65535) + float(real.max()) * 1e-12
            np.testing.assert_allclose(decoded.ravel(), real, rtol=0, atol=bound)
            assert float(ds.RescaleIntercept) == 0
        else:
            np.testing.assert_array_equal(decoded.ravel(), real)
        if modality == "PT":
            assert float(ds.FrameReferenceTime) == _PET_TIMES[t] + 37 * int(z[0])
            assert int(ds.ActualFrameDuration) == _DURATIONS[t]
            assert float(ds.DecayFactor) == 1.125 + 0.25 * t + 0.001 * int(z[0])
        elif expected.shape[3] > 1:
            assert float(ds.TemporalResolution) == 2500
    assert occupied.all()


@pytest.mark.parametrize("modality", ["CT", "MR", "PT"])
@pytest.mark.parametrize("timepoints", [1, 3], ids=["3d", "4d"])
@pytest.mark.parametrize(("permutation", "signs"), _ORIENTATIONS)
def test_every_storage_orientation_preserves_world_voxels_and_time(
    tmp_path, modality, timepoints, permutation, signs
):
    datasets, expected, affine = _convert_orientation(
        tmp_path, modality, timepoints, permutation, signs
    )
    _assert_physical_identity(datasets, expected, affine, modality)


@pytest.mark.parametrize("timepoints", [1, 3], ids=["3d", "4d"])
@pytest.mark.parametrize(("permutation", "signs"), _ORIENTATIONS)
def test_pet_local_precision_follows_physical_plane_through_every_orientation(
    tmp_path, timepoints, permutation, signs
):
    datasets, expected, affine = _convert_orientation(
        tmp_path, "PT", timepoints, permutation, signs, fractional=True
    )
    _assert_physical_identity(datasets, expected, affine, "PT", fractional=True)


@pytest.mark.parametrize("overwrite", [False, True])
def test_late_pet_encoding_failure_never_publishes_partial_series(tmp_path, overwrite):
    pixels, affine, reference = _make_case(tmp_path, "PT", 3)
    data = np.full(pixels.shape, 0.125)
    data[..., -1] = np.nextafter(0.0, 1.0)
    nifti = nib.Nifti1Image(data, np.diag([-1, -1, 1, 1]) @ affine)
    nifti.header.set_xyzt_units("mm", "unknown")
    path = tmp_path / "input.nii.gz"
    nib.save(nifti, path)
    output = tmp_path / "output"
    if overwrite:
        output.mkdir()
        (output / "keep.txt").write_text("existing output")
    with pytest.raises(PixelEncodingError):
        convert(path, reference, output, kind="image", overwrite=overwrite)
    if overwrite:
        assert [p.name for p in output.iterdir()] == ["keep.txt"]
        assert (output / "keep.txt").read_text() == "existing output"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))


@pytest.mark.parametrize("corruption", ["x", "y", "z", "time_pixels", "time_tags"])
def test_physical_audit_detects_spatial_and_temporal_corruption(tmp_path, corruption):
    original, expected, affine = _convert_orientation(tmp_path, "PT", 3, (2, 0, 1), (-1, 1, -1))
    # Mutation selection, like the audit itself, must not depend on result.files order.
    original = original[::2] + original[1::2]
    _assert_physical_identity(original, expected, affine, "PT")
    damaged = deepcopy(original)
    if corruption in {"x", "y"}:
        for ds in damaged:
            ds.PixelData = np.flip(ds.pixel_array, 1 if corruption == "x" else 0).tobytes()
    else:
        inverse = np.linalg.inv(affine)

        def frame_key(ds):
            position = inverse @ [*ds.ImagePositionPatient, 1]
            return (int(ds.ImageIndex) - 1) // _SHAPE[2], int(np.rint(position[2]))

        by_frame = {frame_key(ds): ds for ds in original}
        for ds in damaged:
            t, z = frame_key(ds)
            source = by_frame[t, _SHAPE[2] - 1 - z] if corruption == "z" else by_frame[2 - t, z]
            if corruption == "time_tags":
                ds.FrameReferenceTime = source.FrameReferenceTime
            else:
                ds.PixelData = source.PixelData
    with pytest.raises(AssertionError):
        _assert_physical_identity(damaged, expected, affine, "PT")
