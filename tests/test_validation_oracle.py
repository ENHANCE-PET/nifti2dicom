"""Exercise the public-data oracle without networking or converter helpers."""

from copy import deepcopy

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, PositronEmissionTomographyImageStorage, generate_uid

from validation.idc_roundtrip import audit, source_volume


def public_like_frames(tmp_path, *, timepoints=3):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    series, frame = generate_uid(), generate_uid()
    paths, outputs = [], []
    for t in range(timepoints):
        for z in range(3):
            ds = Dataset()
            ds.SOPClassUID = PositronEmissionTomographyImageStorage
            ds.SOPInstanceUID = generate_uid()
            ds.SeriesInstanceUID, ds.FrameOfReferenceUID = series, frame
            ds.Modality = "PT"
            ds.SeriesType = ["DYNAMIC" if timepoints > 1 else "STATIC", "IMAGE"]
            ds.FrameReferenceTime = (100, 400, 1200)[t] if timepoints > 1 else 100 + z * 30
            ds.ActualFrameDuration = 50 + z
            ds.Units, ds.DecayCorrection = "BQML", "NONE"
            ds.Rows, ds.Columns = 2, 4
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [10, -20, 30 + z * 5]
            ds.PixelSpacing = [3, 2]
            ds.ImageIndex = t * 3 + 3 - z  # Coherent nonconformant source reversal.
            ds.InstanceNumber = timepoints * 3 - (t * 3 + z)
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 0
            ds.RescaleSlope, ds.RescaleIntercept = 1, 0
            ds.PixelData = (np.arange(8, dtype="<u2") + t * 100 + z * 10).tobytes()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            path = source / f"{ds.InstanceNumber}.dcm"
            pydicom.dcmwrite(path, ds, enforce_file_format=True)
            paths.append(path)
            converted = deepcopy(ds)
            converted.ImageIndex = t * 3 + z + 1
            converted.SOPInstanceUID = generate_uid()
            converted.file_meta.MediaStorageSOPInstanceUID = converted.SOPInstanceUID
            destination = output / path.name
            pydicom.dcmwrite(destination, converted, enforce_file_format=True)
            outputs.append(destination)
    selection = {"SeriesInstanceUID": series, "NumberOfTimeSlices": timepoints}
    truth = source_volume(sorted(paths), selection)
    return outputs, truth


@pytest.mark.parametrize("timepoints", [1, 3])
def test_public_oracle_uses_physical_positions_and_times_not_native_indices(tmp_path, timepoints):
    paths, truth = public_like_frames(tmp_path, timepoints=timepoints)
    report = audit(list(reversed(paths)), *truth)
    assert report["coverage_complete"] and report["acquisition_metadata_preserved"]
    assert report["maximum_pixel_error"] == report["maximum_world_error_mm"] == 0
    assert report["timepoints"] == timepoints


@pytest.mark.parametrize("corruption", ["x", "y", "z", "time", "missing", "duplicate", "timing"])
def test_public_oracle_rejects_corrupted_dicom_objects(tmp_path, corruption):
    paths, truth = public_like_frames(tmp_path)
    headers = {path: pydicom.dcmread(path) for path in paths}
    target = next(
        p
        for p, ds in headers.items()
        if float(ds.FrameReferenceTime) == 100 and ds.ImagePositionPatient[2] == 30
    )
    ds = headers[target]
    if corruption in {"x", "y"}:
        ds.PixelData = np.flip(ds.pixel_array, axis=1 if corruption == "x" else 0).tobytes()
    elif corruption in {"z", "time"}:
        other = next(
            p
            for p, h in headers.items()
            if float(h.FrameReferenceTime) == (1200 if corruption == "time" else 100)
            and h.ImagePositionPatient[2] == (30 if corruption == "time" else 40)
        )
        ds.PixelData, headers[other].PixelData = headers[other].PixelData, ds.PixelData
        pydicom.dcmwrite(other, headers[other], enforce_file_format=True)
    elif corruption == "timing":
        ds.ActualFrameDuration = 999
    elif corruption == "missing":
        paths.remove(target)
    else:
        paths.append(target)
    pydicom.dcmwrite(target, ds, enforce_file_format=True)
    with pytest.raises(AssertionError):
        audit(paths, *truth)
