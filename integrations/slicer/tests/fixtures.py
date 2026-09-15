"""Small authored DICOM fixtures; no downloads or converter internals."""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def pet_series(directory: Path, *, varying_acquisition=False, slice_times=False, static=False):
    directory.mkdir(parents=True)
    study, series, reference = (generate_uid() for _ in range(3))
    base = np.arange(1, 21, dtype=np.uint16).reshape(4, 5)
    # Hand-defined source mapping: an empty frame, faint fractions, then values
    # above 65535. A uint16 receiver cannot retain the last two frames.
    stored = [
        np.zeros((2, 4, 5), dtype=np.uint16),
        np.stack([base, base + 30]),
        np.stack([base + 20, base + 30]),
    ]
    slopes = [(1.0, 1.0), (0.125, 3.5), (10.25, 10000.5)]
    times, durations = [0.0, 2000.0, 17000.0], [2000, 15000, 30000]
    if static:
        stored, slopes, times, durations = stored[:1], slopes[:1], times[:1], durations[:1]
    expected = []
    paths = []
    for t, planes in enumerate(stored):
        expected.append(
            np.stack([plane.astype(float) * slopes[t][z] for z, plane in enumerate(planes)])
        )
        for z, plane in enumerate(planes):
            ds = Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
            ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
            ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
            ds.PatientName, ds.PatientID = "Synthetic^PET", "N2D-QA"
            ds.PatientBirthDate, ds.PatientSex = "", ""
            ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = (
                study,
                series,
                reference,
            )
            ds.StudyDate, ds.SeriesDate, ds.AcquisitionDate, ds.ContentDate = ("20260914",) * 4
            ds.StudyTime, ds.SeriesTime, ds.ContentTime = ("120000",) * 3
            ds.AcquisitionTime = (
                ["120000", "120002", "120017"][t] if varying_acquisition else "120000"
            )
            ds.StudyID, ds.SeriesNumber = "QA", 7
            ds.StudyDescription, ds.SeriesDescription = (
                "Synthetic geometry QA",
                "Dynamic PET regression",
            )
            ds.AccessionNumber, ds.ReferringPhysicianName = "", ""
            ds.Modality, ds.PatientPosition = "PT", "HFS"
            ds.SeriesType = ["STATIC" if static else "DYNAMIC", "IMAGE"]
            ds.ImageType = ["DERIVED", "PRIMARY"]
            ds.NumberOfSlices, ds.NumberOfTimeSlices = 2, len(stored)
            ds.ImageIndex = t * 2 + z + 1
            ds.InstanceNumber = 100 - ds.ImageIndex
            ds.FrameReferenceTime = str(times[t] + (z * 0.25 if slice_times else 0.0))
            ds.ActualFrameDuration = durations[t]
            ds.Units, ds.DecayCorrection, ds.CountsSource = "BQML", "NONE", "EMISSION"
            ds.CorrectedImage = ["ATTN"]
            ds.Rows, ds.Columns = plane.shape
            ds.PixelSpacing, ds.SliceThickness = [1.5, 2.5], 4.0
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.ImagePositionPatient = [-12.0, 23.0, -9.0 + z * 4.0]
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
            ds.RescaleSlope, ds.RescaleIntercept = str(slopes[t][z]), "0"
            ds.PixelData = plane.tobytes()
            path = directory / f"part-{20 - ds.ImageIndex:02d}.dcm"
            pydicom.dcmwrite(path, ds, enforce_file_format=True)
            paths.append(path)
    return paths, expected


def change_header(path, update):
    ds = pydicom.dcmread(path)
    update(ds)
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
