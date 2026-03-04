"""Shared test fixtures for nifti2dicom tests."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filewriter import dcmwrite
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a clean temporary directory."""
    return tmp_path


@pytest.fixture
def sample_nifti_3d(tmp_path: Path) -> Path:
    """Create a minimal 3-D NIfTI file (4x4x3 voxels)."""
    data = np.random.randint(0, 1000, (4, 4, 3), dtype=np.int16).astype(np.float64)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "test_3d.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture
def sample_nifti_4d(tmp_path: Path) -> Path:
    """Create a minimal 4-D NIfTI file (4x4x3x2 — 2 timepoints)."""
    data = np.random.randint(0, 1000, (4, 4, 3, 2), dtype=np.int16).astype(np.float64)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "test_4d.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture
def sample_rgb_nifti(tmp_path: Path) -> Path:
    """Create a minimal RGB NIfTI file (4x4x3x3 — 3 channels)."""
    data = np.random.randint(0, 255, (4, 4, 3, 3), dtype=np.uint8)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "test_rgb.nii.gz"
    nib.save(img, str(path))
    return path


def _make_dicom_slice(
    instance_number: int,
    rows: int = 4,
    cols: int = 4,
    modality: str = "CT",
) -> Dataset:
    """Create a minimal valid DICOM dataset."""
    ds = Dataset()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds.file_meta = file_meta

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.InstanceNumber = instance_number
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Modality = modality
    ds.RescaleSlope = "1.0"
    ds.RescaleIntercept = "0.0"
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = "1.0"
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.SeriesDescription = "test"
    ds.SeriesNumber = 1
    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"

    pixel_data = np.zeros((rows, cols), dtype=np.int16)
    ds.PixelData = pixel_data.tobytes()

    return ds


@pytest.fixture
def sample_dicom_dir(tmp_path: Path) -> Path:
    """Create a directory with 3 minimal DICOM files."""
    dicom_dir = tmp_path / "dicom_ref"
    dicom_dir.mkdir()

    # Shared UIDs for the series
    study_uid = generate_uid()
    series_uid = generate_uid()

    for i in range(1, 4):  # 3 slices to match our 3-slice NIfTI
        ds = _make_dicom_slice(i, rows=4, cols=4)
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        dcmwrite(str(dicom_dir / f"slice_{i:04d}.dcm"), ds, write_like_original=False)

    return dicom_dir
