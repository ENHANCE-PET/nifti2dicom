"""RGB NIfTI → DICOM conversion.

Creates Secondary Capture DICOM images with 8-bit RGB photometric
interpretation.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filewriter import dcmwrite
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

from nifti2dicom import cli_theme as theme
from nifti2dicom.dicom_io import load_dicom_series


def _make_rgb_dataset(
    slice_array: np.ndarray,
    reference: Dataset,
    series_uid: str,
    study_uid: str,
    instance_number: int,
    ipp: list[float] | None = None,
) -> Dataset:
    """Build a Secondary Capture DICOM dataset from an RGB slice."""
    ds = Dataset()

    # File meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.7")
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # UIDs
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = "1"
    ds.InstanceNumber = str(instance_number)

    # Patient info from reference
    for attr in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge"):
        val = getattr(reference, attr, None)
        if val is not None:
            setattr(ds, attr, val)

    # Spatial info
    if ipp is not None:
        ds.ImagePositionPatient = [str(v) for v in ipp]
    if hasattr(reference, "ImageOrientationPatient"):
        ds.ImageOrientationPatient = reference.ImageOrientationPatient
    if hasattr(reference, "SliceThickness"):
        ds.SliceThickness = reference.SliceThickness
    if hasattr(reference, "PixelSpacing"):
        ds.PixelSpacing = reference.PixelSpacing

    # Image attributes
    ds.Modality = "SC"
    ds.PhotometricInterpretation = "RGB"
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0
    rows, cols = slice_array.shape[:2]
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelData = np.ascontiguousarray(slice_array[:, :, :3].astype(np.uint8)).tobytes()
    ds.ImageType = ["DERIVED", "SECONDARY"]

    return ds


def convert_rgb_nifti_to_dicom(
    ref_dir: str | Path,
    nifti_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Convert an RGB NIfTI image to a DICOM series.

    Parameters
    ----------
    ref_dir : path
        Reference DICOM series directory.
    nifti_path : path
        Path to the RGB NIfTI file.
    output_dir : path
        Where to write the output DICOM series.
    """
    ref_dir = Path(ref_dir)
    nifti_path = Path(nifti_path)
    output_dir = Path(output_dir)

    theme.section("LOADING", number="01")
    theme.info(f"RGB NIfTI: {nifti_path}")
    theme.info(f"Reference DICOM: {ref_dir}")

    # Load reference
    dicom_slices, _ = load_dicom_series(ref_dir)
    reference = dicom_slices[0]

    # Load reference image geometry via SimpleITK
    dicom_sitk = sitk.ImageSeriesReader()
    dicom_names = dicom_sitk.GetGDCMSeriesFileNames(str(ref_dir))
    dicom_sitk.SetFileNames(dicom_names)
    ref_image = dicom_sitk.Execute()

    # Load RGB NIfTI and align to reference geometry
    rgb_sitk = sitk.ReadImage(str(nifti_path))
    rgb_sitk.SetOrigin(ref_image.GetOrigin())
    rgb_sitk.SetDirection(ref_image.GetDirection())
    arr = sitk.GetArrayFromImage(rgb_sitk)  # (Z, Y, X, C)
    arr = arr[:, :, :, :3]  # Drop alpha if present

    theme.info(f"RGB shape: {arr.shape}")

    # Fresh UIDs
    series_uid = generate_uid()
    study_uid = getattr(reference, "StudyInstanceUID", generate_uid())

    # Write
    theme.section("WRITING", number="02")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = arr.shape[0]
    with theme.progress(total, "Writing RGB DICOM slices") as tick:
        for i in range(total):
            ipp = list(ref_image.TransformIndexToPhysicalPoint((0, 0, i)))
            ds = _make_rgb_dataset(arr[i], reference, series_uid, study_uid, i + 1, ipp=ipp)
            out_file = os.path.join(str(output_dir), f"slice_{i + 1:04d}.dcm")
            dcmwrite(out_file, ds, write_like_original=False)
            tick()

    theme.ok(f"Wrote {total} RGB slices to {output_dir}")
