"""Copied optional codes must not invalidate PET or destroy acquisition facts."""

import json
from copy import deepcopy

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.uid import PositronEmissionTomographyImageStorage, generate_uid

from nifti2dicom import convert


def _code(**fields):
    item = Dataset()
    for keyword, value in fields.items():
        setattr(item, keyword, value)
    return item


@pytest.fixture
def pet_case(tmp_path, sample_dicom_dir):
    originals = [pydicom.dcmread(path) for path in sorted(sample_dicom_dir.glob("*.dcm"))]
    frame_uid = generate_uid()
    for t in range(2):
        for z, original in enumerate(originals):
            ds = deepcopy(original)
            ds.Modality = "PT"
            ds.FrameOfReferenceUID = frame_uid
            ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = (
                PositronEmissionTomographyImageStorage
            )
            ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
            ds.SeriesType = ["DYNAMIC", "IMAGE"]
            ds.SeriesDate, ds.SeriesTime = "20260913", "120000"
            ds.Units, ds.CountsSource = "BQML", "EMISSION"
            ds.DecayCorrection, ds.CorrectedImage = "START", ["DECY"]
            ds.FrameReferenceTime = t * 2500 + z * 37
            ds.ActualFrameDuration = 2000 + t * 1000
            ds.DecayFactor = 1.05 + t * 0.1 + z * 0.01
            ds.NumberOfSlices, ds.NumberOfTimeSlices = 3, 2
            ds.ImageIndex = ds.InstanceNumber = t * 3 + z + 1
            isotope = Dataset()
            isotope.RadionuclideHalfLife = 6586.2
            isotope.RadionuclideTotalDose = 100000000 + t * 17 + z
            isotope.Radiopharmaceutical = "Synthetic test tracer"
            isotope.RadiopharmaceuticalStartTime = "115000"
            isotope.RadionuclideCodeSequence = [
                _code(CodeValue="C-111A1", CodingSchemeDesignator="SRT", CodeMeaning="18F")
            ]
            isotope.AdministrationRouteCodeSequence = [
                _code(CodeValue="IV", CodingSchemeDesignator="99TEST", CodeMeaning="Test route")
            ]
            ds.RadiopharmaceuticalInformationSequence = [isotope]
            pydicom.dcmwrite(
                sample_dicom_dir / f"slice_{ds.InstanceNumber:04d}.dcm",
                ds,
                enforce_file_format=True,
            )
    data = np.arange(96, dtype=np.int16).reshape(4, 4, 3, 2)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[2, 3] = 1
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm", "unknown")
    path = tmp_path / "pet.nii.gz"
    nib.save(image, path)
    return path, sample_dicom_dir, data


@pytest.mark.parametrize(
    ("fields", "retained"),
    [
        ({}, False),  # RIDER source: empty coded item, not an empty sequence.
        ({"CodingSchemeDesignator": "99SDM", "CodeMeaning": "FLT"}, False),
        ({"CodeValue": "T1", "CodeMeaning": "Tracer"}, False),
        ({"CodeValue": "T1", "CodingSchemeDesignator": "99TEST"}, False),
        ({"CodeValue": "T1", "CodingSchemeDesignator": " ", "CodeMeaning": "Tracer"}, False),
        ({"CodeValue": " ", "CodingSchemeDesignator": "99TEST", "CodeMeaning": "Tracer"}, False),
        ({"CodeValue": "T1", "CodingSchemeDesignator": "99TEST", "CodeMeaning": " "}, False),
        (
            {"CodeValue": "T1", "URNCodeValue": "urn:example:t1", "CodeMeaning": "Tracer"},
            False,
        ),
        ({"CodeValue": "T1", "CodingSchemeDesignator": "99TEST", "CodeMeaning": "Tracer"}, True),
        (
            {
                "LongCodeValue": "synthetic-long-tracer-identifier",
                "CodingSchemeDesignator": "99TEST",
                "CodeMeaning": "Tracer",
            },
            True,
        ),
        ({"URNCodeValue": "urn:example:tracer", "CodeMeaning": "Tracer"}, True),
    ],
)
@pytest.mark.parametrize("only_last_frame", [False, True], ids=["all_frames", "late_frame"])
def test_pet_optional_code_cleanup_preserves_isotope_facts_and_reports_warnings(
    tmp_path, pet_case, fields, retained, only_last_frame, capsys
):
    path, reference, values = pet_case
    source_paths = sorted(reference.glob("*.dcm"))
    for source_path in source_paths[-1:] if only_last_frame else source_paths:
        ds = pydicom.dcmread(source_path)
        ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalCodeSequence = [
            _code(**fields)
        ]
        pydicom.dcmwrite(source_path, ds, enforce_file_format=True)
    before = {source_path: source_path.read_bytes() for source_path in source_paths}
    sources = {int(ds.ImageIndex): ds for ds in map(pydicom.dcmread, source_paths)}
    result = convert(path, reference, tmp_path / "output", kind="image")
    assert len(result.files) == 6
    for output in map(pydicom.dcmread, result.files):
        original = sources[int(output.ImageIndex)]
        expected = deepcopy(original.RadiopharmaceuticalInformationSequence)
        if not retained and "RadiopharmaceuticalCodeSequence" in expected[0]:
            del expected[0].RadiopharmaceuticalCodeSequence
        assert output.RadiopharmaceuticalInformationSequence == expected
        for keyword in ("FrameReferenceTime", "ActualFrameDuration", "DecayFactor", "Units"):
            assert getattr(output, keyword) == getattr(original, keyword)
        t, z = divmod(int(output.ImageIndex) - 1, 3)
        decoded = output.pixel_array * float(output.RescaleSlope) + float(output.RescaleIntercept)
        np.testing.assert_array_equal(decoded, values[::-1, ::-1, z, t].T)
    warnings = [
        message for message in result.warnings if "RadiopharmaceuticalCodeSequence" in message
    ]
    assert len(warnings) == (0 if retained else 1)
    manifest = json.loads((result.output / "conversion.json").read_text())
    assert manifest["warnings"] == list(result.warnings)
    assert all(source_path.read_bytes() == contents for source_path, contents in before.items())
    captured = capsys.readouterr()
    assert not captured.out and not captured.err
