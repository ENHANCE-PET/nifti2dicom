"""SEG round trips must preserve labels and patient-space correspondence."""

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.uid import generate_uid

from nifti2dicom.errors import ConversionError, LabelError
from tests.conftest import _make_dicom_slice


@pytest.fixture
def seg_reference(tmp_path):
    directory = tmp_path / "reference"
    directory.mkdir()
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    for index in range(3):
        ds = _make_dicom_slice(index + 1)
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.ImagePositionPatient = [0, 0, 2 - index]
        for key in (
            "PatientBirthDate",
            "PatientSex",
            "AccessionNumber",
            "StudyID",
            "StudyDate",
            "StudyTime",
        ):
            setattr(ds, key, "")
        pydicom.dcmwrite(directory / f"{index}.dcm", ds, enforce_file_format=True)
    return directory


def save_mask(tmp_path, values):
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(values, np.diag([-1.0, -1.0, 1.0, 1.0])), path)
    return path


def convert_mask(path, reference, output, **kwargs):
    import nifti2dicom

    assert hasattr(nifti2dicom, "convert"), "SEG must use the public pipeline"
    return nifti2dicom.convert(path, reference, output, kind="seg", **kwargs)


def test_sparse_labels_and_source_order_roundtrip(tmp_path, seg_reference):
    import highdicom as hd

    values = np.zeros((4, 4, 3), dtype=np.uint16)
    values[0, 1, 0] = 300
    values[2, 3, 2] = 5
    path = save_mask(tmp_path, values)
    result = convert_mask(
        path,
        seg_reference,
        tmp_path / "out",
        labels={"300": "Liver", "5": "Lesion"},
        algorithm_type="manual",
    )
    assert result.label_mapping == {5: 1, 300: 2}
    seg = hd.seg.segread(result.files[0])
    source = sorted(
        (pydicom.dcmread(f) for f in seg_reference.iterdir()),
        key=lambda ds: float(ds.ImagePositionPatient[2]),
    )
    recovered = seg.get_pixels_by_source_instance(
        [ds.SOPInstanceUID for ds in source],
        combine_segments=True,
        assert_missing_frames_are_empty=True,
    )
    assert recovered[0, 1, 0] == 2
    assert recovered[2, 3, 2] == 1
    assert np.count_nonzero(recovered) == 2
    assert str(seg.SegmentationType) == "BINARY"


def test_automatic_provenance_and_snomed_survive(tmp_path, seg_reference):
    values = np.zeros((4, 4, 3), dtype=np.uint16)
    values[1, 1, 1] = 1
    result = convert_mask(
        save_mask(tmp_path, values),
        seg_reference,
        tmp_path / "out",
        labels={
            "organ_indices": {"1": {"name": "Liver", "SNOMED": {"ID": "10200004", "name": "Liver"}}}
        },
        algorithm_type="automatic",
        algorithm_name="MOOSE",
        algorithm_version="1.0",
    )
    ds = pydicom.dcmread(result.files[0])
    segment = ds.SegmentSequence[0]
    assert segment.SegmentAlgorithmType == "AUTOMATIC"
    assert segment.SegmentAlgorithmName == "MOOSE"
    assert segment.SegmentedPropertyTypeCodeSequence[0].CodeValue == "10200004"


@pytest.mark.parametrize("value", [0, -1, 0.5, float("nan")])
def test_invalid_masks_fail_without_output(tmp_path, seg_reference, value):
    path = save_mask(tmp_path, np.full((4, 4, 3), value, dtype=np.float32))
    output = tmp_path / "out"
    with pytest.raises(ConversionError):
        convert_mask(path, seg_reference, output, algorithm_type="manual")
    assert not output.exists()


def test_provenance_is_not_guessed(tmp_path, seg_reference):
    path = save_mask(tmp_path, np.ones((4, 4, 3), dtype=np.uint8))
    with pytest.raises(LabelError, match="[Aa]lgorithm|created"):
        convert_mask(path, seg_reference, tmp_path / "out")


def test_inspect_reports_missing_provenance(tmp_path, seg_reference):
    from nifti2dicom import inspect

    path = save_mask(tmp_path, np.ones((4, 4, 3), dtype=np.uint8))
    result = inspect(path, seg_reference, kind="seg")
    assert any("algorithm" in warning.lower() for warning in result.warnings)


def test_null_codes_are_rejected(tmp_path, seg_reference):
    path = save_mask(tmp_path, np.ones((4, 4, 3), dtype=np.uint8))
    labels = {"1": {"name": "Organ", "type": {"value": None, "scheme": None, "meaning": None}}}
    with pytest.raises(LabelError, match="code|Code|coded"):
        convert_mask(path, seg_reference, tmp_path / "out", labels=labels, algorithm_type="manual")


def test_foreground_voxel_edges_must_fit_reference(tmp_path, seg_reference):
    path = tmp_path / "coarse_mask.nii.gz"
    affine = np.diag([-8.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [-1.5, -1.5, 1.0]
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), affine), path)
    with pytest.raises(ConversionError, match="outside"):
        convert_mask(path, seg_reference, tmp_path / "out", algorithm_type="manual")


@pytest.mark.parametrize("reference_first", [True, False])
def test_both_historical_seg_argument_orders(tmp_path, seg_reference, reference_first):
    from nifti2dicom.converter import save_dicom_from_nifti_seg

    path = save_mask(tmp_path, np.ones((4, 4, 3), dtype=np.uint8))
    args = (seg_reference, path) if reference_first else (path, seg_reference)
    result = save_dicom_from_nifti_seg(
        *args, tmp_path / "out", {"1": "Region"}, algorithm_type="manual"
    )
    assert len(result.files) == 1


def test_unicode_seg_metadata_survives_serialization(tmp_path, seg_reference):
    for file in seg_reference.glob("*.dcm"):
        source = pydicom.dcmread(file)
        source.SpecificCharacterSet = "ISO_IR 192"
        source.PatientName = "山田^太郎"
        pydicom.dcmwrite(file, source, enforce_file_format=True)
    result = convert_mask(
        save_mask(tmp_path, np.ones((4, 4, 3), dtype=np.uint8)),
        seg_reference,
        tmp_path / "out",
        labels={"1": "肝臓"},
        description="測定",
        algorithm_type="manual",
    )
    ds = pydicom.dcmread(result.files[0])
    assert ds.SpecificCharacterSet == "ISO_IR 192"
    assert ds.SegmentSequence[0].SegmentLabel == "肝臓"
    assert ds.SeriesDescription == "測定"
    assert str(ds.PatientName) == "山田^太郎"


@pytest.mark.parametrize(
    "code, retained",
    [
        ({"CodeValue": "123", "CodeMeaning": "Procedure"}, False),
        ({"CodeValue": "123", "CodingSchemeDesignator": "99TEST"}, False),
        ({"CodeValue": "123", "CodeMeaning": "Procedure", "CodingSchemeDesignator": ""}, False),
        (
            {"CodeValue": "123", "CodeMeaning": "Procedure", "CodingSchemeDesignator": "99TEST"},
            True,
        ),
        ({"LongCodeValue": "long-procedure-code", "CodeMeaning": "Procedure"}, False),
        ({"URNCodeValue": "urn:example:procedure", "CodeMeaning": "Procedure"}, True),
    ],
)
def test_incomplete_optional_source_codes_are_omitted_not_invented(
    tmp_path, seg_reference, code, retained
):
    for path in seg_reference.glob("*.dcm"):
        source = pydicom.dcmread(path)
        item = pydicom.Dataset()
        for keyword, value in code.items():
            setattr(item, keyword, value)
        source.ProcedureCodeSequence = [item]
        pydicom.dcmwrite(path, source, enforce_file_format=True)
    before = {path: path.read_bytes() for path in seg_reference.glob("*.dcm")}
    result = convert_mask(
        save_mask(tmp_path, np.ones((4, 4, 3), dtype=np.uint8)),
        seg_reference,
        tmp_path / "out",
        algorithm_type="manual",
    )
    output = pydicom.dcmread(result.files[0])
    assert ("ProcedureCodeSequence" in output) == retained
    assert any("ProcedureCodeSequence" in warning for warning in result.warnings) != retained
    if retained:
        assert output.ProcedureCodeSequence[0] == item
    assert all(path.read_bytes() == original for path, original in before.items())
