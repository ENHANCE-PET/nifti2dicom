"""Exercise the CLI with real NIfTI and DICOM files, including machine output."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from click.testing import CliRunner

from nifti2dicom.cli import cli


@pytest.mark.parametrize(
    "args",
    [
        ["missing.nii.gz", "--reference", "missing-dicom", "--json"],
        ["--json", "convert", "-n", "missing.nii.gz", "-d", "missing-dicom"],
        ["inspect", "missing.nii.gz", "--reference", "missing-dicom", "--json"],
    ],
)
def test_missing_paths_produce_one_json_error(args):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert report["error"]["code"] == "invalid_input"
    assert "missing.nii.gz" in report["error"]["message"]
    assert report["error"]["hint"]
    assert result.stderr == ""


@pytest.mark.parametrize(
    "args",
    [
        ["input.nii.gz", "--json"],
        ["input.nii.gz", "--reference", "dicom", "--kind", "invalid", "--json"],
        ["convert", "--unknown-option", "--json"],
    ],
)
def test_usage_errors_are_machine_readable(args):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert report["error"]["code"] == "usage_error"
    assert report["error"]["message"]
    assert result.stderr == ""


def test_direct_conversion_json_is_clean_and_files_decode(
    tmp_path, sample_nifti_3d, sample_dicom_dir
):
    output = tmp_path / "output"
    result = CliRunner().invoke(
        cli,
        [str(sample_nifti_3d), "--reference", str(sample_dicom_dir), "-o", str(output), "--json"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["status"] == "ok"
    assert report["kind"] == "image"
    assert Path(report["output"]) == output
    assert len(report["files"]) == 3
    assert all(pydicom.dcmread(path).pixel_array.shape == (4, 4) for path in report["files"])
    assert result.stderr == ""


def test_direct_conversion_without_output_uses_safe_default(sample_nifti_3d, sample_dicom_dir):
    result = CliRunner().invoke(
        cli, [str(sample_nifti_3d), "--reference", str(sample_dicom_dir), "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert Path(report["output"]).is_dir()
    assert len(report["files"]) == 3
    assert sample_nifti_3d.is_file()


def test_human_summary_uses_stderr_with_no_ansi_when_redirected(
    tmp_path, sample_nifti_3d, sample_dicom_dir
):
    result = CliRunner().invoke(
        cli,
        [
            str(sample_nifti_3d),
            "--reference",
            str(sample_dicom_dir),
            "-o",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "01" in result.stderr
    assert "3" in result.stderr
    assert "3/3" in result.stderr
    assert result.stderr.count("WRITING") == 1
    assert "ENCODE" not in result.stderr
    assert "output" in result.stderr
    assert "\x1b[" not in result.stderr


def test_quiet_still_converts_without_output(tmp_path, sample_nifti_3d, sample_dicom_dir):
    output = tmp_path / "output"
    result = CliRunner().invoke(
        cli,
        [str(sample_nifti_3d), "--reference", str(sample_dicom_dir), "-o", str(output), "--quiet"],
    )
    assert result.exit_code == 0, result.output
    assert result.output == ""
    assert len(list(output.glob("*.dcm"))) == 3


def test_quiet_errors_remain_visible():
    result = CliRunner().invoke(cli, ["missing.nii.gz", "--reference", "missing-dicom", "--quiet"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "missing.nii.gz" in result.stderr
    assert "Traceback" not in result.stderr


def test_inspect_reports_shape_and_does_not_write(tmp_path, sample_nifti_3d, sample_dicom_dir):
    before = set(tmp_path.rglob("*"))
    result = CliRunner().invoke(
        cli, ["inspect", str(sample_nifti_3d), "--reference", str(sample_dicom_dir), "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["status"] == "ok"
    assert report["kind"] == "image"
    assert report["timepoints"] == 1
    assert report["modality"] == "CT"
    assert report["spacing"] == [1.0, 1.0, 1.0]
    assert set(tmp_path.rglob("*")) == before


def test_legacy_convert_flags_and_force_replace_output(tmp_path, sample_nifti_3d, sample_dicom_dir):
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.txt").write_text("previous output")
    result = CliRunner().invoke(
        cli,
        [
            "convert",
            "-n",
            str(sample_nifti_3d),
            "-d",
            str(sample_dicom_dir),
            "-o",
            str(output),
            "-desc",
            "CLI regression",
            "--force",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert len(report["files"]) == 3
    assert not (output / "old.txt").exists()
    assert pydicom.dcmread(report["files"][0]).SeriesDescription == "CLI regression"


def test_existing_output_error_preserves_files(tmp_path, sample_nifti_3d, sample_dicom_dir):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    result = CliRunner().invoke(
        cli,
        [str(sample_nifti_3d), "--reference", str(sample_dicom_dir), "-o", str(output), "--json"],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "output_error"
    assert marker.read_text() == "keep"


def test_legacy_rgb_creates_color_dicom(tmp_path, sample_rgb_nifti, sample_dicom_dir):
    result = CliRunner().invoke(
        cli,
        [
            "rgb",
            "-n",
            str(sample_rgb_nifti),
            "-d",
            str(sample_dicom_dir),
            "-o",
            str(tmp_path / "rgb-output"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["kind"] == "rgb"
    assert pydicom.dcmread(report["files"][0]).pixel_array.shape == (4, 4, 3)


def test_legacy_resample_uses_selected_reference_geometry(
    tmp_path, sample_nifti_3d, sample_dicom_dir
):
    result = CliRunner().invoke(
        cli,
        [
            "resample",
            "-n",
            str(sample_nifti_3d),
            "-d",
            str(sample_dicom_dir),
            "--spatial-dir",
            str(sample_dicom_dir),
            "-o",
            str(tmp_path / "resampled"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    positions = [list(pydicom.dcmread(path).ImagePositionPatient) for path in report["files"]]
    assert positions == [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]


@pytest.mark.parametrize("debug", [False, True])
def test_unexpected_errors_are_concise_unless_debug_is_requested(monkeypatch, debug):
    from nifti2dicom import api

    def fail(*args, **kwargs):
        raise RuntimeError("writer failed")

    monkeypatch.setattr(api, "convert", fail)
    args = ["input.nii.gz", "--reference", "dicom", "--json"]
    if debug:
        args.append("--debug")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["error"]["code"] == "internal_error"
    assert "writer failed" in report["error"]["message"]
    assert "--debug" in report["error"]["hint"]
    assert ("Traceback" in result.stderr) is debug


def test_legacy_segment_preserves_label_names_and_provenance(tmp_path, sample_dicom_dir):
    frame_uid = pydicom.uid.generate_uid()
    for path in sample_dicom_dir.glob("*.dcm"):
        source = pydicom.dcmread(path)
        source.FrameOfReferenceUID = frame_uid
        pydicom.dcmwrite(path, source, enforce_file_format=True)
    nifti = tmp_path / "mask.nii.gz"
    data = np.zeros((4, 4, 3), dtype=np.uint16)
    data[1:3, 1:3, 1] = 300
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[2, 3] = 1
    nib.save(nib.Nifti1Image(data, affine), nifti)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"300": "Example tissue"}))
    result = CliRunner().invoke(
        cli,
        [
            "segment",
            "-n",
            str(nifti),
            "-d",
            str(sample_dicom_dir),
            "-j",
            str(labels),
            "-o",
            str(tmp_path / "segments"),
            "--algorithm-type",
            "manual",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["kind"] == "seg"
    assert report["label_mapping"] == {"300": 1}
    dataset = pydicom.dcmread(report["files"][0])
    assert dataset.SegmentSequence[0].SegmentLabel == "Example tissue"
    assert dataset.SegmentSequence[0].SegmentAlgorithmType == "MANUAL"
    assert dataset.Manufacturer == "nifti2dicom"
    assert dataset.ManufacturerModelName == "nifti2dicom"


def test_invalid_labels_json_is_actionable(tmp_path, sample_nifti_3d, sample_dicom_dir):
    labels = tmp_path / "labels.json"
    labels.write_text("{invalid json}")
    result = CliRunner().invoke(
        cli,
        [
            str(sample_nifti_3d),
            "--reference",
            str(sample_dicom_dir),
            "--labels",
            str(labels),
            "--json",
        ],
    )
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["error"]["code"] == "invalid_labels"
    assert "label" in report["error"]["message"].lower()
    assert report["error"]["hint"]
    assert result.stderr == ""


def test_inspect_rgb_identifies_the_channel_axis(sample_rgb_nifti, sample_dicom_dir):
    result = CliRunner().invoke(
        cli,
        ["inspect", str(sample_rgb_nifti), "--reference", str(sample_dicom_dir), "--kind", "rgb"],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "channel" in result.stderr
    assert "(1, 3, 4, 4, 3)" in result.stderr


def test_ambiguous_series_error_lists_available_choices(sample_nifti_3d, sample_dicom_dir):
    source = pydicom.dcmread(next(sample_dicom_dir.glob("*.dcm")))
    first_uid = source.SeriesInstanceUID
    second_uid = pydicom.uid.generate_uid()
    source.SeriesInstanceUID = second_uid
    source.SOPInstanceUID = pydicom.uid.generate_uid()
    source.file_meta.MediaStorageSOPInstanceUID = source.SOPInstanceUID
    pydicom.dcmwrite(sample_dicom_dir / "second-series.dcm", source, enforce_file_format=True)
    result = CliRunner().invoke(
        cli, ["inspect", str(sample_nifti_3d), "--reference", str(sample_dicom_dir)]
    )
    assert result.exit_code == 1
    assert first_uid in result.stderr
    assert second_uid in result.stderr
    assert "--series-uid" in result.stderr
    assert result.stdout == ""
