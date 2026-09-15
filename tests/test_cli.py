"""Tests for nifti2dicom.cli."""

from __future__ import annotations

from click.testing import CliRunner

from nifti2dicom.cli import cli


class TestCli:
    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "nifti2dicom" in result.output

    def test_help_shows_subcommands(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "convert" in result.output
        assert "segment" in result.output
        assert "rgb" in result.output
        assert "resample" in result.output
        assert "inspect" in result.output
        assert "--reference" in result.output
        assert "--kind" in result.output
        assert "--json" in result.output
        assert "--version" in result.output
        assert "\x1b[" not in result.output

    def test_convert_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--dicom-dir" in result.output
        assert "--nifti" in result.output

    def test_segment_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["segment", "--help"])
        assert result.exit_code == 0
        assert "--labels-json" in result.output

    def test_rgb_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["rgb", "--help"])
        assert result.exit_code == 0

    def test_resample_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["resample", "--help"])
        assert result.exit_code == 0
        assert "--spatial-dir" in result.output

    def test_no_arguments_prints_help_without_error(self) -> None:
        result = CliRunner().invoke(cli, [])
        assert result.exit_code == 0
        assert "--reference" in result.output

    def test_direct_input_missing_reference_has_actionable_error(self) -> None:
        result = CliRunner().invoke(cli, ["input.nii.gz"])
        assert result.exit_code == 2
        assert "--reference" in result.stderr
        assert "Traceback" not in result.output
