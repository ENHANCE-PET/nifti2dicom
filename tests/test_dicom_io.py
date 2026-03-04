"""Tests for nifti2dicom.dicom_io."""

from __future__ import annotations

from pathlib import Path

import pydicom
import pytest

from nifti2dicom.dicom_io import is_dicom_file, load_dicom_series, is_dicom_compressed
from nifti2dicom.exceptions import NoDicomFilesError


class TestIsDicomFile:
    def test_valid_dicom(self, sample_dicom_dir: Path) -> None:
        dcm = next(sample_dicom_dir.glob("*.dcm"))
        assert is_dicom_file(dcm)

    def test_non_dicom_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "not_dicom.txt"
        txt.write_text("hello")
        assert not is_dicom_file(txt)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert not is_dicom_file(tmp_path / "missing.dcm")

    def test_hidden_file_excluded(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.write_text("data")
        assert not is_dicom_file(hidden)


class TestLoadDicomSeries:
    def test_loads_sorted_by_instance(self, sample_dicom_dir: Path) -> None:
        slices, filenames = load_dicom_series(sample_dicom_dir)
        assert len(slices) == 3
        assert len(filenames) == 3
        # Verify sorted by InstanceNumber
        instance_nums = [int(s.InstanceNumber) for s in slices]
        assert instance_nums == sorted(instance_nums)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(NoDicomFilesError):
            load_dicom_series(empty)


class TestIsDicomCompressed:
    def test_uncompressed_returns_false(self, sample_dicom_dir: Path) -> None:
        ds = pydicom.dcmread(str(next(sample_dicom_dir.glob("*.dcm"))))
        assert not is_dicom_compressed(ds)

    def test_no_pixel_data_returns_false(self) -> None:
        ds = pydicom.Dataset()
        assert not is_dicom_compressed(ds)
