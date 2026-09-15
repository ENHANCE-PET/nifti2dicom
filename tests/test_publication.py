"""Failed replacement must retain the user's previous output, even on rollback failure."""

from pathlib import Path

import pytest

from nifti2dicom.errors import OutputError
from nifti2dicom.publication import staged_output


def test_double_rename_failure_retains_recoverable_backup(tmp_path, monkeypatch):
    target = tmp_path / "result"
    target.mkdir()
    (target / "original.txt").write_text("irreplaceable")
    rename = Path.rename

    def fail_publish_and_rollback(self, destination):
        if ".staging-" in self.name or self.name == "previous":
            raise PermissionError("simulated rename failure")
        return rename(self, destination)

    monkeypatch.setattr(Path, "rename", fail_publish_and_rollback)
    with pytest.raises(OutputError) as caught:
        with staged_output(target, overwrite=True) as stage:
            (stage / "new.dcm").write_bytes(b"new")
    backups = list(tmp_path.rglob("original.txt"))
    assert len(backups) == 1
    assert backups[0].read_text() == "irreplaceable"
    assert str(backups[0].parent) in caught.value.hint
