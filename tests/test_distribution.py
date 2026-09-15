"""Archive inspection works on actual archives without extracting their contents."""

import io
import tarfile
import zipfile

import pytest

from validation.check_distribution import check_archive


def archive_fixture(tmp_path, kind, extra=None):
    names = ["nifti2dicom/api.py", "pyproject.toml", "README.md", "LICENSE"]
    if extra:
        names.append(extra)
    path = tmp_path / ("example.whl" if kind == "wheel" else "example.tar.gz")
    if kind == "wheel":
        with zipfile.ZipFile(path, "w") as archive:
            for name in names + ["example.dist-info/METADATA"]:
                archive.writestr(name, b"fixture")
    else:
        with tarfile.open(path, "w:gz") as archive:
            for name in names:
                info = tarfile.TarInfo("example-1.0/" + name)
                info.size = 7
                archive.addfile(info, io.BytesIO(b"fixture"))
    return path


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_clean_distributions_pass(tmp_path, kind):
    assert check_archive(archive_fixture(tmp_path, kind)) > 0


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "name",
    [
        "ctkDICOM.sql",
        ".superpowers/sdd/progress.md",
        "docs/superpowers/plan.md",
        "validation/patient.nii.gz",
        "tests/image.dcm",
        "validation/truth.npz",
        ".env",
    ],
)
def test_distribution_rejects_local_or_imaging_artifacts(tmp_path, kind, name):
    with pytest.raises(ValueError, match="local/data artifacts"):
        check_archive(archive_fixture(tmp_path, kind, name))
