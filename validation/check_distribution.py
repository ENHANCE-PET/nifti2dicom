"""Reject local databases, imaging data and work files in built distributions."""

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def check_archive(path: Path) -> int:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        required = {"nifti2dicom/api.py"}
        if not any(name.endswith(".dist-info/METADATA") for name in names):
            raise ValueError("Wheel is missing package metadata.")
    else:
        with tarfile.open(path) as archive:
            names = [
                str(PurePosixPath(name).relative_to(PurePosixPath(name).parts[0]))
                for name in archive.getnames()
            ]
        required = {"nifti2dicom/api.py", "pyproject.toml", "README.md", "LICENSE"}
    forbidden = []
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            any(part in {".superpowers", ".git", ".venv", "venv"} for part in parts)
            or name.startswith("docs/superpowers/")
            or any(part.startswith(".env") for part in parts)
            or name.lower().endswith(
                (
                    ".sql",
                    ".sqlite",
                    ".sqlite3",
                    ".db",
                    ".dcm",
                    ".nii",
                    ".nii.gz",
                    ".nrrd",
                    ".mha",
                    ".mhd",
                    ".npz",
                )
            )
        ):
            forbidden.append(name)
    if forbidden:
        raise ValueError("Distribution contains local/data artifacts: " + ", ".join(forbidden))
    missing = required - set(names)
    if missing:
        raise ValueError("Distribution is missing required files: " + ", ".join(sorted(missing)))
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.archives:
        print(f"{path.name}: {check_archive(path)} entries checked; no local/data artifacts")


if __name__ == "__main__":
    main()
