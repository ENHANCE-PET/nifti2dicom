"""Allow running as ``python -m nifti2dicom``."""

from __future__ import annotations

from nifti2dicom.cli import cli

if __name__ == "__main__":
    cli()
