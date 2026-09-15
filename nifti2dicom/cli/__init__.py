"""Command-line entry point; all conversion work belongs to the public API."""

from nifti2dicom.cli.commands import cli

__all__ = ["cli"]
