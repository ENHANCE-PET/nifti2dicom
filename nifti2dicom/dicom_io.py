"""Fast DICOM loading — magic-byte check, single-pass read."""

from __future__ import annotations

from pathlib import Path

import pydicom

from nifti2dicom.exceptions import NoDicomFilesError

_DICM_MAGIC = b"DICM"
_PREAMBLE_OFFSET = 128


def is_dicom_file(path: str | Path) -> bool:
    """Check whether *path* is a DICOM file using the 128-byte preamble + DICM magic.

    Falls back to pydicom for files without the standard preamble (some old DICOM
    files omit it).
    """
    path = Path(path)
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    try:
        with open(path, "rb") as fh:
            fh.seek(_PREAMBLE_OFFSET)
            magic = fh.read(4)
        if magic == _DICM_MAGIC:
            return True
        # Fallback: try reading without the preamble (rare but valid)
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        # Validate it's actually DICOM by checking for a fundamental attribute
        return hasattr(ds, "SOPClassUID") or hasattr(ds, "Modality")
    except Exception:
        return False


def load_dicom_series(directory: str | Path) -> tuple[list[pydicom.Dataset], list[str]]:
    """Load all DICOM files from *directory* in a single pass.

    Returns (slices, filenames) sorted by InstanceNumber.

    Raises :class:`NoDicomFilesError` if no valid DICOM files are found.
    """
    directory = Path(directory)
    pairs: list[tuple[pydicom.Dataset, str]] = []

    for entry in sorted(directory.iterdir()):
        if not entry.is_file() or entry.name.startswith("."):
            continue
        # Fast magic-byte check first
        try:
            with open(entry, "rb") as fh:
                fh.seek(_PREAMBLE_OFFSET)
                magic = fh.read(4)
            if magic != _DICM_MAGIC:
                # Fallback: rare DICOM files without preamble
                try:
                    ds = pydicom.dcmread(str(entry), force=True)
                    if not (hasattr(ds, "SOPClassUID") or hasattr(ds, "Modality")):
                        continue
                    pairs.append((ds, str(entry)))
                    continue
                except Exception:
                    continue
            ds = pydicom.dcmread(str(entry))
            pairs.append((ds, str(entry)))
        except Exception:
            continue

    if not pairs:
        raise NoDicomFilesError(str(directory))

    pairs.sort(key=lambda p: int(getattr(p[0], "InstanceNumber", 0)))
    slices = [p[0] for p in pairs]
    filenames = [p[1] for p in pairs]
    return slices, filenames


def is_dicom_compressed(ds: pydicom.Dataset) -> bool:
    """Check whether a DICOM dataset uses compressed transfer syntax."""
    if "PixelData" not in ds:
        return False
    try:
        ts = ds.file_meta.TransferSyntaxUID
        uncompressed = {
            pydicom.uid.ExplicitVRLittleEndian,
            pydicom.uid.ImplicitVRLittleEndian,
            pydicom.uid.ExplicitVRBigEndian,
        }
        return ts not in uncompressed
    except AttributeError:
        return False
