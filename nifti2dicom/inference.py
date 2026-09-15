"""Resolve image meaning from explicit choices and NIfTI metadata."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel as nib

from nifti2dicom.errors import InputError
from nifti2dicom.models import ImageKind


def infer_kind(path: Path, kind: str, *, has_labels: bool) -> ImageKind:
    if kind not in {"auto", "image", "seg", "rgb"}:
        raise InputError(
            f"Unknown conversion kind '{kind}'.", hint="Choose auto, image, seg or rgb."
        )
    if kind != "auto":
        if has_labels and kind != "seg":
            raise InputError(
                "Label descriptions were supplied for a non-segmentation conversion.",
                hint="Use kind='seg' or remove the labels option.",
            )
        return cast(ImageKind, kind)
    if has_labels:
        return "seg"
    try:
        img = nib.load(str(path))
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
        raise InputError(
            f"Could not open NIfTI image: {path}", hint="Choose a readable .nii or .nii.gz file."
        ) from exc
    if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
        raise InputError("This input is not a NIfTI image.", hint="Use a .nii or .nii.gz file.")
    if img.get_data_dtype().fields is not None and {"R", "G", "B"} <= set(
        img.get_data_dtype().fields
    ):
        return "rgb"
    if int(img.header["intent_code"]) == 1002:  # NIFTI_INTENT_LABEL
        return "seg"
    return "image"


def default_output(path: Path) -> Path:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return path.parent / f"{name}_dicom"
