"""NIfTI → DICOM SEG conversion.

Uses SimpleITK to resample the NIfTI segmentation onto the reference
DICOM geometry so the mask is pixel-aligned with the source images.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import highdicom as hd
import SimpleITK as sitk
from pydicom.filewriter import dcmwrite
from pydicom.sr.codedict import codes

from nifti2dicom import cli_theme as theme
from nifti2dicom.dicom_io import load_dicom_series


def _normalize_organ_index(raw: dict) -> dict[str, str]:
    """Normalize label JSON into a flat ``{label: name}`` dict.

    Accepted formats
    ----------------
    Flat::

        {"1": "liver", "2": "spleen"}

    MOOSE-nested::

        {"organ_indices": {"1": {"name": "liver", "SNOMED": {...}}, ...}}

    Raises
    ------
    ValueError
        If the structure is unrecognised or empty.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"Label JSON must be a dict, got {type(raw).__name__}"
        )

    # Unwrap MOOSE wrapper
    if "organ_indices" in raw and isinstance(raw["organ_indices"], dict):
        raw = raw["organ_indices"]

    if not raw:
        raise ValueError("Label JSON is empty — no organ labels found")

    organ_index: dict[str, str] = {}
    for label, value in raw.items():
        if isinstance(value, str):
            organ_index[label] = value
        elif isinstance(value, dict) and isinstance(value.get("name"), str):
            organ_index[label] = value["name"]
        else:
            raise ValueError(
                f"Unrecognized format for label '{label}': "
                f"expected a string or a dict with a 'name' string"
            )
    return organ_index


def convert_nifti_seg_to_dicom(
    ref_dir: str | Path,
    nifti_path: str | Path,
    output_path: str | Path,
    organ_index: dict,
    *,
    manufacturer: str = "Quantitative Imaging and Medical Physics",
    manufacturer_model_name: str = "nifti2dicom",
    software_versions: str = "2.0",
) -> None:
    """Convert a NIfTI segmentation to a DICOM SEG object.

    Parameters
    ----------
    ref_dir : path
        Directory containing the reference DICOM series.
    nifti_path : path
        Path to the multilabel NIfTI segmentation file.
    output_path : path
        Output directory for the DICOM SEG file.
    organ_index : dict
        Label mapping — accepts both flat ``{"1": "liver"}`` and
        MOOSE-nested ``{"organ_indices": {"1": {"name": ...}}}`` formats.
    manufacturer : str
        DICOM Manufacturer tag.
    manufacturer_model_name : str
        DICOM ManufacturerModelName tag.
    software_versions : str
        DICOM SoftwareVersions tag.
    """
    organ_index = _normalize_organ_index(organ_index)

    ref_dir = Path(ref_dir)
    nifti_path = Path(nifti_path)
    output_path = Path(output_path)

    theme.section("LOADING", number="01")
    theme.info(f"Segmentation NIfTI: {nifti_path}")
    theme.info(f"Reference DICOM: {ref_dir}")

    # Load reference DICOM — scan all files, not just *.dcm
    ref_slices, _ = load_dicom_series(ref_dir)

    # Load reference DICOM geometry via SimpleITK
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(ref_dir))
    reader.SetFileNames(dicom_names)
    ref_sitk = reader.Execute()

    # Load NIfTI segmentation and resample onto reference geometry.
    # SimpleITK handles the RAS↔LPS mapping; nearest-neighbour
    # preserves integer labels.
    seg_sitk = sitk.ReadImage(str(nifti_path), sitk.sitkUInt8)
    resampled = sitk.Resample(
        seg_sitk,
        ref_sitk,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,  # background value
    )
    # GetArrayFromImage → (Z, Y, X) matching DICOM pixel order
    multilabel_mask = sitk.GetArrayFromImage(resampled)

    theme.info(f"Mask shape: {multilabel_mask.shape}, labels: {len(organ_index)}")

    # Build segment descriptions
    theme.section("SEGMENTS", number="02")

    organ_categories = {
        "Liver", "Heart", "Lung", "Kidneys", "Bladder",
        "Brain", "Pancreas", "Spleen", "Adrenal-glands",
    }

    segment_descriptions = []
    for label_str, organ_name in organ_index.items():
        category_code = (
            codes.SCT.Organ if organ_name in organ_categories else codes.SCT.Tissue
        )
        desc = hd.seg.SegmentDescription(
            segment_number=int(label_str),
            segment_label=organ_name,
            segmented_property_category=category_code,
            segmented_property_type=codes.SCT.Tissue,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )
        segment_descriptions.append(desc)
        theme.info(f"Segment {label_str}: {organ_name}")

    # Construct DICOM SEG
    theme.section("WRITING", number="03")

    seg = hd.seg.Segmentation(
        source_images=ref_slices,
        pixel_array=multilabel_mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=100,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=manufacturer,
        manufacturer_model_name=manufacturer_model_name,
        software_versions=software_versions,
        device_serial_number=datetime.now().strftime("%Y%m%d%H%M%S"),
    )

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / (nifti_path.name + ".dcm")
    dcmwrite(str(out_file), seg, write_like_original=False)

    theme.ok(f"Wrote DICOM SEG to {out_file}")
