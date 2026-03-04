"""NIfTI → DICOM SEG conversion.

Fixes:
  - Scans all files (not just ``*.dcm``)
  - Configurable manufacturer / model name
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import highdicom as hd
import nibabel as nib
import numpy as np
from pydicom.sr.codedict import codes

from nifti2dicom import cli_theme as theme
from nifti2dicom.dicom_io import load_dicom_series
from nifti2dicom.orientation import orient_nifti


def convert_nifti_seg_to_dicom(
    ref_dir: str | Path,
    nifti_path: str | Path,
    output_path: str | Path,
    organ_index: dict[str, str],
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
    organ_index : dict[str, str]
        Mapping of ``{label_int: organ_name}``.
    manufacturer : str
        DICOM Manufacturer tag.
    manufacturer_model_name : str
        DICOM ManufacturerModelName tag.
    software_versions : str
        DICOM SoftwareVersions tag.
    """
    ref_dir = Path(ref_dir)
    nifti_path = Path(nifti_path)
    output_path = Path(output_path)

    theme.section("LOADING", number="01")
    theme.info(f"Segmentation NIfTI: {nifti_path}")
    theme.info(f"Reference DICOM: {ref_dir}")

    # Load reference DICOM — scan all files, not just *.dcm
    ref_slices, _ = load_dicom_series(ref_dir)

    # Load and orient the segmentation
    img: nib.Nifti1Image = nib.load(str(nifti_path))  # type: ignore[assignment]
    data, _, _ = orient_nifti(img)
    multilabel_mask = data.astype(np.uint8)

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
    seg.save_as(str(out_file))

    theme.ok(f"Wrote DICOM SEG to {out_file}")
