#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar | Aaron Selfridge
# Institution: Medical University of Vienna | University of California, Davis
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team | EXPLORER Molecular Imaging Center
# Date: 09.02.2023
# Version: 0.1.0
#
# Description:
# The main module of nifti2dicom. This module contains the main function that is executed when nifti2dicom is run.
# ----------------------------------------------------------------------------------------------------------------------

import os
import glob
import pydicom
import nibabel as nib
import numpy as np
from rich.progress import Progress, track
from concurrent.futures import ThreadPoolExecutor
from nifti2dicom.constants import ANSI_ORANGE, ANSI_GREEN, ANSI_VIOLET, ANSI_RESET, ORGAN_INDEX
import emoji
import highdicom as hd
from pydicom.sr.codedict import codes
from pydicom.dataset import Dataset
from datetime import datetime


def check_directory_exists(directory_path: str) -> None:
    """
    Checks if the specified directory exists.

    :param directory_path: The path to the directory.
    :type directory_path: str
    :raises: Exception if the directory does not exist.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Error: The directory '{directory_path}' does not exist.")

def load_reference_dicom_series(directory_path: str) -> tuple:
    """
    Loads a DICOM series from a directory.

    :param directory_path: The path to the directory containing the DICOM series.
    :type directory_path: str
    :return: A tuple containing the slices and filenames of the DICOM series.
    :rtype: tuple
    """
    valid_extensions = ['.dcm', '.ima', '.DCM', '.IMA']
    files = [f for f in glob.glob(os.path.join(directory_path, '*')) if os.path.splitext(f)[1] in valid_extensions and not os.path.basename(f).startswith('.')]
    slices = [pydicom.dcmread(s) for s in files]
    slices_and_names = sorted(zip(slices, files), key=lambda s: s[0].InstanceNumber)
    return zip(*slices_and_names)


def save_slice(slice_data, normalized_data, series_description, filename, output_dir, modality):
    # Convert data to 16-bit integer and set to PixelData
    if modality == "CT":
        # Reverse the rescaling to get back to the original stored values
        slice_data.PixelData = (normalized_data - float(slice_data.RescaleIntercept)) / float(slice_data.RescaleSlope)
    else:
        slice_data.PixelData = normalized_data
    slice_data.PixelData = slice_data.PixelData.astype(np.int16).tobytes()

    slice_data.SeriesNumber *= 10
    if slice_data.SeriesDescription:
        slice_data.SeriesDescription = series_description
    slice_data.save_as(os.path.join(output_dir, os.path.basename(filename)))


def save_dicom_from_nifti_image(ref_dir, nifti_path, output_dir, series_description, force_overwrite=False):
    print('')
    print(f'{ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} IDENTIFIED DATASETS:{ANSI_RESET}')
    print('')

    nifti_image = nib.load(nifti_path)
    image_data = nifti_image.get_fdata()
    num_dims = len(image_data.shape)
    print(f' {ANSI_ORANGE}* Image dimensions: {num_dims}{ANSI_RESET}')
    print(f' {ANSI_GREEN}* Loading NIfTI image: {nifti_path}{ANSI_RESET}')

    image_data = np.flip(image_data, (1, 2))
    image_data = image_data.T
    image_data = image_data.reshape((-1,) + image_data.shape[-2:])

    print(f' {ANSI_GREEN}* Reference DICOM series directory: {ref_dir}{ANSI_RESET}')
    dicom_slices, filenames = load_reference_dicom_series(ref_dir)
    reference_slice = dicom_slices[0]

    modality = reference_slice.Modality

    expected_shape = (len(dicom_slices), reference_slice.Columns, reference_slice.Rows)
    if expected_shape != image_data.shape:
        print(f' {ANSI_ORANGE}* Expected data shape: {expected_shape}, but got: {image_data.shape}{ANSI_RESET}')
        return

    if os.path.exists(output_dir):
        if force_overwrite and os.path.isdir(output_dir):
            print(f' {ANSI_ORANGE} Deleting existing directory: {output_dir}{ANSI_RESET}')
            shutil.rmtree(output_dir)
        else:
            print(f' {ANSI_ORANGE} {output_dir} already exists.{ANSI_RESET}')
            return

    print(f' {ANSI_GREEN}* Output directory: {output_dir}{ANSI_RESET}')
    os.mkdir(output_dir)

    total_slices = len(dicom_slices)
    with Progress() as progress:
        task = progress.add_task("[cyan] Writing DICOM slices:", total=total_slices)

        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, (slice_data, filename) in enumerate(zip(dicom_slices, filenames)):
                normalized_data = image_data[idx]
                futures.append(
                    executor.submit(save_slice, slice_data, normalized_data, series_description, filename, output_dir,
                                    modality))

            for idx, future in enumerate(futures):
                future.result()
                progress.update(task, advance=1,
                                description=f"[cyan] Writing DICOM slices... [{idx + 1}/{total_slices}]")


def save_dicom_from_nifti_seg(nifti_file: str, ref_dicom_series_dir: str, output_path: str, ORGAN_INDEX: dict) -> None:
    """
    Convert a NIFTI segmentation image to a DICOM Segmentation object.

    Parameters
    ----------
    nifti_file : str
        Path to the NIFTI segmentation file.
    ref_dicom_series_dir : str
        Directory containing the reference DICOM series files.
    output_path : str
        Path to save the resulting DICOM Segmentation file.
    ORGAN_INDEX : dict
        Dictionary mapping label values to organ or tissue names.

    Returns
    -------
    None
        The function saves the resulting DICOM file to the specified output path.

    """
    print('')
    print(f'{ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} IDENTIFIED DATASETS:{ANSI_RESET}')
    print('')
    # Load the reference DICOM series
    ref_series = [pydicom.dcmread(f) for f in sorted(glob.glob(os.path.join(ref_dicom_series_dir, "*.dcm")))]
    print(f' {ANSI_GREEN}* Reference DICOM series directory: {ref_dicom_series_dir}{ANSI_RESET}')
    # Load and preprocess the NIFTI segmentation
    print(f' {ANSI_GREEN}* Loading NIfTI segmentation: {nifti_file}{ANSI_RESET}')
    multilabel_mask = nib.load(nifti_file).get_fdata().astype(np.uint8)
    multilabel_mask = np.flip(multilabel_mask, (1, 2))
    multilabel_mask = multilabel_mask.T
    multilabel_mask = multilabel_mask.reshape((-1,) + multilabel_mask.shape[-2:])

    # Generate segment descriptions based on labels in the mask
    labels = np.unique(multilabel_mask)[1:]
    segment_descriptions = []
    for label, organ_name in track(ORGAN_INDEX.items(), description="[cyan] Processing segments...",
                                   total=len(ORGAN_INDEX)):
        category_code = (
            codes.SCT.Organ if organ_name in ['Liver', 'Heart', 'Lung', 'Kidneys', 'Bladder', 'Brain', 'Pancreas',
                                              'Spleen', 'Adrenal-glands']
            else codes.SCT.Tissue
        )
        type_code = codes.SCT.Tissue

        description = hd.seg.SegmentDescription(
            segment_number=int(label),
            segment_label=organ_name,
            segmented_property_category=category_code,
            segmented_property_type=type_code,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )
        segment_descriptions.append(description)

    # Construct the DICOM Segmentation object
    seg = hd.seg.Segmentation(
        source_images=ref_series,
        pixel_array=multilabel_mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=100,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="Quantitative Imaging and Medical Physics",
        manufacturer_model_name="MOOSE (Multi-organ objective segmentation)",
        software_versions="2.0",
        device_serial_number=datetime.now().strftime("%Y%m%d%H%M%S"),  # Using current timestamp as serial number
    )

    # Save the DICOM SEG object with same filename as NIFTI file
    seg.save_as(os.path.join(output_path, os.path.basename(nifti_file) + ".dcm"))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert NIfTI images to DICOM format using a reference DICOM series.")
    parser.add_argument("reference_dir", type=str, help="Path to the directory containing the reference DICOM series.")
    parser.add_argument("nifti_path", type=str, help="Path to the NIfTI file to be converted.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the converted DICOM files will be saved.")
    parser.add_argument("series_description", type=str, help="Series description to be added to the DICOM header.")
    parser.add_argument("is_seg", type=bool, help="True if the NIfTI file is a segmentation, False otherwise.")
    args = parser.parse_args()
    if args.is_seg:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        save_dicom_from_nifti_seg(args.nifti_path, args.reference_dir, args.output_dir, ORGAN_INDEX)
    else:
        save_dicom_from_nifti(args.reference_dir, args.nifti_path, args.output_dir, args.series_description)
