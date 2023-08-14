#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar | Aaron Selfridge | Siqi Li
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
from nifti2dicom.constants import ANSI_ORANGE, ANSI_GREEN, ANSI_VIOLET, ANSI_RESET
import emoji
import highdicom as hd
from pydicom.sr.codedict import codes
from pydicom.dataset import Dataset
from datetime import datetime
from nifti2dicom.display import display_welcome_message
import json 

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


def save_slice(slice_data, normalized_data, series_description, filename, output_dir, modality, vendor):
    """
    Save a DICOM slice to a file.
    :param slice_data: DICOM slice data
    :type slice_data: Dataset
    :param normalized_data: Normalized data from the NIfTI image
    :type normalized_data: numpy.ndarray
    :param series_description: Description of the series
    :type series_description: str
    :param filename: output filename of the converted DICOM slice
    :type filename: str
    :param output_dir: output directory to store the converted DICOM slice
    :type output_dir: str
    :param modality: Modality of the image (CT or PT)
    :type modality: str
    :param vendor: Vendor from which the DICOM series was obtained (ux or sms)
    :type vendor: str
    :return: None
    """
    if modality == "CT":
        # Reverse the rescaling to get back to the original stored values
        slice_data.PixelData = (normalized_data - float(slice_data.RescaleIntercept)) / float(slice_data.RescaleSlope)
    elif modality == "PT":
        # Don't ask me why there are different rescaling methods for both vendors
        if vendor == 'ux':
            slice_data.PixelData = normalized_data
            slice_data.RescaleIntercept = 0
            slice_data.RescaleSlope = 1
        elif vendor == 'sms':
            max_value = np.max(normalized_data)
            if max_value > 65535:
                slice_data.PixelData = normalized_data * (65535 / max_value)
                # fix the rescale slope and intercept accordingly
                slice_data.RescaleSlope = max_value / 65535
                slice_data.RescaleIntercept = 0
            # Reverse the rescaling to get back to the original stored values
            slice_data.PixelData = (normalized_data - float(slice_data.RescaleIntercept)) / float(slice_data.RescaleSlope)
        else:
            raise ValueError(f"Unknown vendor: {vendor}")
    else:
        raise ValueError(f"Unknown modality: {modality}")
    slice_data.PixelData = slice_data.PixelData.astype(np.int16).tobytes()

    slice_data.SeriesNumber *= 10
    if slice_data.SeriesDescription:
        slice_data.SeriesDescription = series_description
    slice_data.save_as(os.path.join(output_dir, os.path.basename(filename)))


def save_dicom_from_nifti_image(ref_dir, nifti_path, output_dir, series_description, vendor, force_overwrite=False):
    """
    Convert a NIfTI image to a DICOM series.
    :param ref_dir: DICOM series directory which serves as a reference for the conversion
    :type ref_dir: str
    :param nifti_path: Path to the nifti file
    :type nifti_path: str
    :param output_dir: Output directory to store the converted DICOM series
    :type output_dir: str
    :param series_description: Series description to be added to the DICOM header
    :type series_description: str
    :param vendor: The vendor from which the DICOM series was obtained (ux or sms)
    :type vendor: str
    :param force_overwrite: Force overwrite of the output directory if it already exists
    :type force_overwrite: bool
    :return:
    """
    print('')
    print(f'{ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} IDENTIFIED DATASETS:{ANSI_RESET}')
    print('')

    nifti_image = nib.load(nifti_path)
    image_data = nifti_image.get_fdata()
    num_dims = len(image_data.shape)
    print(f' {ANSI_ORANGE}* Image dimensions: {num_dims}{ANSI_RESET}')
    print(f' {ANSI_GREEN}* Loading NIfTI image: {nifti_path}{ANSI_RESET}')

    if vendor == 'ux':
        image_data = np.flip(image_data, (1, 2))
        image_data = image_data.T
        image_data = image_data.reshape((-1,) + image_data.shape[-2:])
    if vendor == 'sms':
        image_data = np.flip(image_data, (1, 3))
        image_data = np.flip(image_data, (3,))  # Flip along the time axis
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
                                    modality, vendor))

            for idx, future in enumerate(futures):
                future.result()
                progress.update(task, advance=1,
                                description=f"[cyan] Writing DICOM slices... [{idx + 1}/{total_slices}]")


def save_dicom_from_nifti_seg(nifti_file: str, ref_dicom_series_dir: str, output_path: str, ORGAN_INDEX: dict) -> None:
    """
    Convert a NIFTI segmentation image to a DICOM Segmentation object.
    :param nifti_file: Path to the NIFTI segmentation file.
    :type nifti_file: str
    :param ref_dicom_series_dir: Path to the directory containing the reference DICOM series.
    :type ref_dicom_series_dir: str
    :param output_path: Path to the directory where the converted DICOM files will be saved.
    :type output_path: str
    :param ORGAN_INDEX: Dictionary containing the organ index.
    :type ORGAN_INDEX: dict
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
    parser.add_argument("-d", "--dicom_dir", type=str, help="Path to the directory containing the reference DICOM "
                                                          "series.")
    parser.add_argument("-n", "--nifti_path", type=str, help="Path to the NIfTI file to be converted.")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the directory where the converted DICOM files "
                                                             "will  be saved.")
    parser.add_argument("-desc", "--series_description", type=str, help="Series description to be added to the DICOM header.")
    parser.add_argument("-t", "--type", type=str, choices=['img','seg'], help=("Are you converting an image or a segmentation?."))
    # vendor is optional because it is not needed for the segmentation conversion

    parser.add_argument("-v", "--vendor", type=str, choices=['sms', 'ux'], required=False, default='ux',
                        help="Vendor of the reference DICOM series.")
    parser.add_argument("-j", "--json", type=str, help=f"Path to the JSON file containing the label to region index. ")
    


    # Parse the arguments
    args = parser.parse_args()

    # Display the welcome message
    display_welcome_message()

    # Check the type of conversion

    if args.type == 'seg' and args.json: # if the type is segmentation and the json file is provided
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        with open(args.json, 'r') as f:
            organ_index = json.load(f)
        save_dicom_from_nifti_seg(args.nifti_path, args.dicom_dir, args.output_dir, organ_index)
    elif args.type == 'seg' and not args.json: # if the type is segmentation and the json file is not provided
        raise ValueError(f"Please provide a JSON file containing the label to region index.")
    
    elif args.type == 'img':  # if the type is image
        save_dicom_from_nifti_image(args.dicom_dir, args.nifti_path, args.output_dir, args.series_description, args.vendor)
    else:
        raise ValueError(f"Unknown type: {args.type}")