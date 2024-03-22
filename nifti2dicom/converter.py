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

import glob
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import emoji
import highdicom as hd
import nibabel as nib
import numpy as np
import pydicom
from nifti2dicom.constants import ANSI_ORANGE, ANSI_GREEN, ANSI_VIOLET, ANSI_RESET, TAGS_TO_EXCLUDE
from nifti2dicom.display import display_welcome_message
from pydicom.sr.codedict import codes
from rich.progress import Progress, track


def check_directory_exists(directory: str) -> None:
    """
    Checks if the specified directory exists.
    :param directory: The path to the directory.
    :type directory: str
    :raises: Exception if the directory does not exist.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Error: The directory '{directory}' does not exist.")


def is_dicom_file(file_path) -> bool:
    try:
        pydicom.dcmread(file_path)
        return True
    except pydicom.errors.InvalidDicomError:
        return False


def is_dicom_compressed(dicom_dataset) -> bool:
    try:
        if 'PixelData' in dicom_dataset:
            transfer_syntax = dicom_dataset.file_meta.TransferSyntaxUID
            uncompressed_syntaxes = [
                pydicom.uid.ExplicitVRLittleEndian,
                pydicom.uid.ImplicitVRLittleEndian,
                pydicom.uid.ExplicitVRBigEndian
            ]
            return transfer_syntax not in uncompressed_syntaxes
        else:
            print("No pixel data found in this DICOM file.")
            return False
    except Exception as e:
        print(f"Failed to check DICOM compression: {e}")
        return False


def load_dicom_series(directory: str) -> tuple:
    """
    Loads a DICOM series from a directory.
    :param directory: The path to the directory containing the DICOM series.
    :type directory: str
    :return: A tuple containing the slices and filenames of the DICOM series.
    :rtype: tuple
    """
    files = [f for f in glob.glob(os.path.join(directory, '*')) if
             is_dicom_file(f) and not os.path.basename(f).startswith('.')]
    slices = [pydicom.dcmread(s) for s in files]
    slices_and_names = sorted(zip(slices, files), key=lambda s: s[0].InstanceNumber)
    return zip(*slices_and_names)


def save_slice(slice_data, normalized_data, series_description, filename, output_dir, modality,
               reference_header_data=None):
    """
    Save a DICOM slice to a file.
    :param slice_data: DICOM slice data
    :type slice_data: pydicom.dataset.Dataset
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
    :param reference_header_data: Modality of the image (CT or PT)
    :type reference_header_data: pydicom.dataset.Dataset
    :return: None
    """
    if is_dicom_compressed(slice_data):
        slice_data.decompress()

    if modality == "CT":
        # Reverse the rescaling to get back to the original stored values
        slice_data.PixelData = (normalized_data - float(slice_data.RescaleIntercept)) / float(slice_data.RescaleSlope)
    elif modality == "PT":
        # Don't ask me why there are different rescaling methods for both vendors
        max_value = np.max(normalized_data)
        if max_value > 65535:
            slice_data.PixelData = normalized_data * (65535 / max_value)
            # fix the rescale slope and intercept accordingly
            slice_data.RescaleSlope = max_value / 65535
            slice_data.RescaleIntercept = 0
        # Reverse the rescaling to get back to the original stored values
        slice_data.PixelData = (normalized_data - float(slice_data.RescaleIntercept)) / float(slice_data.RescaleSlope)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    slice_data.PixelData = slice_data.PixelData.astype(np.int16).tobytes()

    if reference_header_data is not None:
        for tag in slice_data:
            if tag.tag not in reference_header_data:
                del slice_data[tag]

        for tag in reference_header_data:
            parameter_tag_name = tag.name
            if parameter_tag_name not in TAGS_TO_EXCLUDE:
                slice_data[tag.tag].value = tag.value

    slice_data.SeriesNumber *= 10
    if slice_data.SeriesDescription:
        slice_data.SeriesDescription = slice_data.SeriesDescription + '_' + series_description
    slice_data.save_as(os.path.join(output_dir, os.path.basename(filename)))


def vprint(*args, verbose=False, **kwargs):
    """
    Conditional print function that outputs messages to the console if verbose is True.

    :param args: Positional arguments to be printed.
    :param verbose: Flag to control the print behavior. Only prints if True.
    :type verbose: bool
    :param kwargs: Keyword arguments to be passed to the built-in print function.
    """
    if verbose:
        print(*args, **kwargs)


def save_dicom_from_nifti_image(ref_dir, nifti_path, output_dir, vendor="ux",
                                series_description="converted by nifti2dicom", header_dir=None, force_overwrite=False,
                                verbose=False):
    """
    Convert a NIfTI image to a DICOM series, with optional verbose output.
    :param ref_dir: DICOM series directory which serves as a reference for the conversion
    :param nifti_path: Path to the nifti file
    :param output_dir: Output directory to store the converted DICOM series
    :param series_description: Series description to be added to the DICOM header
    :param vendor: The vendor from which the DICOM series was obtained (ux or sms)
    :param header_dir: The path to the header reference directory
    :param force_overwrite: Force overwrite of the output directory if it already exists
    :param verbose: If True, print messages during the process
    :return:
    """
    vprint(verbose=verbose, end='\n')
    vprint(f'{ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} IDENTIFIED DATASETS:{ANSI_RESET}',
           verbose=verbose, end='\n')

    nifti_image = nib.load(nifti_path)
    image_data = nifti_image.get_fdata()
    num_dims = len(image_data.shape)
    vprint(f' {ANSI_ORANGE}* Image dimensions: {num_dims}{ANSI_RESET}', verbose=verbose)
    vprint(f' {ANSI_GREEN}* Loading NIfTI image: {nifti_path}{ANSI_RESET}', verbose=verbose)

    # if the vendor is sms or ux and a 3d image use the following
    if num_dims == 3:
        image_data = np.flip(image_data, (1, 2))
        image_data = image_data.T
        image_data = image_data.reshape((-1,) + image_data.shape[-2:])
    # if the vendor is ux and a 4d image use the following
    elif vendor == 'ux' and num_dims == 4:
        image_data = np.flip(image_data, (1, 2))
        image_data = image_data.T
        image_data = image_data.reshape((-1,) + image_data.shape[-2:])
    # if the vendor is sms and a 4d image use the following
    elif vendor == 'sms' and num_dims == 4:
        image_data = np.flip(image_data, (1, 3))
        image_data = np.flip(image_data, (3,))  # Flip along the time axis
        image_data = image_data.T
        image_data = image_data.reshape((-1,) + image_data.shape[-2:])
    else:
        raise ValueError(f"Unknown vendor: {vendor}")

    header_slice_data = None
    if header_dir is not None:
        vprint(f' {ANSI_GREEN}* Header data will be copied from: {header_dir}{ANSI_RESET}', verbose=verbose)
        vprint(f' {ANSI_GREEN}* Spatial information will be taken from: {ref_dir}{ANSI_RESET}', verbose=verbose)
        parameter_dicom_slices, _ = load_dicom_series(header_dir)
        header_slice_data = parameter_dicom_slices[0]
    else:
        vprint(f' {ANSI_GREEN}* Reference DICOM series directory: {ref_dir}{ANSI_RESET}', verbose=verbose)

    dicom_slices, filenames = load_dicom_series(ref_dir)
    reference_slice = dicom_slices[0]
    if is_dicom_compressed(reference_slice):
        vprint(f' {ANSI_ORANGE}* DICOM is compressed. Will decompress to convert.{ANSI_RESET}', verbose=verbose)

    modality = reference_slice.Modality

    expected_shape = (len(dicom_slices), reference_slice.Columns, reference_slice.Rows)
    if expected_shape != image_data.shape:
        vprint(f' {ANSI_ORANGE}* Expected data shape: {expected_shape}, but got: {image_data.shape}{ANSI_RESET}',
               verbose=verbose)
        return

    if os.path.exists(output_dir):
        if force_overwrite and os.path.isdir(output_dir):
            vprint(f' {ANSI_ORANGE} Deleting existing directory: {output_dir}{ANSI_RESET}', verbose=verbose)
            shutil.rmtree(output_dir)
        else:
            vprint(f' {ANSI_ORANGE} {output_dir} already exists.{ANSI_RESET}', verbose=verbose)
            return

    vprint(f' {ANSI_GREEN}* Output directory: {output_dir}{ANSI_RESET}', verbose=verbose)
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
                                    modality, header_slice_data))

            for idx, future in enumerate(futures):
                future.result()
                progress.update(task, advance=1,
                                description=f"[white] Writing DICOM slices... [{idx + 1}/{total_slices}]")


def save_dicom_from_nifti_seg(nifti_file: str, ref_dicom_series_dir: str, output_path: str, ORGAN_INDEX: dict,
                              verbose=False) -> None:
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
    :param verbose: If True, print messages during the process
    :type verbose: bool

    """
    vprint(verbose=verbose, end='\n')
    vprint(f'{ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} IDENTIFIED DATASETS:{ANSI_RESET}',
           verbose=verbose, end='\n')

    # Load the reference DICOM series
    ref_series = [pydicom.dcmread(f) for f in sorted(glob.glob(os.path.join(ref_dicom_series_dir, "*.dcm")))]
    vprint(f' {ANSI_GREEN}* Reference DICOM series directory: {ref_dicom_series_dir}{ANSI_RESET}', verbose=verbose)
    # Load and preprocess the NIFTI segmentation
    vprint(f' {ANSI_GREEN}* Loading NIfTI segmentation: {nifti_file}{ANSI_RESET}', verbose=verbose)
    multilabel_mask = nib.load(nifti_file).get_fdata().astype(np.uint8)
    multilabel_mask = np.flip(multilabel_mask, (1, 2))
    multilabel_mask = multilabel_mask.T
    multilabel_mask = multilabel_mask.reshape((-1,) + multilabel_mask.shape[-2:])

    # Generate segment descriptions based on labels in the mask
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
    parser.add_argument("-d", "--dicom_dir", type=str, required=True,
                        help="Path to the directory containing the reference DICOM series.")
    parser.add_argument("-hd", "--header_source_dicom_dir", type=str, required=False, default=None,
                        help="Path to the directory containing the header reference DICOM series.")
    parser.add_argument("-n", "--nifti_path", type=str, required=True,
                        help="Path to the NIfTI file to be converted.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Path to the directory where the converted DICOM files will  be saved.")
    parser.add_argument("-desc", "--series_description", required=False, default='converted by nifti2dicom',
                        type=str, help="Series description to be added to the DICOM header.")
    parser.add_argument("-t", "--type", type=str, choices=['img', 'seg'], required=True,
                        help="Are you converting an image or a segmentation?")
    parser.add_argument("-v", "--vendor", type=str, choices=['sms', 'ux'], required=False, default='ux',
                        help="Vendor of the reference DICOM series. Only needed for 4D images.")
    parser.add_argument("-j", "--json", type=str,
                        help=f"Path to the JSON file containing the label to region index. ")

    # Parse the arguments
    args = parser.parse_args()

    # Display the welcome message
    display_welcome_message()

    # Check the type of conversion
    if args.type == 'seg' and args.json:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        with open(args.json, 'r') as f:
            organ_index = json.load(f)
        save_dicom_from_nifti_seg(args.nifti_path, args.dicom_dir, args.output_dir, organ_index)
    elif args.type == 'seg' and not args.json:
        raise ValueError(f"Please provide a JSON file containing the label to region index.")

    elif args.type == 'img':
        save_dicom_from_nifti_image(args.dicom_dir, args.nifti_path, args.output_dir, args.vendor,
                                    args.series_description, args.header_source_dicom_dir)
    else:
        raise ValueError(f"Unknown type: {args.type}")
