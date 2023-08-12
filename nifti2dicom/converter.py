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
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor
from nifti2dicom.constants import ANSI_ORANGE, ANSI_GREEN, ANSI_VIOLET, ANSI_RESET
import emoji

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


def save_dicom_from_nifti(ref_dir, nifti_path, output_dir, series_description, force_overwrite=False):
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert NIfTI images to DICOM format using a reference DICOM series.")
    parser.add_argument("reference_dir", type=str, help="Path to the directory containing the reference DICOM series.")
    parser.add_argument("nifti_path", type=str, help="Path to the NIfTI file to be converted.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the converted DICOM files will be saved.")
    parser.add_argument("series_description", type=str, help="Series description to be added to the DICOM header.")

    args = parser.parse_args()
    save_dicom_from_nifti(args.reference_dir, args.nifti_path, args.output_dir, args.series_description)
