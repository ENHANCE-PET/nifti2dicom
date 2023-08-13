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
# The display module of nifti2dicom. This module contains the display messages for nifti2dicom.
# ----------------------------------------------------------------------------------------------------------------------


import pyfiglet
from nifti2dicom import constants


def display_welcome_message():
    """
    Displays the welcome message.
    :return:
    """
    logo_color_code = constants.ANSI_VIOLET
    print(' ')
    result = logo_color_code + pyfiglet.figlet_format('nifti2dicom', font='slant').rstrip()  + constants.ANSI_RESET
    text = (logo_color_code + 'A package to convert NIfTI images to DICOM format using a reference DICOM series. '
                              'Nifti2dicom is a part of the ENHANCE.PET (https://enhance.pet) framework' +
            constants.ANSI_RESET)

    print(result)
    print(text)
    print(' ')


