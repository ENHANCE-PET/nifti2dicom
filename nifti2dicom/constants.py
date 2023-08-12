#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 09.02.2023
# Version: 2.0.0
#
# Description:
# The main module of nifti2dicom. This module contains the main function that is executed when nifti2dicom is run.
# ----------------------------------------------------------------------------------------------------------------------


# COLOR CODES
ANSI_ORANGE = '\033[38;5;208m'
ANSI_GREEN = '\033[38;5;40m'
ANSI_VIOLET = '\033[38;5;141m'
ANSI_RESET = '\033[0m'


ORGAN_INDEX = {
    1: 'Adrenal-glands',
    2: 'Aorta',
    3: 'Bladder',
    4: 'Brain',
    5: 'Heart',
    6: 'Kidneys',
    7: 'Liver',
    8: 'Pancreas',
    9: 'Spleen',
    10: 'Thyroid',
    11: 'Inferior-vena-cava',
    12: 'Lung',
    13: 'Carpal',
    14: 'Clavicle',
    15: 'Femur',
    16: 'Fibula',
    17: 'Humerus',
    18: 'Metacarpal',
    19: 'Metatarsal',
    20: 'Patella',
    21: 'Pelvis',
    22: 'Phalanges-of-the-hand',
    23: 'Radius',
    24: 'Ribcage',
    25: 'Scapula',
    26: 'Skull',
    27: 'Spine',
    28: 'Sternum',
    29: 'Tarsal',
    30: 'Tibia',
    31: 'Phalanges-of-the-feet',
    32: 'Ulna',
    33: 'Skeletal-muscle',
    34: 'Subcutaneous-fat',
    35: 'Torso-fat',
    36: 'Psoas'
}
