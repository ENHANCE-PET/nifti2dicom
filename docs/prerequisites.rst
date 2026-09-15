Prerequisites
=============

Use Python 3.10 or newer and enough memory to hold the input volume.

A coherent classic CT, MR or PET reference series supplies patient, study and
acquisition context that a NIfTI file cannot reconstruct. Use the original
source series for segmentations. References must have valid physical geometry;
multiple series need an explicit selection.

PET inputs must already use the reference's units. Dynamic PET needs matching
reference frames with trustworthy timing. The converter cannot infer SUV
versus activity concentration from voxel values alone.
