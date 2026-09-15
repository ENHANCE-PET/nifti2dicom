Introduction
============

nifti2dicom converts scalar 2D, 3D and 4D NIfTI images into classic CT, MR
or PET series using a reference DICOM scan. It also supports binary and
multilabel DICOM SEG, and 2D/3D RGB Secondary Capture.

One shared pipeline serves the Python API and command line. It reads and
validates inputs, resolves physical geometry, encodes output in staging,
reads it back, and publishes only after validation succeeds.
