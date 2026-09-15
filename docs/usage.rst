Usage
=====

Images
------

.. code-block:: bash

   nifti2dicom scan.nii.gz --reference ./dicom
   nifti2dicom inspect scan.nii.gz --reference ./dicom
   nifti2dicom scan.nii.gz --reference ./dicom -o ./converted --geometry reference

The default output is a new scan_dicom directory beside the input, containing
DICOM files and a conversion.json report. Existing output requires an explicit
--overwrite. Input files are protected from replacement.

Use --series-uid when the reference directory contains several series.
Use --json for machine-readable results and errors, --quiet for no normal
output, and --debug to diagnose unexpected failures.

Segmentations
-------------

.. code-block:: bash

   nifti2dicom mask.nii.gz --reference ./dicom --kind seg --algorithm-type manual
   nifti2dicom organs.nii.gz --reference ./dicom --labels labels.json \
     --algorithm-type automatic --algorithm-name MyModel --algorithm-version 1.0

Use 0 for background and positive integers for segments. Binary and multilabel
masks both use BINARY SEG; BINARY does not mean a single segment.
Sparse input labels are remapped without losing their original identity.
A simple labels.json maps label numbers to names:

.. code-block:: json

   {"5": "Lesion", "300": "Liver"}

For interoperable anatomy, provide coded category and type descriptions.
See the repository README for the full schema and MOOSE compatibility.
Missing semantic codes produce an explicit unspecified-region warning.
Algorithm provenance is required rather than guessed.

SEG uses nearest-neighbor interpolation onto the reference grid. Foreground
outside that grid or lost segments cause a clear error. Probability maps
are never silently thresholded.

Python
------

.. code-block:: python

   from nifti2dicom import convert, inspect, ConversionError

   info = inspect("scan.nii.gz", "./dicom")
   try:
       result = convert("scan.nii.gz", "./dicom", "./converted")
   except ConversionError as error:
       print(error.message, error.hint)
   else:
       print(result.files)
       print(result.warnings)

Library calls are quiet. An optional on_progress callback receives typed
events. Historical conversion functions and CLI subcommands remain adapters
to the same implementation.
