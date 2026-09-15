Installation
============

Python 3.10 or newer is required.

.. code-block:: bash

   pip install nifti2dicom
   nifti2dicom --help

For development, run from the repository:

.. code-block:: bash

   pip install -e '.[dev]'
   pytest
   ruff check nifti2dicom tests
   ruff format --check nifti2dicom tests
   python -m mypy nifti2dicom --ignore-missing-imports
   python -m build
