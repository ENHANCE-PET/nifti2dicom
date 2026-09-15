"""DICOM writers consuming validated image and reference contracts."""

from nifti2dicom.writers.image import iter_image_datasets, write_images
from nifti2dicom.writers.rgb import write_rgb

__all__ = ["iter_image_datasets", "write_images", "write_rgb"]
