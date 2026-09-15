"""Resampling is a physical-coordinate operation, never an array flip."""

import numpy as np
import pytest

from nifti2dicom.errors import GeometryError
from nifti2dicom.models import Geometry, ImageVolume


@pytest.mark.parametrize("problem", ["nan", "zero", "singular", "shear", "last_row"])
def test_invalid_spatial_geometry_is_rejected(problem):
    from nifti2dicom.geometry import validate_geometry

    affine = np.eye(4)
    if problem == "nan":
        affine[0, 0] = np.nan
    elif problem == "zero":
        affine[0, 0] = 0
    elif problem == "singular":
        affine[:3, 1] = affine[:3, 0]
    elif problem == "shear":
        affine[0, 1] = 0.05
    else:
        affine[3, 0] = 1
    with pytest.raises(GeometryError):
        validate_geometry(Geometry(affine, (2, 3, 4)))


def test_identity_resampling_preserves_labels_and_multiple_timepoints():
    from nifti2dicom.geometry import resample_to_reference

    data = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    data[0, 1, 2, 3] = 300
    data[1, 2, 1, 4] = 17
    geometry = Geometry(np.diag([-2.0, -3.0, 5.0, 1.0]), (5, 4, 3))
    result = resample_to_reference(
        ImageVolume(data, geometry, time_spacing=1.25), geometry, labels=True
    )
    np.testing.assert_array_equal(result.data, data)
    assert result.time_spacing == 1.25


def test_resampling_rotates_and_translates_in_physical_space():
    from nifti2dicom.geometry import resample_to_reference

    data = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)
    source = ImageVolume(data, Geometry(np.eye(4), (4, 3, 2)))
    affine = np.array([[0, -1, 0, 3], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    result = resample_to_reference(source, Geometry(affine, (3, 4, 2)))
    assert result.data.shape == (1, 2, 4, 3)
    assert result.data[0, 1, 2, 1] == data[0, 1, 1, 1]
    assert result.data[0, 0, 0, 2] == data[0, 0, 2, 3]


def test_linear_interpolation_retains_fractional_values_and_nearest_retains_labels():
    from nifti2dicom.geometry import resample_to_reference

    source = ImageVolume(
        np.array([[[[0, 10, 20]]]], dtype=np.int16), Geometry(np.eye(4), (3, 1, 1))
    )
    affine = np.eye(4)
    affine[0, 3] = 0.25
    target = Geometry(affine, (2, 1, 1))
    np.testing.assert_allclose(resample_to_reference(source, target).data, [[[[2.5, 12.5]]]])
    np.testing.assert_array_equal(
        resample_to_reference(source, target, labels=True).data, [[[[0, 10]]]]
    )


def test_rgb_resampling_preserves_channels():
    from nifti2dicom.geometry import resample_to_reference

    data = np.array([[[[[0.0, 10.0, 100.0], [10.0, 20.0, 200.0]]]]])
    source = ImageVolume(data, Geometry(np.eye(4), (2, 1, 1)), kind="rgb")
    affine = np.eye(4)
    affine[0, 3] = 0.5
    output = resample_to_reference(source, Geometry(affine, (1, 1, 1)))
    np.testing.assert_allclose(output.data[0, 0, 0, 0], [5, 15, 150])


def test_partial_fov_is_allowed_but_no_overlap_fails():
    from nifti2dicom.geometry import resample_to_reference

    source = ImageVolume(np.ones((1, 2, 3, 4)), Geometry(np.eye(4), (4, 3, 2)))
    partial_affine = np.eye(4)
    partial_affine[0, 3] = 2
    result = resample_to_reference(source, Geometry(partial_affine, (4, 3, 2)))
    np.testing.assert_array_equal(result.data[0, 0, 0], [1, 1, 0, 0])
    partial_affine[0, 3] = 40
    with pytest.raises(GeometryError, match="overlap"):
        resample_to_reference(source, Geometry(partial_affine, (4, 3, 2)))


def test_label_resampling_accepts_big_endian_pixels_and_retains_large_labels():
    from nifti2dicom.geometry import resample_to_reference

    source = ImageVolume(
        np.array([[[[0, 300, 17]]]], dtype=">u2"), Geometry(np.eye(4), (3, 1, 1)), kind="seg"
    )
    affine = np.eye(4)
    affine[0, 3] = 0.75
    result = resample_to_reference(source, Geometry(affine, (2, 1, 1)), labels=True)
    np.testing.assert_array_equal(result.data, [[[[300, 17]]]])


def test_disjoint_oblique_slabs_are_rejected_even_when_axis_aligned_boxes_intersect():
    from nifti2dicom.geometry import resample_to_reference

    a = np.sqrt(0.5)
    affine = np.array([[a, -a, 0, 0], [a, a, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    source = ImageVolume(np.ones((1, 1, 1, 8)), Geometry(affine, (8, 1, 1)))
    target = affine.copy()
    target[:3, 3] = [-2 * a, 2 * a, 0]
    with pytest.raises(GeometryError, match="overlap"):
        resample_to_reference(source, Geometry(target, (8, 1, 1)))
