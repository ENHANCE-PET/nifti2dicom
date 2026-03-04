"""SimpleITK resampling utility."""

from __future__ import annotations

import SimpleITK as sitk

_INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
}


def resample_image(
    image: sitk.Image,
    interpolation: str = "linear",
    output_spacing: tuple[float, ...] | None = None,
    output_size: tuple[int, ...] | None = None,
) -> sitk.Image:
    """Resample a SimpleITK image to new spacing/size.

    Parameters
    ----------
    image : sitk.Image
        Input image.
    interpolation : str
        One of ``"nearest"``, ``"linear"``, ``"bspline"``.
    output_spacing : tuple[float, ...], optional
        Target spacing. If *None*, uses the input image spacing.
    output_size : tuple[int, ...], optional
        Target size. If *None*, computed from spacing ratio.

    Returns
    -------
    sitk.Image
        Resampled image.

    Raises
    ------
    ValueError
        If *interpolation* is not recognized.
    """
    interp = _INTERPOLATORS.get(interpolation)
    if interp is None:
        raise ValueError(
            f"Unknown interpolation '{interpolation}'. "
            f"Choose from: {', '.join(_INTERPOLATORS)}"
        )

    if output_spacing is None:
        output_spacing = image.GetSpacing()

    if output_size is None:
        in_size = image.GetSize()
        in_spacing = image.GetSpacing()
        output_size = tuple(
            round(in_size[i] * (in_spacing[i] / output_spacing[i]))
            for i in range(len(in_size))
        )

    return sitk.Resample(
        image,
        output_size,
        sitk.Transform(),
        interp,
        image.GetOrigin(),
        output_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelIDValue(),
    )
