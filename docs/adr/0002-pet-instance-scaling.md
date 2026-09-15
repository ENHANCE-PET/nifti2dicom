# Quantize classic PET per image instance

Status: accepted, 2026-09-14.

## Context

One scale over an entire 3D/4D PET scan couples low-signal images to the scan's
largest value. Public FLT testing measured a 2.14% whole-volume mean bias in an
early timepoint despite satisfying the global half-step error bound. Per-timepoint
scaling reduces that coupling across time but retains it between slices.

The [PET Image Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.4.html)
defines RescaleSlope on each image and requires zero RescaleIntercept and 16-bit
storage. A single slope across the series is not our PET export contract.

## Decision

The image writer calls the existing pixel encoder on each output PET plane.
It selects signed storage for the whole series if any value is negative,
otherwise unsigned, as required by
[C.8.9.1.1.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.html).
The encoder honors that storage constraint and owns nearest rounding, decimal-string precision,
overflow checks and measured reconstruction error. The writer serializes the
returned pixels and rescale tags together; DerivationDescription records the
maximum error for that image. Integer planes fitting the selected representation
keep exact values. In mixed-sign PET, positive integers above 32767 may need
quantization to honor both signed storage and the required zero intercept.
All-zero planes are valid. A calculated slope that underflows to zero fails
explicitly instead of defaulting to one and erasing the input.

CT/MR retain their existing shared scale and volume-wide error description.
No public option or second numerical encoder is introduced. Geometry, frame
matching, time ordering, units and decay metadata do not change. A late encoding
failure is handled by the existing staged-publication boundary.

## Consequences and limits

PET readers must apply each instance's own slope with the series' signedness;
stacking raw stored pixels and applying only the first image's slope is wrong.
Per-plane scales reduce the available quantization step for low-signal planes,
but do not guarantee smaller error for every individual voxel, exact arbitrary
floating-point recovery, or preservation of every faint voxel beside a bright
voxel in the same plane. Clinical accuracy and viewer acceptance are separate
from numerical and IOD checks. We do not guess a clinical error threshold.

The [precision validation report](../validation/2026-09-14-pet-precision.md)
compares whole-scan, timepoint and instance scaling on identical public inputs,
then checks serialized output against original DICOM values and world coordinates.
