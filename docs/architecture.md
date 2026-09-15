# Conversion architecture

The public API and CLI share one pipeline: inspect inputs, resolve meaning and
geometry, encode into a staging directory, validate output, then publish it.
The library is quiet by default. Progress callbacks and structured results are
the only communication from conversion code to presentation code.

## Responsibilities

| Module | Owns |
| --- | --- |
| `api.py` | Stable `convert` and `inspect` interfaces |
| `pipeline.py` | Conversion orchestration and output publication |
| `models.py` | Geometry, image, reference, progress and result contracts |
| `inference.py` | Explicit and inferred image/SEG/RGB meaning |
| `geometry.py` | Physical coordinates and resampling |
| `readers/` | Validated NIfTI images and coherent DICOM reference series |
| `pixels.py` | Quantization with measurable reconstruction error |
| `writers/` | DICOM image, RGB and segmentation encoding |
| `errors.py` | Stable error codes, explanations and recovery guidance |
| `cli/` | Commands and LION-inspired presentation |

Legacy entry points delegate to this pipeline; they must not implement their
own orientation, resampling, encoding, or UID policies.

## Invariants

- Geometry uses LPS millimeters; the affine maps `(x, y, z)` voxel indices.
- Scalar arrays have shape `(time, slice, row, column)`. RGB arrays add a final
  channel dimension. 2D has one slice and one timepoint; 3D has one timepoint.
- Temporal axes survive until encoding. Missing timing is never fabricated.
- A DICOM series has a coherent patient, study, frame of reference and geometry.
- Image conversion preserves the NIfTI grid by default. Reference resampling
  requires an explicit choice; SEG uses the reference grid.
- Integer intensities alone do not imply segmentation. Three volumes alone do
  not imply color. Explicit options take precedence over metadata inference.
- Pixel bytes, bit depth, signedness, rescale parameters and transfer syntax
  are generated together. Silent clipping and wrapping are forbidden.
- CT/MR retain one scale across the image volume. PET is quantized per output
  plane so bright slices/timepoints cannot coarsen another plane's scale.
  Numerical encoding stays in `pixels.py`; writers select its input extent.
- Derived series and instances receive fresh UIDs. Metadata copying is scoped.
- Output is validated in staging. Failure never publishes a partial series or
  destroys an existing output. Input/output overlap is rejected.
- Known problems have a stable code, human explanation and recovery hint.

## Supported scope

Reference-based conversion of scalar 2D/3D/4D NIfTI to classic CT, MR, and PET
image series; RGB Secondary Capture; 3D multilabel SEG. Explicit RGB accepts
channel-last arrays as well as NIfTI RGB datatype. 4D scalar output carries
temporal identifiers; timing is included when known. Enhanced multiframe input
and output, diffusion metadata reconstruction, arbitrary vector/tensor inputs,
and 4D SEG require explicit future profiles and currently fail clearly.

Classic MR quantitative output and 4D CT use documented standard extensions.
PET requires matching frames to preserve trusted timing and decay metadata.
See [DICOM profiles](dicom-conformance.md) for exact attributes and validation
evidence. External checks and target-viewer acceptance remain release gates.

NIfTI cannot establish patient identity or reconstruct missing acquisition
metadata. Reference selection must be unambiguous. Unknown spatial units use
millimeters with a recorded warning, following common NIfTI practice.

Rounded DICOM positions are checked against an endpoint-anchored regular grid.
The precision-dependent tolerance is capped at 0.005 mm per plane; it does not
permit accumulated drift. Original source positions and instance identities
are retained. This conservative policy does not accept every possible rounded
regular stack, particularly when rounded endpoints bias the fitted grid.

Image-writer frame matching is a separate check against that validated grid.
It requires exact permuted dimensions and discrete flip offsets. Physical
corner agreement may account for NIfTI-1 affine roundoff, but is capped at
0.001 mm regardless of coordinate magnitude. PET acquisition timing remains
conditional on both this correspondence and coherent temporal metadata.

For exact MR reorientation that mixes acquisition planes, the writer can retain
explicitly allowlisted, volume-uniform acquisition facts. Temporal identifiers
and declared volume counts must agree; conflicting facts are omitted with clear
warnings. Series facts are assigned once, while conditional image facts follow
the output profile. See [the reference recovery decision](adr/0004-reference-recovery-and-volume-facts.md).

The reader can recover a consistently reversed native PET slice index only
when complete indexing and independent increasing frame times make recovery
unambiguous. It reports the source nonconformance, retains original identities,
and leaves the writer responsible for conformant output indices.

Image and SEG writers share basic copied-code completeness checks in
`writers/codes.py`. Each writer owns an explicit optional-field allowlist:
PET tracer codes within isotope information, or SEG study codes copied by
highdicom. Incomplete sequences are omitted from output copies with warnings
returned to the pipeline. Required fields and surrounding metadata are not
recursively stripped. Missing codes are never guessed; references are untouched.

## Verification

Tests use synthetic asymmetric landmarks and independently calculated physical
coordinates, decode written pixel data, reconstruct real intensities, inspect
temporal tags and SEG source references, and exercise the CLI and transactional
failure paths. Viewer and external IOD validation supplement these tests;
passing pydicom reads alone is not a conformance claim.

Opt-in public-data tools live in `validation/`; source images remain outside
the repository. The pinned IDC manifest records attribution and licenses.
`idc_roundtrip.py` reconstructs every source voxel independently of converter
geometry/grouping helpers. Its own regressions corrupt actual DICOM pixels and
metadata to verify detection. `dicom_conformance.py` retains raw results from
both validators, including error diagnostics emitted with a zero exit status.

PET precision tests combine fractional asymmetric landmarks, independently
computed per-plane error bounds and all 48 signed spatial axis permutations
in 3D/4D. Integer planes fitting the selected representation remain exact beside
floating-point planes. PET signedness is constant across the series, including
when negative values first occur in a later timepoint.
Numeric underflow must raise a clear error, and a failure in a late plane must
leave an existing output untouched. See [the scaling decision](adr/0002-pet-instance-scaling.md).
