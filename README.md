# nifti2dicom

Convert NIfTI images and masks into DICOM with a reference scan.

- Scalar **2D, 3D and 4D** images → classic CT, MR or PET series.
- **Binary and multilabel** masks → one multi-segment DICOM SEG.
- **2D/3D RGB** images → Secondary Capture.
- Physical-coordinate geometry, quantitative pixel encoding, structured errors,
  and output validation before publication.

## Quick start

```bash
pip install nifti2dicom
nifti2dicom scan.nii.gz --reference ./dicom
```

Output goes to `scan_dicom/` beside the input. It contains DICOM files and a
`conversion.json` report with warnings, geometry and label mapping. Supply
`-o ./converted` to choose another location.

The NIfTI's dimensions, orientation and spacing are read automatically. The
reference provides patient/study and modality metadata; it does not need the
same dimensions or number of slices for CT/MR. PET requires matching frames
to preserve acquisition timing and correction metadata. A reference may be a file or a directory
containing a single series, including nested folders.

```bash
# Inspect inferred meaning and reference geometry without creating output
nifti2dicom inspect scan.nii.gz --reference ./dicom

# Explicitly resample into the reference's physical grid
nifti2dicom scan.nii.gz --reference ./dicom --geometry reference

# Select a series when a folder contains several
nifti2dicom scan.nii.gz --reference ./dicom --series-uid 1.2.3.4

# Machine-readable result or error; no progress text on stdout
nifti2dicom scan.nii.gz --reference ./dicom --json
```

## Segmentations

DICOM `BINARY` describes each segment's pixels, **not the number of segments**.
Both a binary mask and a multilabel mask can become a BINARY SEG containing
multiple segments. The encoder uses highdicom and verifies the written masks
against their source DICOM frames.

```bash
# A manually drawn binary mask
nifti2dicom mask.nii.gz --reference ./dicom --kind seg --algorithm-type manual

# Multilabel output from a model
nifti2dicom organs.nii.gz --reference ./dicom --labels labels.json \
  --algorithm-type automatic --algorithm-name MOOSE --algorithm-version 3.0
```

Values must be nonnegative integers: 0 is background. Values such as 5 and 300
are supported without an 8-bit cast. The conversion report records the mapping
from input label values to consecutive DICOM segment numbers.

A simple label mapping:

```json
{
  "5": "Lesion",
  "300": "Liver"
}
```

For semantic interoperability, include coded category and type descriptions.
Names alone cannot establish anatomy; missing codes use an explicitly
unspecified local code and produce a warning. Supply real codes from your
segmentation schema rather than inferring them from pixel values.

MOOSE's nested `organ_indices` format and its `SNOMED.ID`/`SNOMED.name` entries
are accepted. A richer file can also carry algorithm provenance:

```json
{
  "algorithm": {"type": "automatic", "name": "MyModel", "version": "1.0"},
  "labels": {
    "300": {
      "name": "Liver",
      "category": {
        "value": "123037004", "scheme": "SCT", "meaning": "Anatomical Structure"
      },
      "type": {"value": "10200004", "scheme": "SCT", "meaning": "Liver"}
    }
  }
}
```

SEG always aligns to the reference grid using nearest-neighbor interpolation.
Foreground outside that grid, disappearance of a segment during resampling,
and an entirely empty mask are reported clearly. Masks are never silently
thresholded. Automatic and semiautomatic masks require their creator's name
and version; conversion software is not treated as the segmentation algorithm.

Separate overlapping masks, probability/FRACTIONAL output, newer LABELMAP SEG,
and time-varying SEG are not yet supported. Do not collapse overlapping masks
into one integer map: that would discard overlap.

See [highdicom's SEG guide](https://highdicom.readthedocs.io/en/latest/seg.html)
and the [DICOM segmentation module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html).

## RGB

```bash
nifti2dicom color.nii.gz --reference ./dicom --kind rgb
```

Explicit RGB supports arrays shaped `(X,Y,3)` or `(X,Y,Z,3)`. Channels must be
8-bit integer values from 0 to 255. NIfTI's structured RGB datatype is inferred
automatically. An ordinary fourth axis of length three is treated as three
scalar volumes unless RGB is explicitly selected.

## Python API

```python
from nifti2dicom import convert, inspect, ConversionError

info = inspect("scan.nii.gz", "./dicom")
try:
    result = convert("scan.nii.gz", "./dicom", "./converted")
except ConversionError as error:
    print(error.message)
    print(error.hint)
    print(error.code)  # Stable identifier for automation
else:
    print(result.files)     # Only the DICOM files
    print(result.warnings)  # Assumptions and conversion notices
```

Python calls print nothing. Pass `on_progress=callback` to receive typed progress
events. `result.to_dict()` is JSON-ready.

## Automatic decisions and limits

- Native geometry is preserved by default; resampling is explicit.
- Reference stacks tolerate scanner coordinate rounding within at most 0.005 mm
  of the grid spanning their first and last planes, with a warning above 0.001 mm.
  Every plane is checked; uneven spacing and accumulated drift still fail.
- Frame matching separately allows bounded NIfTI affine roundoff: exact index
  correspondence and all eight physical corners must agree, with a 0.001 mm
  absolute cap. A genuinely shifted or stretched PET grid remains an error.
- Spatial units are converted to millimeters. Unknown units assume millimeters
  and are recorded as a warning.
- Label descriptions or NIfTI label intent select SEG. Integer values alone do not.
- Multiple reference series require `--series-uid`; patient/study mismatches fail.
- 4D images retain timepoint identities. Timing comes from explicit NIfTI time
  units or matching reference frames, never from a made-up acquisition duration.
  Input time order is preserved; a NIfTI without per-frame provenance cannot
  reveal whether an upstream tool already scrambled its volumes.
- A complete, consistently reversed native PET slice index can be recovered
  with an explicit warning when independent frame timing proves the ordering.
  Mixed/missing indices and contradictory timing still fail; source files are
  never rewritten. MR exact reorientation preserves volume-uniform acquisition
  times even when the output slice planes change. Conflicting plane facts are
  omitted with warnings, not copied onto unrelated planes.
- Reference CT/MR/PT acquisition fields must be available where required by the
  selected output profile. PET values must already use the reference's units:
  SUV and activity concentration cannot be distinguished from a NIfTI array alone.
- Classic MR uses a standard-extended rescale mapping; 4D CT uses temporal
  extensions. The receiving viewer must support these attributes. See the
  [DICOM output profiles and validation evidence](docs/dicom-conformance.md).
- Sheared/singular geometry, complex/vector/tensor data, enhanced/multiframe
  references, temporal RGB, and gated/reprojection PET fail explicitly.
- Scalar encoding uses 16-bit integers plus rescale parameters. Integer values
  fitting the selected signed/unsigned representation are exact; other values
  are quantized. Maximum reconstruction error is recorded in DICOM
  `DerivationDescription`. PET scales each image independently
  to preserve low-signal slices/timepoints; consumers must apply each image's
  own rescale tags. CT/MR retain a shared volume-wide scale.
  PET signedness stays constant across the series; any negative value selects
  signed storage, so positive values above 32767 can require quantization.
- Volumes are loaded into memory. Very large 4D datasets need adequate RAM.
- SEG omits incomplete optional study codes; PET omits incomplete optional
  tracer codes within copied isotope metadata. Both use a shared completeness
  check and record a warning. Complete codes, other isotope fields and source
  files are preserved; missing codes are never invented.

Validation includes serialized pixel decoding and SEG label/source-frame round
trips. This is not a blanket DICOM conformance or viewer-compatibility claim:
external IOD validators and target viewers remain part of release acceptance.

See the [real-data validation report](docs/validation/2026-09-12-lmu-real-data.md)
for fresh LION/MOOSE inference, independent SEG decoding, and known limitations.
The [public PET audit](docs/validation/2026-09-13-idc-pet.md) checks genuine
32- and 45-timepoint acquisitions, spatial/time reversal detection, and all
output voxels. Its optional tracer-code defect is now fixed: all 3,126
regenerated files passed current-standard validation. The
[follow-up report](docs/validation/2026-09-13-pet-metadata-fix.md) records that
evidence and PET precision concerns from volume-wide 16-bit scaling. The
[precision follow-up](docs/validation/2026-09-14-pet-precision.md) compares scaling
policies and validates the per-image PET encoding introduced to address them.
The [baseline Slicer audit](docs/validation/2026-09-14-slicer.md) exposed receiver
defects in default dynamic PET import. The optional
[Slicer PET integration](integrations/slicer/README.md) fixes those paths without
changing converter output. The [repair acceptance report](docs/validation/2026-09-14-slicer-fix.md)
passes normal installed import for all 51.2 million public PET voxels, both
dynamic representations, and binary/multilabel SEG. Stock Slicer without the
integration still has the baseline defects; compatibility is limited to the
tested installation and profiles.
The [expanded IDC repair acceptance](docs/validation/2026-09-15-idc-repair.md)
adds Philips/Siemens static and 45-frame PET, sagittal 25-frame MR, and ten
separate CT phases: 291 million source voxels, all 10,910 output files checked
with dciodvfy, and strict Slicer checks of both dynamic representations. It
records the PET ordering/MR metadata fixes and remaining profile limitations.

## Errors and output safety

Expected failures explain what happened and how to resolve it. Use `--debug`
for tracebacks on unexpected failures, `--quiet` to suppress normal output, or
`--json` for structured automation.

An existing output is an error. With `--overwrite`, old output remains in place
until the new conversion passes validation. A failed restore preserves a backup
and reports its path. Input/output overlap is rejected. Interruptions clean up
staging files; a hard process kill may leave a named lock directory.

## Upgrading from earlier versions

Existing `convert`, `segment`, `rgb` and `resample` subcommands still work.
Python conversion functions and PUMA wrappers delegate to the same pipeline.

Intentional changes:

- Different image/reference dimensions are supported.
- Existing output errors instead of silently skipping.
- Vendor-specific flips are gone; the deprecated vendor argument is ignored.
- SEG requires truthful algorithm provenance, including via the labels JSON.
- Pixel overflow errors or quantitative rescaling replace silent clipping.
- Generated filenames are stable `IM_000001.dcm` sequences, not template filenames.
- Use `result.files` or `*.dcm`; the output also includes a JSON report.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check nifti2dicom tests integrations/slicer validation
ruff format --check nifti2dicom tests integrations/slicer validation
python -m mypy nifti2dicom --ignore-missing-imports
python -m build
python -m validation.check_distribution dist/*.tar.gz dist/*.whl
```

Start with [architecture](docs/architecture.md) and [AGENTS.md](AGENTS.md).
Tests use asymmetric synthetic data to verify physical coordinates, quantitative
values, temporal frames, segmentation references and failure recovery.
Optional reproducible public-data checks are described in
[validation/README.md](validation/README.md); downloads and viewer tests are
not dependencies of ordinary unit tests or package installation.

MIT licensed. Authors: Lalith Kumar Shiyam Sundar, Aaron Selfridge and Siqi Li.
