# Real-data acceptance checks — 2026-09-12

These checks evaluate conversion fidelity and DICOM structure, not model
accuracy, clinical safety, or acceptance by a particular PACS/viewer. Patient
files and identifying paths stayed on LMU infrastructure. Case aliases below
are local QA labels, not patient identifiers.

## Scope and isolation

Fresh LION and MOOSE inference ran in allocated Slurm GPU jobs on `lmu-core`.
The existing cached models/environments were reused after cache-version checks;
inputs were copied into private staging. Conversion used an isolated Python
3.12 environment. Original DICOM, NIfTI and model files were not modified.
Input content hashes were compared before and after conversion.

Private artifacts, scripts, logs and input manifests are under:

```text
/data2/core-rad-digitx/projects/nifti2dicom-validation-20260912.B3giAv
```

Do not commit that directory or its private manifests. The CT source's
deidentification was not independently established. The PET source came from
the existing public Lung_Dx project on the server.

## Fresh model outputs

| Case | Model | Result | Source planes | Foreground voxels |
| --- | --- | --- | ---: | ---: |
| `lion001_binary_r4` | LIONZ 1.0.5, FDG Dataset789_Tumors, model 27032026 | 1 tumor segment | 419 | 18,294 |
| `moose001_multilabel_r4` | MOOSEZ 3.2.2, clin_ct_PUMA | 18 present anatomy segments | 140 | 16,883,774 |

LION inference completed in job 4231833; MOOSE completed in job 4231842.
MOOSE's model contains 23 foreground classes; absent classes are omitted with
a warning. Present source IDs were `1,2,4,5,6,7,8,9,11,12,13,15,16,17,19,21,22,23`,
mapped to consecutive DICOM segment numbers without changing their identities.

The installed PUMA schema supplied names but no semantic codes. Those names
are retained, with explicit unspecified-code warnings; these tests do not
establish interoperable anatomical coding. LION's segment used supplied SCT
category/type codes. Both record automatic-model provenance, not the converter
as the segmentation algorithm.

For both outputs:

1. Independently resample the original mask in physical space with nearest
   neighbor and reconstruct the SEG by source SOP Instance UID using highdicom.
2. Decode the saved SEG with the separate dcmqi 1.5.7/DCMTK implementation.
3. Reconstruct on the original NIfTI mask grid and compare every voxel.

Both comparisons were exact. dcmqi produced one LION mask and 18 MOOSE masks;
each case had **zero differing voxels**. Final independent jobs: 4231865 and 4231866.
The cached dcmqi/dicom3tools containers matched their SHA256SUMS manifest.

## Image conversion checks

| Case | DICOM files | Timepoints | Decoded voxels | Maximum value error |
| --- | ---: | ---: | ---: | ---: |
| Real CT 3D | 140 | 1 | 46,520,320 | 0 |
| True 2D NIfTI derived from that CT | 1 | 1 | 332,288 | 0 |
| Real CT with constructed temporal axis | 280 | 2 | 93,040,640 | 0 |
| Real PET activity concentration | 419 | 1 | 16,760,000 | 0.55794446 Bq/ml |

Each output pixel's patient-space coordinates were mapped independently through
the original NIfTI affine. Maximum grid discrepancy was below 3e-11 voxels.
The temporal stress test uses the CT and CT+17 with 2.5-second sample spacing.
It verifies conversion mechanics; **it is not a real dynamic acquisition**.

PET input was independently reconstructed with SimpleITK/GDCM. All 16,760,000
values exactly matched original DICOM activity concentration in BQML, and the
physical grid was verified before conversion. Output's maximum value error
was 0.557944457731 Bq/ml, within half its quantization step
(0.557944457771 Bq/ml). All 419 output frames retained the corresponding
FrameReferenceTime, ActualFrameDuration, DecayFactor, Units and DecayCorrection.

The server's dcm2niix-derived test input had approximately 1.2 mm end-to-end
drift; the converter correctly rejected matching its PET timing to those
planes. That negative case was preserved, not accepted by widening tolerance.

## Defects found and regression coverage

- Scanner PET coordinates were serialized at mixed two-/three-decimal
  precision. Neighboring gaps varied from 2.020 to 2.030 mm, although all 419
  planes lie within 0.005 mm of an endpoint-anchored 2.027 mm grid. The reader
  now accepts this bounded rounding, records a warning and still rejects
  nonuniform spacing and accumulated drift. Source positions are untouched.
- Correct NIfTI-1 PET affines also exposed an overly strict fixed voxel-space
  frame-matching tolerance. Matching now requires exact discrete correspondence
  and checks affine coefficient roundoff at all eight physical corners, capped
  at 0.001 mm displacement. Regressions retain rejection of real translations,
  subtle accumulated drift, and excessive tolerance at large coordinate origins.
- CT output lacked the conditional Laterality attribute. Unknown values now
  remain empty; known values are retained only for matching source frames.
  Optional CT spacing/classification metadata is no longer inferred.
- The CT source contained a procedure code without its coding scheme.
  highdicom copied that optional malformed sequence into SEG. The writer now
  omits incomplete optional study codes with a recorded warning; complete
  codes remain unchanged. Tests also cover long codes and URN codes.

## External IOD validation

dicom-validator 0.9.0 used DICOM 2026c in a separate validator environment:

| Output | Current-standard result |
| --- | --- |
| CT 3D | All 140 files: zero errors |
| CT 2D | One file: zero errors |
| PET 3D | All 419 files: zero errors |
| CT temporal stress test | All 280 files: only the three documented temporal extensions |
| LION SEG | Stock parser: six functional-group errors; corrected macro table: zero |
| MOOSE SEG | Stock parser: six functional-group errors; corrected macro table: zero |

The SEG parser defect and in-memory diagnostic correction are described in
[DICOM profiles](../dicom-conformance.md). No DICOM file, installed validator
code or cached standard was modified to obtain those corrected results.
The original MOOSE SEG had one additional real missing-coding-scheme error;
the fixed output does not. Raw results and diagnostic results are both saved.

Final current-standard job: 4231867. The cached dicom3tools validator is from
January 2017. It additionally objects
to current-valid SEG attributes/omissions and the omitted optional CT ImageType
classification. Its nonzero reports remain in the QA evidence; they are not
presented as clean passes. Local-code warnings are also retained.

## Provenance and limits

The final `_r4` outputs used wheel SHA256:

```text
f93df1a454fd43c3473f55bae315190f3137e266a104c898b2e03d5f9308cdd2
```

MOOSE checkpoint SHA256, verified unchanged after inference:

```text
4c650925809e06abd5e3c7a458ed41230333f2e3ca4624e899ebf9b51864619b
```

Runtime: Python 3.12, numpy 2.5.3, pydicom 3.0.2, highdicom 0.28.1,
SimpleITK 2.5.6. Remote reports retain script, package-source and per-file
hashes. Failed setup/validation attempts are preserved rather than overwritten
as successful runs; the successful MOOSE runner includes a multiprocessing guard.

Final conversion jobs: CT 4231860, PET 4231861, LION SEG 4231862, MOOSE SEG
4231863. They completed conversion/pixel checks but intentionally exited
nonzero because their legacy-validator findings are not suppressed. The
separate current-standard reports above distinguish those findings. The
installed-wheel Linux test run (job 4231864) passed all **304 tests**; the local
macOS/Python 3.13 run also passed all 304. Ruff, mypy, package builds and
warning-as-error Sphinx builds passed. Test-first fixes received read-only review.

See the [sanitized machine-readable summary](2026-09-12-lmu-summary.json).
Documentation-only changes after the tested wheel do not change its recorded
package-source digest: `056c40487a36bf68ebd41edb36a3a411f32c5911d3ba9b500d99e89988ccf26e`.

Still required before a clinical release: target-viewer/PACS acceptance,
representative genuine dynamic 4D and MR acquisitions, broader scanner profiles,
and validated semantic label schemas. Probability masks, overlapping-mask
inputs, time-varying SEG and enhanced multiframe profiles remain unsupported.
