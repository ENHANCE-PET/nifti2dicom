# PET metadata fix and revalidation — 2026-09-13

The incomplete optional tracer-code defect identified by the
[initial IDC PET audit](2026-09-13-idc-pet.md) is fixed. All **3,126 regenerated
PET instances** pass unmodified dicom-validator 0.9.0 against DICOM 2026c with
zero errors. The earlier failed outputs and validation reports remain intact.

## Narrow implementation

`writers/codes.py` now holds the shared basic completeness check for copied
optional codes. Each writer supplies its own explicit field allowlist:

- PET: RadiopharmaceuticalCodeSequence within each copied isotope item.
- SEG: the three previously handled optional General Study code sequences.

The PET tracer-code sequence is optional under
[PS3.3 C.8.9.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.2.html).
An included coded item needs one nonblank code-value form and a meaning;
short/long codes also need a coding scheme, while URN codes do not. See the
[basic Code Sequence Macro](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_8.8.html).
The helper rejects blank or conflicting basic fields. It does not validate
terminology membership, all VR/value-form constraints, or enhanced code macros.

Cleanup acts on output copies, never reference datasets. It omits the affected
optional sequence with a warning, retaining the enclosing isotope information,
all other fields and complete codes. It does not recursively remove unknown
sequences or repair required metadata by deleting it. Empty sequences are
retained. The existing public conversion API and writer return types are unchanged.

The image writer's optional warning callback lets the pipeline record one
deduplicated omission warning in ConversionResult and conversion.json. The
library remains quiet; presentation stays with the CLI/caller.

## Test-first regression evidence

Before implementation, 16 malformed-code cases failed and all six complete-code
cases passed. These include the real empty-item and missing-code-value shapes,
blank fields, conflicting value forms, valid short/long/URN codes, and defects
present only in the final frame. Expectations assert serialized isotope fields,
decoded pixels, timing, unchanged source files, report warnings and a quiet API.

A separate direct-writer test checks multiple isotope items, in-memory source
immutability and one warning callback across repeated affected frames.

After the fix:

- Full suite: **620 passed**, including the 293 spatial/temporal orientation cases.
- Ruff lint and formatting: passed.
- Mypy: passed across 33 source files.
- Wheel and source-distribution builds: passed; the wheel contains the shared helper.
- Independent read-only code review: no findings in the bounded change.

## Real-data revalidation

The exact prior public source selection and reviewed audit scripts were reused.
New outputs were written separately; source DICOM files were not edited to
obtain a pass. No validator or standard definitions were patched.

| Case | Temporal volumes | Files checked | Previous IOD errors | New IOD errors | Instances with optional code omitted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static RIDER | 1 | 47 | 94 | 0 | 47 |
| Dynamic RIDER | 32 | 1,504 | 0 | 0 | 0 |
| Dynamic FLT | 45 | 1,575 | 1,575 | 0 | 1,575 |

The independent physical-coordinate audit rechecked all **51,216,384 voxels**,
complete temporal/spatial coverage, each frame's timing and applicable decay
factor, deliberate spatial/time reversal sensitivity, and rejection of wrong
uniform NIfTI timing without output publication. All results matched the prior
audit exactly, including its measured quantization error.

An additional source-to-output metadata comparison checked every instance:

- Pixel bytes, rescale values, geometry and temporal/decay metadata are identical
  to the corresponding previous output.
- All other isotope metadata equals the original source sequence exactly.
- Complete tracer codes survive unchanged; the two malformed-code cases each
  produce one omission warning in the manifest.
- Original source hashes, NIfTI fixture hashes and prior output hashes still match.

Conversion: Python 3.13.3, numpy 2.4.2, nibabel 5.4.0, pydicom 3.0.1.
External validation: separate Python 3.12.11, pydicom 3.0.2,
dicom-validator 0.9.0, unmodified DICOM 2026c definitions.

`dciodvfy` was not run on these regenerated outputs. No target PACS/viewer or
clinical acceptance is claimed by the current-standard validator result.

## Separate finding: temporal-volume precision

The encoding policy was deliberately unchanged by this metadata fix. A separate
diagnostic compared decoded output activity against original DICOM values for
each temporal volume, over the **entire spatial grid including background**.
This is a whole-volume time–activity curve, not an anatomical or tumor ROI.

| Case | Maximum absolute whole-volume mean bias | Maximum relative L1 error |
| --- | ---: | ---: |
| Static RIDER | 0.00013% | 0.00324% |
| Dynamic RIDER | 0.04475% | 0.38898% |
| Dynamic FLT | 2.14136% | 8.69238% |

Mean bias is `(output sum − source sum) / source sum × 100` for a nonzero
source sum. Relative L1 error is `sum(abs(output − source)) / sum(abs(source))
× 100`; it is not a per-voxel relative error or clinical accuracy estimate.

The FLT fourth temporal volume (zero-based index 3) has source mean
13.74519246 BQML and output mean 13.45085806 BQML, a −2.14136% bias. The
volume-wide scale permits a half-step error of 7.84083035 BQML in every frame.
No entire nonzero temporal volume was rounded to zero, but some low-valued
voxels were. These measurements justify evaluating per-timepoint/per-instance
scaling and explicit precision limits next. They do not establish a clinical
tolerance, and no encoding-policy change has been made yet.

## Artifacts and remaining work

The [machine-readable summary](2026-09-13-pet-metadata-fix.json) records source
selection, output/source aggregate hashes, package and wheel hashes, validator
provenance, warning counts and precision metrics. Package-source hash covers
the conversion code; subsequent documentation-only edits do not change it.

Local QA artifacts are under:

```text
/private/tmp/nifti2dicom-pet-codes-r1.fHombk/
```

This directory contains the regenerated outputs, NIfTI fixtures, metadata-diff
and per-timepoint precision reports, validator reports and build artifacts.
Its source-data and original-audit-script links refer to the previous public QA
directory. No medical-image data is committed to the repository.

Still required before release: a quantitative PET precision policy and tests,
target-viewer/PACS acceptance, genuine 4D MR/CT and broader scanner coverage,
and automated external validation. Upstream spatial/temporal provenance limits
described in the original audit remain unchanged.
