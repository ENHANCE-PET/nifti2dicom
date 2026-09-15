# IDC repair and independent acceptance — 2026-09-15

Status: supported-profile repair and acceptance complete. All final scalar
audits and strict native-viewer checks pass; raw MR base-IOD extension findings
remain explicitly reported below.

This extends the [candidate audit](2026-09-15-idc-candidates.md) with actual
downloads, fixes, full pixel reconstruction and independent receiver checks.
It is supported-profile engineering evidence, not clinical certification or
a promise that every NIfTI/DICOM acquisition is interpretable.

## What changed

- Siemens dynamic PET: recover only a complete, consistent descending native
  slice index with independently increasing frame times. The source encoding
  is explicitly reported as nonconformant; source files and identities remain
  unchanged. Fresh output uses conformant physical ordering.
- Sagittal dynamic MR: exact reorientation now preserves volume-uniform
  acquisition facts even when output planes differ from acquisition planes.
  Mixed plane facts are omitted with warnings, not assigned arbitrarily.
- MR timing: empty reference TemporalResolution cannot overwrite explicit
  NIfTI spacing; declared temporal completeness is checked for one remaining
  volume as well as multiple volumes.
- Conditional metadata: MR TriggerTime follows actual heart-gating ScanOptions.
  Laterality and CT/MR PatientPosition are series-uniform or empty with
  appropriate warnings. Proven, uniform CT AXIAL/LOCALIZER classification is
  preserved only when acquisition planes match.
- Packaging: local Slicer SQL databases and internal work notes no longer
  enter distributions. Actual archive inspection runs in CI and before publish.
  Root-level Slicer databases and their sidecars are also ignored by git.

These decisions remain in the reader/writer boundaries, with no vendor-name
branches, source-header rewriting, or new resampling policy. See
[ADR 0004](../adr/0004-reference-recovery-and-volume-facts.md).

## Real-data reconstruction

Downloaded all 6,610 source instances in the fifteen selected IDC v24 series,
approximately 608.34 MB. Source UIDs, licenses, collection attribution and DOIs
are retained in the [pinned manifest](2026-09-15-idc-candidates.json). The data
uses CC BY 3.0 or CC BY 4.0; no authentication or private clinical data was used.

The oracle establishes space directly from source ImagePositionPatient,
ImageOrientationPatient and PixelSpacing. Dynamic PET time groups come from
FrameReferenceTime, not filenames, InstanceNumber or converter grouping code.
MR groups use explicit TemporalPositionIdentifier. Each source image is decoded
with its own scaling before creating the NIfTI input.

Every output voxel is mapped back through world coordinates to the original
source grid. Assertions require complete coverage, no duplicate physical voxels,
correct time identity, preserved acquisition facts, and each output plane's
half-quantization-step bound plus 1e-7 numerical tolerance. Native source-world
positions, not only a fitted affine, are compared. Source files are fingerprinted
before and after conversion.

| Source case | Timepoints | Output files | Voxels checked | Maximum value error | Maximum world error, mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Philips static PET | 1 | 213 | 4,416,768 | 0.185732 | 0.000015224 |
| Siemens static PET | 1 | 42 | 1,360,800 | 0.162534 | 0.000061311 |
| Philips dynamic PET | 45 | 2,025 | 41,990,400 | 4.046539 | 0.000000707 |
| Siemens dynamic PET | 45 | 3,330 | 93,985,920 | 22.684611 | 0.000038507 |
| QIN sagittal dynamic MR | 25 | 4,800 | 18,432,000 | 0.098940 | 0.000007644 |
| Ten separate CT respiratory phases | 1 each | 500 | 131,072,000 | 0, exact | 0.000008370 |

Total: **291,257,888 voxels and 10,910 output images**. PET errors use the source
BQML units; MR uses its source-scaled intensity units. All per-plane bounds pass.
These are measured quantization errors, not a claim of universally lossless PET/MR.

The MR source has twenty sagittal planes per timepoint. Canonical reorientation
produces 192 planes per timepoint while retaining the same physical voxels;
the file-count increase is not interpolation. All 25 acquisition times survive,
including the 69.45-second gap; no uniform TemporalResolution is invented.
The Philips static PET's bed/slice-specific timing and durations also survive.

The ten CT respiratory phases are separate series, with two Frame of Reference
identities and no unambiguous temporal tags. Each passes a 3D conversion. Their
combined reference correctly raises `ambiguous_reference`, identifies all ten
series, and explains explicit selection. No partial output is created; an
existing output is preserved even when overwrite was requested. This is not
presented as a successful inferred 4D CT conversion.

## Independent DICOM conformance

All final files were checked without altering validator code, standard tables,
or input datasets:

- Official dicom3tools `dciodvfy` 1.00.snapshot.20260901071806:
  **zero errors and zero nonzero exits across all 10,910 files**.
- dicom-validator 0.9.0, unmodified DICOM 2026c:
  zero findings in all 6,110 CT/PET files. MR has exactly two unexpected-tag
  findings per image: RescaleIntercept and RescaleSlope, totaling 9,600.
  Those are the existing [declared MR rescale extensions](../dicom-conformance.md),
  not deleted or hidden to make a base-IOD validator green. The strict validator
  command therefore intentionally returns nonzero for the complete set.

Warnings remain visible: deidentified legacy person-name forms, unknown StudyID
or laterality, legacy PET coding designators, and standard-extended MR rescaling.
No identity, anatomy or scanner fact was fabricated to silence them.

The earlier outputs reproduced missing Laterality and forbidden nongated MR
TriggerTime. CT also exposed a dciodvfy ImageType value-3 demand despite zero
process exit status. The archive preserves raw diagnostic counts independently
of exit status. Known CT classification is now retained; unknown classification
is still not guessed just to satisfy a validator. See
[General Series](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.3.html),
[MR Image](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.3.html),
and [CT Image](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.2.html).

## Native Slicer acceptance

The installed Slicer 5.12.0, revision a5df714, uses the previously installed
external PET integration. No application-bundle or receiver production code was
modified in this repair. Normal default DICOM selection is used, not forced
loadables. Each dynamic case is checked as both a Sequence and MultiVolume.

The acceptance code independently checks every loaded pixel against serialized
pydicom values and original source truth, all physical positions, source image
references, time labels, browser proxies and display-component navigation.
Stock MR sequences use ordinal indices linked to elapsed-millisecond metadata;
both parts of that mapping are checked against source acquisition times.

Review hardened the oracle itself: NaN/Inf pixels fail before error arithmetic;
coverage requires exact voxel count and uniqueness; and the loaded dynamic
representation must exactly match the requested one. Each import binds the
actual DICOM file set using pre/post content fingerprints and records the
corresponding roundtrip report hash. All eight final strict reruns pass:

- Two static PET scalar imports, two 45-frame PET sequences, and the 25-frame
  MR sequence: **160,185,888 unique source voxels**, 117 frames/volumes.
- All three dynamic cases also pass as MultiVolumes: an additional
  154,408,320 voxel comparisons. Across eight imports, 314,594,208 comparisons
  cover all 232 loaded frames/volumes.
- Every loaded value equals independent pydicom decoding exactly, every
  per-plane quantization bound passes, and maximum world discrepancy is
  0.000061311 mm. Sequence proxy buffers and MultiVolume display components
  are checked at every timepoint, not just the first or last frame.

Both source-grid truth and final DICOM input fingerprints match the conversion
audits. The installed external integration's five Python files match the
repository and the previously validated installation. Representative final MR
and PET screenshots were also inspected; numerical source-world comparisons,
not visual plausibility alone, establish spatial agreement.

Binary and multilabel SEG checks reuse the existing synthetic masks and source
references: both pass with zero differing mask voxels and all 47 referenced
image instances retained. The accompanying 770,048-voxel static PET and the
32-frame, 24,641,536-voxel RIDER MultiVolume also pass the strict checker.
These test conversion/import fidelity, not segmentation-model accuracy. No
new segmentation inference is claimed for these scalar cases.

## Regression and distribution verification

- **827 package tests** pass on Python 3.13.3.
- **135 pure Slicer tests** pass, including corruption checks for nonfinite
  pixels, duplicate coverage and wrong requested representations.
- **15 native Slicer regressions** pass in the installed application.
- Ruff check/format, mypy across 33 package source files, and `git diff --check`
  pass. Independent review has no remaining Critical or Important findings.
- Source archive and wheel build successfully; archive inspection rejects local
  databases, imaging files and internal work notes without deleting local files.
- The actual wheel installs outside the checkout on Python 3.12.11. Its source
  hashes match the audited converter; CLI help and small independent 2D/3D/4D
  conversions all pass physical-coordinate and quantization checks.

## Reproduce and interpret

Commands and developer tools are in [validation/README.md](../../validation/README.md).
Image data remains outside git, in the explicit IDC cache. The compact
[machine-readable evidence](2026-09-15-idc-repair.json) records source selection,
per-case metrics, raw conformance diagnostic counts, source/output/report/code
fingerprints, every Slicer frame's pixel hash, timing metadata and exact import
provenance. Full local artifacts are under
`/private/tmp/nifti2dicom-idc-gaps.7pu3cy/roundtrip-reviewed`; legacy SEG/viewer
fixtures remain under `/private/tmp/nifti2dicom-slicer-fix.nilYDL`.

The supported scope remains classic scalar image profiles plus the existing
SEG profile. Enhanced/gated PET, overlapping/probabilistic/time-varying SEG,
arbitrary vector/diffusion reconstruction and unproven upstream temporal identity
are not covered by these results. Other PACS/viewer installations still need
their own acceptance tests.
