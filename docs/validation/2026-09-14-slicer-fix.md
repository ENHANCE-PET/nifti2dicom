# Slicer PET repair acceptance — 2026-09-14

## Outcome

The known default-import PET failures from the
[baseline audit](2026-09-14-slicer.md) are fixed **with the installed external
integration**. Slicer 5.12.0 now loads both public dynamic acquisitions as
Sequences automatically, preserving every decoded voxel, physical plane and
timepoint. Both acquisitions also pass as double-precision MultiVolumes.
Static PET and binary/multilabel SEG continue to pass through the native readers.

This is an implemented, locally installed receiver repair, not a diagnostic
forced-grouping workaround or an upstream Slicer fix. The converter and exported
DICOM files were not changed. Stock Slicer without the integration still has
the baseline defects; no universal viewer/PACS or clinical-readiness claim is made.

| Public case | Files / timepoints | Default import | Alternate MultiVolume |
| --- | --- | --- | --- |
| Static RIDER PET | 47 / 1 | Pass; 770,048 voxels | Not applicable |
| Dynamic RIDER PET | 1,504 / 32 | Pass; 24,641,536 voxels | Pass; float64, zero added value error |
| Dynamic ACRIN FLT PET | 1,575 / 45 | Pass; 25,804,800 voxels | Pass; float64, zero added value error |

The default runs check **51,216,384 distinct source-grid voxels**. The alternate
representation checks the same dynamic data again; those repetitions are not
counted as additional unique data. All 77 dynamic sequence indices and all 77
MultiVolume display components were exercised.

## What changed

The optional [Slicer integration](../../integrations/slicer/README.md) contains
four focused responsibilities: plugin dispatch, pure PET header validation,
native decoding/assembly, and transactional scene cleanup. It does not import
the converter's geometry or pixel implementation.

- Native `ImageIndex`, `NumberOfSlices` and `NumberOfTimeSlices` define PET time
  blocks, including acquisitions with identical acquisition timestamps.
- Uniform per-timepoint `FrameReferenceTime` supplies actual millisecond browser
  indices. Slice-specific times use the native time-slice count rather than an
  invented mean timestamp; original source-image references remain available.
- Sequences own their decoded frame buffers. MultiVolumes allocate float64,
  preventing an empty uint16 first frame from truncating or wrapping later data.
- Native GDCM decoding is checked against every source plane and all four image
  corners. No acquisition warp is hardened and no resampling is performed.
- Automatic display contrast recovers after empty initial frames. This does not
  change voxel values or claim to provide a clinical PET window/SUV calculation.
- Malformed recognized PET profiles fail with a PET-specific explanation.
  Corrupt late pixel data and cancellation discard the new partial series and
  restore existing nodes, hierarchy items and individual view selections.

See [ADR 0003](../adr/0003-slicer-pet-integration.md) for the integration boundary.

## Evidence and independence

All five public runs started the ordinary installed application without
`--testing`, `--additional-module-path`, `--ignore-slicerrc`, forced frame-time
grouping, or the legacy-examination diagnostic modes. DICOM selection used the
native highest-confidence selector and loadable table, as in the normal browser.
Each run used a fresh temporary DICOM database and restored the previous one.

Expected arrays, physical grids, frame times and durations came from the
independent original-DICOM reconstruction used by the baseline audit. Tests map
every loaded voxel through IJK → RAS → LPS into that original grid, requiring
one-to-one coverage. Exported DICOM pixel/rescale values are decoded separately
and compared by source SOP Instance UID. Across all five runs:

- Maximum added viewer value error: **0 BQML**.
- Original-source quantization-bound violations: **0**.
- Maximum physical-grid residual: **9.5367426 × 10⁻⁷ mm**.
- Spatial or temporal reversal detected in the tested data: **none**.

Browser checks retain an immutable pixel checksum, affine copy and source UID
string before navigation. The displayed proxy must match all three afterward;
two aliased live buffers cannot manufacture a passing result. Real millisecond
index values are checked against the independent timing array. First/middle/last
screenshots and automatic-window state are recorded. Separately named windowed
presentation controls are not used as numerical or timing evidence.

The machine-readable [follow-up report](2026-09-14-slicer-fix.json) records each
timepoint's array checksum, source references count, timing, dtype, selected
reader, aggregate errors, script/report hashes, and screenshot hashes.
The checked-in [public acceptance harness](../../integrations/slicer/tests/check_public.py)
replaces the temporary diagnostic-only harness for future runs.

## SEG and standards validation

The native SEG plugin and bundled dcmqi **v1.5.4**, revision **a102298**, reconstructed:

| Fixture | NIfTI label IDs | Foreground voxels | Differences |
| --- | --- | --- | --- |
| Binary sphere | 1 | 3,267 | 0 |
| Multilabel sphere / box / asymmetric shape | 5 / 300 / 1024 | 3,267 / 864 / 480 | 0 |

Each SEG retained all 47 expected source-image references. These are authored
QA masks, not fresh LION/MOOSE inference in this run. Fresh model inference and
independent decoding remain documented in the
[LMU real-data report](2026-09-12-lmu-real-data.md).

All **3,126 PET output hashes** still match their prior successful DICOM 2026c
validator reports. No IOD validator was rerun for this receiver-only fix;
specifically, this is **not a new `dciodvfy` run**. The
[PET precision report](2026-09-14-pet-precision.md) remains the encoding and
current-standard validation evidence for these unchanged files.

## Regression and installation checks

- **730** package tests pass; converter source digest unchanged.
- **109** pure PET parser tests pass and are now included in package CI.
- **15** native Slicer tests pass against the installed copy on ordinary startup:
  default dispatch, constant timestamps, slice-specific timing, signed values,
  oblique axes, precision-safe MultiVolumes, empty-first-frame display, invalid
  metadata, unreadable/lazily decoded headers, corrupt late data, cancellation,
  scene preservation, scalar PET, and legacy non-PET sequence delegation.
- Ruff lint/format, package mypy and source/wheel build pass.
- The original seven native regressions failed before implementation. Added
  axis, view-state and header-error regressions also exposed failures before
  their corresponding fixes. Independent review prompted these safeguards.

The exact tested module is installed at:

```text
/Users/nutellabear/Library/Application Support/nifti2dicom/slicer/e9cd273542cc8d6a/
```

Its files match the repository source byte for byte. The local Slicer build was
compiled to keep revision settings inside the app directory, where writes fail.
Instead, the previously absent `/Users/nutellabear/.slicerrc.py` now registers the
external module through Slicer's supported module factory on startup, guarded
to the validated 5.12.0 version. Existing module paths and the DICOM database
setting are preserved. To remove this installation, remove its startup block
and restart; there is no app patch to undo.

That fallback does **not** repair unrelated extension/settings write warnings.
Strict app-signature verification already failed before registration because
of extra runtime/settings files. No bundled reader source or signature was
edited or re-signed. The tested source and bootstrap hashes, original settings
backup and startup verification are recorded in the JSON report. Slicer's
[revision-settings implementation](https://github.com/Slicer/Slicer/blob/v5.12.0/Base/QTCore/qSlicerCoreApplication.cxx)
and [module registration example](https://github.com/Slicer/Slicer/blob/v5.12.0/Modules/Scripted/ExtensionWizard/ExtensionWizard.py)
document the underlying mechanisms.

Raw logs, reports and screenshots remain under
`/private/tmp/nifti2dicom-slicer-fix.nilYDL/`. This is a local temporary evidence
directory, not a hosted dataset archive. The retained JSON report and repository
tests are durable; large image/truth fixtures are not bundled with the package.

## Scope limits

These public acquisitions are three GE classic-PET series. Gated/enhanced PET,
irregular grids, vendor-private temporal recovery, non-GE clinical acceptance,
and other viewers/PACS remain outside this evidence. Loading is in memory, not
streaming. Passing these tests closes the identified Slicer import defects for
the tested profile; it does not certify every possible acquisition or workflow.
