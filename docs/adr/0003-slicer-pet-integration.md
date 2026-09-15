# 0003: Isolate the Slicer classic PET receiver repair

Status: accepted for the tested Slicer 5.12 integration profile.

## Context

The independent installed-Slicer audit found correct exported PET files but
incorrect default dynamic imports: the legacy importer missed native PET timing,
selected MultiVolume instead of Sequences, cast later frames into the first
frame's uint16 buffer, and retained an empty first frame's display window.
Changing standards-compliant converter scaling to accommodate one receiver
would weaken precision and would not repair its temporal interpretation.

## Decision

Keep an optional, external scripted module under `integrations/slicer/`. Register
an adapter at the existing MultiVolume plugin boundary using Slicer's supported
additional module paths. Do not modify or re-sign the installed app. Keep the
integration independent of the conversion package and its runtime imports.

If the build locates revision settings inside a read-only app directory, use
Slicer's supported user startup script to register/load the same external module
through the module factory. Preserve existing startup code, limit the local
bootstrap to the validated version, and do not silently edit app-bundle settings.
This fallback was required by the tested local installation.

A pure header parser groups classic DYNAMIC\\IMAGE PET using ImageIndex,
NumberOfSlices and NumberOfTimeSlices. This follows the standard's time-slice
outer / slice inner indexing rule, not acquisition timestamps or filenames.
See [DICOM C.8.9.4.1.9](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.4.html).
Uniform per-timepoint FrameReferenceTime becomes the actual millisecond sequence
index. Slice-specific reference times retain a native time-slice count, without
inventing an average timestamp. Source SOP Instance UID references are retained.

The loader uses Slicer's native GDCM scalar decoder for each validated spatial
block, checks the loaded physical grid, and constructs the complete result
before publishing it. Sequences retain independently owned frame buffers;
MultiVolume storage is double precision. Display auto-windowing is independent
of quantitative voxel data. Non-dynamic-PET interpretation remains delegated to
the bundled plugin; static PET and SEG retain their native readers.

## Consequences

- The integration is removable without changing output DICOM or the signed app.
- Missing or inconsistent dynamic PET metadata produces a visible warning and
  failed load, rather than a misleading spatial stack or partial series.
- The supported profile is deliberately narrower than all PET: no gated,
  enhanced, irregular-grid or vendor-private temporal recovery is promised.
- Native Slicer tests supplement pure parser tests and package tests. Public
  acceptance must test default selection and immutable browser-buffer checks;
  diagnostic forced grouping is not acceptance evidence.
- Integration installation must be retested after Slicer upgrades. It is not an
  upstream Slicer fix or a claim about third-party viewers/PACS.
