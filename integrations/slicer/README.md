# Slicer PET integration

An optional receiver-side repair for **Slicer 5.12.0**, tested with its bundled
GDCM, MultiVolume and Sequences modules. It fixes classic dynamic PET import;
it does not change nifti2dicom output or modify Slicer's application files.

## Install

1. Keep this directory in a stable location, or copy `Nifti2DicomPET.py` and the
   `Nifti2DicomPETLib` directory together to one.
2. In Slicer, open **Application Settings → Modules → Additional module paths**,
   add that directory, and restart Slicer. Preserve your existing module paths.
3. Import the DICOM folder normally in the DICOM browser. Dynamic PET defaults
   to a **Volume Sequence**; use the Sequences toolbar to move through time.

Check registration in Slicer's Python console:

```python
print(slicer.modules.dicomPlugins["MultiVolumeImporterPlugin"].__name__)
# Nifti2DicomPETPlugin
```

The existing **DICOM → Preferred multi-volume import format** setting can select
MultiVolume instead. Both representations appear in advanced load options, but
only the preferred representation is selected automatically. Do not manually
override a rejected dynamic PET series to load it as a single scalar volume.

To uninstall, remove only this additional module path and restart. The original
Slicer reader resumes handling PET. Retest before using a different Slicer
version; this is not an upstream Slicer patch or a promise about other viewers.

### If module-path settings cannot be saved

Some Slicer builds store revision-specific settings inside the application
directory. If it is read-only, use Slicer's supported **user startup script**
instead. Preserve any existing startup code; do not edit files inside the app.
For example, add this block to the user's `.slicerrc.py`, adjusting the path:

```python
def _register_pet_integration():
    import logging
    import qt
    import slicer

    if slicer.app.applicationVersion != "5.12.0":
        logging.warning("PET integration skipped: this Slicer version has not been validated.")
        return
    factory = slicer.app.moduleManager().factoryManager()
    factory.registerModule(qt.QFileInfo("/absolute/path/to/Nifti2DicomPET.py"))
    if not factory.loadModules(["Nifti2DicomPET"]):
        logging.error("PET integration could not load. Check its path and the Slicer error log.")


_register_pet_integration()
del _register_pet_integration
```

Restart and check registration as above. Remove only this block to uninstall.
`--testing` and `--ignore-slicerrc` intentionally bypass startup scripts. This
fallback does not repair unrelated extension-manager/settings permissions.

## What is preserved

- Native PET `ImageIndex` time blocks, even with identical acquisition times or
  shuffled files/InstanceNumber. No guessed temporal grouping.
- Actual `FrameReferenceTime` values in milliseconds when uniform within each
  timepoint. If times differ by slice, the browser uses the native time-slice
  count instead of inventing a common timestamp. Source image references retain
  access to the original per-image timing and durations.
- Per-image rescaling, including fractions, signed values and values above
  65535. Sequences own their frame buffers; MultiVolume uses float64 storage.
- Native physical coordinates and source SOP Instance UID references. Both
  image axes and every slice plane are checked after decoding; no resampling.
- Automatic display contrast after empty initial frames. This is a display
  choice, not SUV conversion, normalization or a clinical window preset.

The accepted profile is classic PET Image Storage, `DYNAMIC\IMAGE`, consistent
16-bit scalar storage, a complete native index range and a regular spatial grid
shared across time. Gated/enhanced PET, vendor-private timing recovery and
irregular grids are not covered. Static PET and SEG keep Slicer's native readers;
other legacy multi-volume inputs are delegated to the bundled plugin.

Invalid recognized dynamic PET receives a specific warning during examination
and fails loading. Damaged pixel data and cancellation discard the new partial
series and restore the existing scene's per-view selections. Error details are
available in Slicer's error log. The native scalar examiner may also log an
irregular-stack warning while competing interpretations are examined; the
selected PET sequence is validated independently.

This integration loads the acquisition into memory. Double-precision
MultiVolume and deep-copied Sequences can require substantially more RAM than
the original 16-bit DICOM files. This is not a streaming or out-of-core reader.

## Code map

- `Nifti2DicomPET.py`: registration, profile ownership and DICOM plugin dispatch.
- `Nifti2DicomPETLib/frames.py`: pure header validation and immutable frame groups.
- `Nifti2DicomPETLib/loader.py`: native decoding, grid checks and result assembly.
- `Nifti2DicomPETLib/scene.py`: scoped rollback of reader-created nodes/items and
  view changes. GUI events are processed only between these synchronous scopes.
- `tests/fixtures.py`: independently authored DICOM data.
- `tests/test_frames.py`: pure tests, also run in package CI.
- `tests/check_slicer.py`: native import, precision, timing, geometry and failure
  regressions. Run inside Slicer, not the package Python interpreter.
- `tests/check_public.py`: default-import all-voxel acceptance against independent
  public PET/MR truth arrays; includes native binary/multilabel SEG import.
- `tests/public_oracle.py`: pure acceptance assertions for finite values, exact
  physical coverage, timing and requested representation, with corruption tests
  in `tests/test_public_oracle.py`.

## Verification

From the repository:

```sh
.venv/bin/python -m pytest integrations/slicer/tests
.venv/bin/ruff check integrations/slicer
.venv/bin/ruff format --check integrations/slicer
```

Use absolute paths and a temporary working directory for native tests:

```sh
/Applications/Slicer.app/Contents/MacOS/Slicer --testing --no-splash \
  --additional-module-path /absolute/path/to/integrations/slicer \
  --python-script /absolute/path/to/integrations/slicer/tests/check_slicer.py
```

Public acceptance requires a prepared fixture directory containing `cases.json`.
Each case names its exported `dicom` directory, independent source `truth` NPZ
and `truth_sha256`, `expected_timepoints` and output `expected_slices`. Optional
`modality` defaults to `PT`; the current checker also accepts `MR`. The truth
contains `values` (TZYX) and `affine_lps`, plus:

- PET: `frame_reference_times_ms` and `frame_durations_ms` (T×Z) preserve
  bed/slice-specific timing. Legacy `times_ms` and `durations_ms` (T) are accepted
  when timing is uniform over each volume.
- MR: `acquisition_times` contains the original DICOM TM string for each
  timepoint. Ordinal sequence indices and elapsed-time metadata are checked
  together; irregular intervals are not replaced by a uniform clock.

Static cases default to testing binary and multilabel SEG. Supply `binary-seg/`,
`multilabel-seg/` and `seg-truth.npz`, or set the case's `seg_kinds` to `[]` when
testing scalar conversion only. These inputs are not bundled with the package.
The [IDC roundtrip tool](../../validation/README.md) creates scalar fixtures.

Set `SLICER_QA_ROOT` to this directory, `SLICER_QA_CASE` to its case alias, and a
new `SLICER_QA_RUN` for each run. Execute `check_public.py` inside a normal Slicer
startup with the integration registered; omit `--testing` when registration
uses `.slicerrc.py`. Optional `SLICER_QA_FORMAT=multivolume` tests the alternate
representation. Never run concurrent sessions against shared temporary settings.

Acceptance requires finite values, complete unique physical voxel coverage,
exact serialized-value agreement, source quantization bounds, and the requested
dynamic representation. Reports fingerprint the checker, oracle, case manifest,
and actual DICOM file set before and after import. When present, the adjacent
`roundtrip.json` is fingerprinted too, binding viewer results to the audited
conversion. Passing here does not validate an untested receiver installation.

See [the receiver architecture decision](../../docs/adr/0003-slicer-pet-integration.md)
and the dated validation reports under `docs/validation/` for evidence and limits.
