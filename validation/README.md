# Opt-in public-data acceptance

These tools are developer validation utilities, not package runtime modules.
Run from the repository root with an explicit cache and a new output directory.
Never add imaging files or patient data to git.

The pinned [IDC manifest](../docs/validation/2026-09-15-idc-candidates.json)
contains exact source UIDs, CC BY licenses and source DOIs. Its fifteen selected
series total approximately 608 MB: Philips/Siemens static and dynamic PET,
25-timepoint sagittal MR, and ten separate respiratory CT phases. The extra
description-only negative candidate is not downloaded.

```sh
uv run --no-project --with idc-index==0.12.5 python -m validation.download_idc \
  --cache /absolute/path/idc-v24

.venv/bin/python -m validation.idc_roundtrip \
  --cache /absolute/path/idc-v24 --output /absolute/path/new-qa-run

.venv/bin/python -m validation.idc_ct_ambiguity \
  --cache /absolute/path/idc-v24 \
  --nifti /absolute/path/new-qa-run/ct_respiratory_00/input.nii.gz \
  --report /absolute/path/new-qa-run/ct-ambiguity.json
```

Repeat `--case ALIAS` to select cases. Existing conversions require explicit
`--overwrite`; use a new directory to retain before/after evidence. The oracle
checks every decoded output voxel against independently reconstructed source
physical coordinates, complete coverage, per-image quantization bounds,
acquisition facts, time identities and unchanged source hashes. It does not
import converter geometry or grouping helpers. Its reconstructed-array reversal
checks are sensitivity tests; separate unit tests corrupt actual DICOM objects.

Each case produces a NIfTI input, truth NPZ, Slicer fixture `cases.json`, DICOM
output, and `roundtrip.json` with source/output/code hashes. Imaging artifacts
are local only. NIfTI time units are intentionally unknown to exercise reference
timing preservation rather than an invented uniform sample interval.

## Independent conformance

Obtain unmodified [dicom-validator](https://github.com/pydicom/dicom-validator)
standard JSON files and the official
[dciodvfy](https://dclunie.com/dicom3tools/dciodvfy.html) executable separately.
The recorded run uses DICOM 2026c and dicom3tools 20260901071806.

```sh
uv run --no-project --with dicom-validator==0.9.0 --with pydicom==3.0.1 \
  python -m validation.dicom_conformance \
  --root /absolute/path/new-qa-run \
  --standard /absolute/path/2026c/json --dciodvfy /absolute/path/dciodvfy
```

Every instance is checked and fingerprinted. Raw errors and warnings are retained;
an Error diagnostic is not treated as success merely because a process exits 0.
Strict base-IOD MR checks report the documented RescaleSlope/RescaleIntercept
extensions. The command intentionally returns nonzero for such raw findings;
interpret them against the [declared output profile](../docs/dicom-conformance.md),
not by changing validator definitions or deleting quantitative metadata.

## Normal installed Slicer

Use the [external PET integration](../integrations/slicer/README.md) for the
validated Slicer installation. Set `SLICER_QA_ROOT` to one generated case folder,
`SLICER_QA_CASE` to its alias, and a distinct `SLICER_QA_RUN`. Run
`integrations/slicer/tests/check_public.py` in Slicer, from a temporary working
directory. Normal startup must load the installed integration; `--testing`
skips user startup scripts. Do not run concurrent GUI checks sharing settings.

The new cases have `seg_kinds: []`: they check scalar images, not segmentation
inference. The existing binary/multilabel SEG fixtures remain separate acceptance
cases. Both sequence and MultiVolume representations are checked where supported.

## Distribution hygiene

After building, inspect the actual source archive and wheel:

```sh
.venv/bin/python -m build
.venv/bin/python -m validation.check_distribution dist/*.tar.gz dist/*.whl
```

Source archives have an explicit file allowlist. Both targets exclude local
databases, raw imaging data, environment files and internal work notes. The
archive check fails if such artifacts are present or required package files
are missing. This does not remove any local files and is not a general privacy
scanner; documentation and source still require ordinary review.
