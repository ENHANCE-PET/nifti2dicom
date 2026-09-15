# Working on nifti2dicom

Read `docs/architecture.md` before changing conversion behavior. Read the
specific reader or writer involved rather than importing legacy wrappers into
new code. The public API is `nifti2dicom.convert` and `nifti2dicom.inspect`.

## Where changes belong

- Input classification: `inference.py`.
- NIfTI/DICOM loading: `readers/`; geometry and resampling: `geometry.py`.
- Label descriptions and provenance: `labels.py`.
- Pixel encoding: `pixels.py`; DICOM serialization: `writers/`.
- Copied-code completeness: `writers/codes.py`; writers own optional-field allowlists.
- Orchestration: `pipeline.py`; staged publication: `publication.py`.
- Shared typed values: `models.py`; error codes and hints: `errors.py`.
- Terminal output and flags: `cli/`. No terminal output in library modules.
- Historical modules are compatibility adapters, not alternative engines.

## Invariants to protect

- Shared arrays are TZYX (TZYXC for RGB); affine indices are XYZ in LPS mm.
- Geometry tags and pixels describe the same physical locations.
- Label IDs and physical source-frame references survive SEG round trips.
- Missing anatomy, acquisition timing and segmentation provenance are not invented.
- Rescaling never clips or wraps silently; all output identities are coherent.
- PET slopes are per image, but signedness is constant across the series.
  Integer exactness is conditional on that selected storage representation.
- Validate in staging before publishing; never delete a rollback backup on failure.

## Verification

Run `.venv/bin/python -m pytest` and
`.venv/bin/python -m pytest integrations/slicer/tests`. Lint and format-check
`nifti2dicom tests integrations/slicer validation` with Ruff. Build with
`.venv/bin/python -m build`, then inspect the actual archives using
`.venv/bin/python -m validation.check_distribution dist/*.tar.gz dist/*.whl`
before release. CI exercises Python 3.10–3.13.
Type-check with `.venv/bin/python -m mypy nifti2dicom --ignore-missing-imports`.

Opt-in IDC downloads, independent pixel/geometry audits, external conformance
validators and installed-Slicer commands live in `validation/README.md`.
Keep imaging data and local viewer databases outside distributions and git.
For PR, merge and PyPI publication, follow `docs/releasing.md`. A published
GitHub release triggers package upload; branch pushes alone do not.

Add a regression that fails for the incorrect behavior before implementing a fix.
Assert decoded values and independently calculated world coordinates, not just
file counts or source text. SEG tests should reconstruct by source SOP Instance
UID. Fixtures live in tests; never add test-only branches to production code.

Update README support/migration notes when changing an input profile or public
behavior. Record consequential architecture choices in `docs/adr/`.
