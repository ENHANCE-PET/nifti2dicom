# Releasing nifti2dicom

Publishing a GitHub release automatically uploads the source distribution and
wheel to PyPI through `.github/workflows/python-publish.yml`. Ordinary branch
pushes and merges run CI but do not publish a package.

## Release checklist

1. Choose an unused version on PyPI and update `pyproject.toml`. Keep package
   version declarations consistent. Do not reuse or replace a released version.
2. Open a PR into `main`. Wait for the full Python 3.10–3.13 test matrix, lint,
   type checks, package builds and archive checks to pass on its final commit.
   Review the changed files and relevant real-data/viewer evidence. Unit tests
   alone do not establish DICOM interoperability or clinical suitability.
3. Merge the reviewed PR and verify CI on the merged commit. Do not bypass
   failing checks, even when repository settings do not require them.
4. Publish a GitHub release with a tag such as `v2.1.0`, targeting that exact
   merged commit. Match the tag to the package version and include migration
   notes, supported profiles and important limitations in the release notes.
5. Monitor **Upload Python Package** through completion. It reruns tests,
   Ruff and mypy, builds both archives, checks their contents, then uploads
   through the pinned PyPA publishing action. The existing repository secret
   is named `PYPI_API_TOKEN_N2D`; never place its value in source, logs or docs.
6. Confirm that PyPI serves both files for the expected version. Download them,
   compare their SHA256 hashes with PyPI metadata, inspect the archives and
   test installation outside the repository in a fresh environment. Check the
   CLI and representative conversion paths, not only package import.

## Package boundaries

The wheel contains the runtime package. The source distribution additionally
contains tests, documentation, validation utilities and the optional Slicer
integration. Local imaging data, viewer databases, environment files and
internal work notes must not enter either archive or the public commit.

```sh
python -m build
python -m validation.check_distribution dist/*.tar.gz dist/*.whl
```

Keep public/sanitized validation summaries in `docs/validation/`; keep raw
clinical artifacts and private manifests on their approved storage. See
[the validation guide](../validation/README.md) for opt-in data/receiver checks.

If publishing fails, inspect the workflow and PyPI before retrying: an upload
may have partially succeeded. Do not delete or overwrite published artifacts
to conceal a mismatch. Fix the release process or publish a new version when
the package contents need to change.
