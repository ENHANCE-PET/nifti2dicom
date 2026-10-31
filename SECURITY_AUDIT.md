# Security audit — 2026-10-31

Scope: source input/path handling, serialization/publication, workflows and
runtime dependencies. No clinical data, customer documents or model weights were
added or downloaded. Existing repository guidance and architecture were read.

## Repairs

- Legacy single/batch slice writers checked `exists()` before writing by path.
  Dangling output symlinks bypassed the check and created DICOM files outside the
  output folder; concurrent files could also be overwritten after validation.
- Reject symbolic links and serialize through exclusively created `xb` streams,
  returning OutputError when an output already exists. Modern transactional
  conversion continues using its existing staging/publication path.
- Require pydicom >=3.0.2 to exclude the affected 3.0.0/3.0.1 FileSet releases,
  and Pillow >=13 (also used transitively by highdicom). This requires Python
  >=3.11; CI/support guidance were updated. The Python 3.11 Ruff target also
  required the equivalent datetime.UTC spelling in the image writer.

## Verification

- Four synthetic regressions were observed failing against the original writer;
  all five added security tests pass, including decoded output pixels.
- Full suite: 832 passed. Slicer integration suite: 135 passed.
- Ruff lint and formatting checks pass; mypy reports no issues in 33 source files.
- After the equivalent datetime.UTC syntax migration: fresh lint/format/type
  checks pass, and both full suites were rerun (832 and 135 passed).
- Final Python 3.11 runtime dependency resolution succeeds.
- Final Python 3.11 runtime-only graph: pip-audit checked 15 packages, none skipped,
  and reported no known vulnerabilities.
- `uv build` successfully rebuilt the final sdist and wheel; the distribution
  guard found no local/data artifacts (121 sdist and 38 wheel entries checked).
  `git diff --check` passes.
- Parent independent source review approved the exclusive-write repair.

## Evidence and limits

- Advisory: https://github.com/advisories/GHSA-v856-2rf8-9f28
- Fresh GitHub advisory queries found no entries for nibabel, SimpleITK or
  highdicom. No known advisory finding is not a guarantee of no vulnerabilities.
- Tests ran on macOS/Python 3.13; the minimum Python 3.11 graph was resolved, not
  executed. No native Slicer session, IDC download or clinical validation ran.
- The legacy API deliberately accepts a caller-chosen output directory; exclusive
  file creation prevents final-component symlink/concurrent overwrite attacks,
  not hostile modification of an ancestor directory by its owner.
- Local repairs require a release before package consumers inherit them.
- No remote, push, PR, release or merge was performed.
