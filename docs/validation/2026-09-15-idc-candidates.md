# IDC candidates for the next validation round

Checked 2026-09-15 using `idc-index 0.12.5`, IDC **v24**, and the public
DICOMweb metadata endpoint. This is **source-metadata discovery**, not a new
conversion, pixel-accuracy, viewer, or DICOM-conformance validation result.

The [machine-readable manifest](2026-09-15-idc-candidates.json) contains the SQL
queries, exact study/series UIDs, source DOIs, licenses, immutable series storage
prefixes, metadata URLs, response hashes, and measured header evidence.
All 6,619 instance headers across 16 series were retrieved and checked against
the index counts and series identities. No pixel data was downloaded.

## Selected cases

Sizes are full-series download estimates from the index, in decimal MB.
Timepoints and slice counts below were verified against per-instance headers,
not inferred from filenames or descriptions alone (see the CT exception below).

| Case | Source | Header-confirmed organization | MB | License |
|---|---|---|---:|---|
| Philips 3D PET, GEMINI TF TOF 16 | [NaF Prostate](https://doi.org/10.7937/k9/tcia.2015.isoqthko) | 213 distinct slices, BQML | 9.57 | CC BY 3.0 |
| Siemens 3D PET, Biograph Horizon | [CMB-LCA](https://doi.org/10.7937/3cx3-s132) | 42 distinct slices, BQML | 2.90 | CC BY 4.0 |
| Philips dynamic PET, GEMINI TF TOF 16 | [ACRIN FLT Breast](https://doi.org/10.7937/k9/tcia.2017.ol20zmxg) | 45 timepoints × 45 slices, BQML | 92.41 | CC BY 3.0 |
| Siemens dynamic PET, model 1094 | [ACRIN FLT Breast](https://doi.org/10.7937/k9/tcia.2017.ol20zmxg) | 45 timepoints × 74 slices, BQML | 200.04 | CC BY 3.0 |
| Philips dynamic MR, Achieva | [QIN Breast](https://doi.org/10.7937/k9/tcia.2016.21juebh0) | 25 temporal positions × 20 sagittal slices | 39.37 | CC BY 3.0 |
| Varian respiratory cone-beam CT | [4D Lung](https://doi.org/10.7937/k9/tcia.2016.eln8ygle) | Ten separate phase-labeled series × 50 slices; provenance caveat below | 264.06 | CC BY 3.0 |

Together these require approximately **608.34 MB** of DICOM image downloads.
The two 3D PET series and three dynamic PET/MR series are candidates for the
current classic-image workflow; compatibility and pixel fidelity remain untested.
The selected CT exam is a provenance/phase-handling edge case, not an approved
single-series 4D reference.

## Why these are useful

- **Philips temporal reversal trap:** complete native PET `ImageIndex` values
  identify 45 increasing `FrameReferenceTime` groups, but ascending
  `InstanceNumber` would reverse the time blocks. All source frame durations
  are 4,996 ms despite varying frame-reference intervals. Preserve and assess
  this source metadata; do not invent a replacement duration schedule.
- **Siemens opposite slice traversal:** its native image indices traverse
  decreasing LPS Z, in 3 mm steps. All 45 timepoints have the same spatial grid.
  This is a useful test of assumptions that native slice indexing must match
  increasing projection onto the image-plane normal. No claim of current reader
  acceptance is made here.
- **MR orientation and nonuniform time:** the sagittal orientation is
  `[0, 1, 0, 0, 0, -1]`. Explicit temporal identifiers cover 1–25, each with the
  same 20 positions. Acquisition times increase, but include a 69.45 s interval
  among mostly 17.49 s intervals. A uniform NIfTI time spacing cannot describe
  that schedule faithfully on its own.
- **CT identity ambiguity:** all ten phase series have matching coordinate
  grids, but phases 0% and 10% use a different `FrameOfReferenceUID` from the
  remaining eight. The phase percentages are in `SeriesDescription`, supported
  by the collection's respiratory-imaging description; the standard temporal,
  trigger, and nominal respiratory-percentage fields examined are absent.
  Matching coordinates do not authorize rewriting spatial identity. These are
  respiratory phases, not a measured elapsed-time sequence or perfusion CT.

A negative-selection example was also checked: a TCGA-LUAD CT series labeled
`DYNAMIC MODE` contains nine distinct positions, each imaged once. It does not
establish a 4D volume. Initial MR index candidates also included diffusion and
magnetization-transfer series, demonstrating why temporal-count fields alone
are insufficient selection criteria.

## Next validation gate

Download the selected PET and MR images, independently reconstruct source pixels
and physical coordinates, then run NIfTI → DICOM round trips, temporal-order and
quantitative-value comparisons, DICOM validation, and native viewer checks.
Use the CT case to exercise clear rejection/provenance handling until an explicit
multi-series respiratory-phase workflow is defined. A second independent viewer
or PACS is still needed; finding a dataset does not supply that evidence.

No converter or Slicer integration code was changed during this discovery.
Neither `dciodvfy` nor the DICOM IOD validator was run in this round.
