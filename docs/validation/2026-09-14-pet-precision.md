# PET per-image precision validation — 2026-09-14

## Result and scope

Classic PET now fits a separate rescale slope to each image plane. This removes
the cross-slice/timepoint precision coupling measured in the
[previous audit](2026-09-13-pet-metadata-fix.md). The numerical encoder remains
shared; CT/MR retain their existing volume-wide scale. There is no new CLI flag
or public conversion option. See [ADR 0002](../adr/0002-pet-instance-scaling.md).

The final outputs were compared with the original public DICOM values, not just
with the encoder's predicted error. This is numerical and file-conformance
evidence, **not clinical validation or target-viewer/PACS acceptance**.

## Data and comparison

The same three public IDC v24 series were reused, without modifying original
DICOM files or the float64 reconstructed NIfTI values. All are GE classic PET,
128 × 128, in BQML. Source series UIDs and licenses are recorded in the
[machine-readable report](2026-09-14-pet-precision.json).

- RIDER static: 1 timepoint × 47 slices; 47 files.
- RIDER dynamic: 32 timepoints × 47 slices; 1,504 files.
- ACRIN FLT dynamic: 45 timepoints × 35 slices; 1,575 files.

Sources: [RIDER Lung PET-CT](https://doi.org/10.7937/k9/tcia.2015.ofip7tvm) and
[ACRIN FLT Breast](https://doi.org/10.7937/k9/tcia.2017.ol20zmxg), CC BY 3.0.
No additional download, private patient data or model inference was needed.

Before implementation, all three scaling policies were measured on identical
real-valued arrays. Whole-scan and per-timepoint are comparison policies, not
new selectable production modes. Serialized per-image output was then checked
against that experiment and the original DICOM values.

### Worst absolute timepoint mean bias, percent

Each cell is the maximum across that scan's nonzero timepoints, using the whole
spatial volume including background. The worst timepoint can differ by policy.

| Case | Whole-scan scale, before | Per-timepoint candidate | Per-image scale, final |
| --- | ---: | ---: | ---: |
| RIDER static | 0.00012954% | 0.00012954% | 0.00000822% |
| RIDER dynamic | 0.04475015% | 0.00396813% | 0.00169376% |
| FLT dynamic | 2.14136256% | 0.00931281% | 0.00167765% |

For the previously worst FLT timepoint (fourth, zero-based index 3), source mean
was 13.745192464 BQML. Whole-scan encoding produced 13.450858058 BQML; per-image
encoding produced 13.745075662 BQML. Its mean bias changed from −2.14136% to
−0.00084977%, and its maximum absolute voxel error fell to 0.006402616 BQML.

| Case | Worst relative L1 error, before → final | Nonzero samples rounded to zero, before → final |
| --- | ---: | ---: |
| RIDER static | 0.00323710% → 0.00099304% | 2,761 → 0 |
| RIDER dynamic | 0.38898485% → 0.00197270% | 462,899 → 0 |
| FLT dynamic | 8.69237713% → 0.00196526% | 2,629,296 → 0 |

Mean bias is `100 × (sum(output) − sum(source)) / sum(source)`.
Relative L1 error is `100 × sum(abs(output − source)) / sum(abs(source))`;
it is not mean per-voxel relative error. Zero-denominator timepoints have no
relative metric. Absolute errors and nonzero-to-zero counts are still checked.

### Fixed-region time–activity curves

The audit also used the same central spatial box at every timepoint: each
source axis is restricted to `[size//4, 3*size//4)`. This is a reproducible
geometric region, **not an anatomical or tumor segmentation**. Per-timepoint
source/output means, frame times and durations are in the JSON report.

Worst box mean bias changed from 0.061525% to 0.001682% for dynamic RIDER and
from 0.426876% to 0.001725% for FLT. Not every aggregate metric improves
monotonically: static RIDER box mean bias increased from 0.00005792% to
0.00015631%, while its relative L1 error decreased. Finer quantization changes
rounding-error cancellation; a smaller step does not guarantee smaller bias in
every possible region. No clinical acceptance threshold was inferred.

## Spatial, temporal and metadata checks

The unchanged independent audit mapped **all 51,216,384 output voxels** through
DICOM world coordinates back to original source voxels. Every location and
timepoint was covered exactly once. Maximum world-coordinate disagreement was
0.00000334 mm for static RIDER, 0.00001478 mm for dynamic RIDER and zero for FLT,
unchanged from earlier geometry validation.

Every image satisfied its own independently calculated 16-bit precision bound,
and its recorded encoding error agreed with actual decoded error. Exact integer
planes were checked exactly. FrameReferenceTime, ActualFrameDuration, acquisition
times/dates, units and decay fields remained unchanged. A separate comparison
against the previous output confirmed all dataset fields unchanged except new
output identities/creation timestamps, pixel encoding fields and error descriptions.
Source DICOM and reconstructed NIfTI hashes matched the previous audit.

Deliberate spatial reversals and time reversal in reconstructed arrays remained
detectable; false uniform NIfTI timing was rejected without publishing output.
The permanent synthetic tests additionally mutate actual DICOM pixels/time tags
and use oblique physical-coordinate oracles. These checks do not prove that an
unprovenanced upstream NIfTI was never reordered before conversion.

## DICOM and independent-decoder checks

All **3,126 public instances** passed unmodified `dicom-validator 0.9.0` against
DICOM **2026c**, with zero reported errors. Four additional synthetic mixed-sign
dynamic PET images passed the same validator. The standard cache and validator
were not patched. `dciodvfy` was not run in this pass.

GDCM through SimpleITK 2.5.6 independently decoded every public image as float64.
Its quantitative values agreed exactly with pydicom for all 51,216,384 voxels.
The four mixed-sign synthetic images also agreed exactly. This tests an
independent image decoder, not interactive viewer behavior or PACS ingestion.

The review identified and corrected a cross-instance rule that file-by-file
validation alone would miss: PET PixelRepresentation must remain constant
throughout a series. Output is unsigned if all values are nonnegative, signed
if any value is negative. Per-image slopes remain independent. This follows
[PS3.3 C.8.9.1.1.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.html)
and the zero-intercept/16-bit requirements of the
[PET Image Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.4.html).
In mixed-sign series, positive integers above 32767 may require quantization.

## Regression and failure safety

The full local suite passes **730 tests**. Ruff lint/format checks, mypy over
33 source files and wheel/sdist builds pass. Python 3.13 was tested locally;
this pass did not execute the full CI Python-version matrix.

New regression coverage includes 48 signed spatial axis permutations × 3D/4D
for fractional PET, per-plane precision beside bright neighbors, per-image
error descriptions, exact fitting integers, signed/unsigned and tiny values,
and series-wide signedness when negatives first appear in a later timepoint.
Numeric scale underflow and non-finite computed offsets raise structured
errors. A real late-plane encoding failure leaves no partial output and preserves
an existing output on attempted overwrite.

The initial 99 precision regressions failed under whole-scan scaling before the
fix. Four numeric-underflow/rollback regressions also failed before their fix.
The mixed-sign series regression and forced-signed encoder tests exposed the
review correction, and an extreme-offset regression reproduced the raw exception
before a finite-parameter guard was added. Independent review found the corrected
PET policy sound; the reported extreme-offset error path was also addressed.

## Evidence and remaining limits

Local raw artifacts are under
`/private/tmp/nifti2dicom-pet-precision.Haih8e`: original-source links, regenerated
NIfTI/output, scripts, per-file hashes, complete validator reports, per-plane
errors, timepoint metrics and logs. This temporary location is not durable
dataset hosting. Earlier output generations were retained separately; only
`outputs/` and `reports/` are the final run. The JSON report records source-code,
wheel, script, report and standard hashes for this run.

Per-image scaling cannot make arbitrary floating-point values lossless or
prevent all faint-value loss beside a bright value in the same plane. No
nonzero source samples were lost in these three datasets; that is an observation,
not a general guarantee. Non-GE PET, genuine 4D MR/CT, intended viewer/PACS
acceptance and durable external-validation CI remain separate release work.
SEG behavior was not changed in this pass. Changes remain local and unpushed.
