# DICOM file-export profiles

This describes the output contract for the 2.1 implementation. It is not a
claim of clinical validation, universal viewer support, or a complete DICOM
network conformance statement. The package writes files; it does not implement
C-STORE, DICOMweb, or a media-directory service.

## Common behavior

- Uncompressed Explicit VR Little Endian, with consistent dataset/file-meta
  SOP identities and new series and instance UIDs.
- Patient and study context comes from a selected coherent reference series.
  The source Frame of Reference UID is retained; a missing image reference UID
  causes an explicit warning and a new UID. SEG requires an existing one.
- Pixel geometry is patient LPS millimeters. Native image grids are preserved
  unless reference-grid resampling is requested. SEG uses the reference grid.
- Text is UTF-8. Missing type-2 values are empty, not invented patient facts.
- Scalar samples use signed or unsigned 16-bit integers. CT/MR have one shared
  volume-wide scale; PET has a separately fitted scale for each image instance.
  Reconstructed value = stored value × RescaleSlope + RescaleIntercept.
  Maximum encoding error is recorded in DerivationDescription (per image for
  PET, across the volume for CT/MR). NIfTI scaling
  is applied before encoding; reference scaling is not applied again.
- Matching geometry is not proof of derivation. Image exports do not invent
  SourceImageSequence links. SEG explicitly references the supplied source
  instances after physical-grid alignment.
- Series Laterality is a proven uniform L/R or empty when unknown. MR
  TriggerTime is emitted only for heart-gating ScanOptions. CT AXIAL/LOCALIZER
  classification is retained only from a uniform source with exact matching
  acquisition planes; unknown classification remains absent. Some validators
  require CT ImageType value 3 even when the source cannot establish it; the
  exporter does not invent a class to satisfy that check.

## Supported SOP classes

| Output | SOP Class UID | Profile |
| --- | --- | --- |
| CT | 1.2.840.10008.5.1.4.1.1.2 | Classic CT; temporal extension for 4D |
| MR | 1.2.840.10008.5.1.4.1.1.4 | Classic MR with quantitative rescale extension |
| PET | 1.2.840.10008.5.1.4.1.1.128 | Classic static/dynamic PET, not gated/reprojection |
| RGB | 1.2.840.10008.5.1.4.1.1.7 | RGB Secondary Capture, one image per plane |
| SEG | 1.2.840.10008.5.1.4.1.1.66.4 | BINARY, one segment per positive input label |

Enhanced multiframe references and output are outside these profiles.

## Standard-extended MR and CT attributes

The classic MR IOD does not include the Modality LUT module. To preserve NIfTI
values, this implementation declares a Standard Extended MR profile with the
following additional dictionary attributes in the image instance:

| MR addition | Meaning and relationship |
| --- | --- |
| RescaleIntercept (0028,1052) | Offset applied to this instance's stored pixels |
| RescaleSlope (0028,1053) | Multiplier applied before that offset |

Both attributes are written together and have the same values across the
series. They do not change stored-pixel encoding or the MR SOP Class UID.
Readers that ignore them may report stored values instead of NIfTI values.
The conversion report warns about this receiving-system requirement.

For 4D CT, the following image-instance extensions preserve the temporal axis:

| CT addition | Meaning and relationship |
| --- | --- |
| TemporalPositionIdentifier (0020,0100) | One-based input timepoint index |
| NumberOfTemporalPositions (0020,0105) | Number of input timepoints |
| TemporalResolution (0020,0110) | Timepoint spacing in milliseconds, when known |

Each timepoint contains a complete spatial stack. InstanceNumber and filenames
are time-major, then slice-major. These temporal attributes are already part
of the classic MR module and are not MR extensions. A receiver that ignores
the CT additions may combine timepoints into one spatial stack.

The added attributes are optional Type 3 additions to the base IOD, with their
dictionary semantics preserved. This follows the extension rules in
[DICOM PS3.2 §7.3](https://dicom.nema.org/medical/dicom/current/output/chtml/part02/sect_7.3.html).
See also the [MR IOD table](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.4.3.html)
and [MR module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.3.html).
Strict base-IOD validators may report these declared additions as unexpected.

## PET constraints

PET samples must already use the reference Units; the converter cannot infer
SUV versus activity concentration. PET intercept is zero. Required acquisition
metadata must exist, and reference frames must match the output grid and
timepoints. Known NIfTI spacing must agree with reference timing for every
physical slice, not merely the first slice.

Each output PET image uses its own rescale slope, chosen from that plane's real
values. PixelRepresentation is constant throughout the series: signed if any
value is negative, otherwise unsigned, as required by
[C.8.9.1.1.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.html).
Consumers must decode each instance separately before stacking quantitative
values. Integer planes fitting that representation remain exact; other planes
use nearest rounding after the slope is represented as a DICOM decimal string.
An underflowed or non-finite calculated scale is an explicit encoding error.
This follows the [PET Image Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.4.html).
See [the scaling decision](adr/0002-pet-instance-scaling.md) for trade-offs.
In a mixed-sign series, positive integers above 32767 can require quantization
because signed storage and PET's zero intercept must both be respected.

Frame matching preserves exact discrete index correspondence while allowing
bounded NIfTI-1 affine coefficient roundoff at all eight physical corners.
Tolerance cannot exceed 0.001 mm in physical displacement; a shifted or
progressively stretched grid is not accepted just because its size matches.

FrameReferenceTime and ActualFrameDuration are retained only from matched
reference frames. Sample spacing does not establish an effective frame time
or its origin relative to SeriesDate/SeriesTime. Such times are never invented.
Decay-corrected output also requires the corresponding source DecayFactor.
Dynamic PET uses native NumberOfTimeSlices, NumberOfSlices and ImageIndex;
MR-style temporal attributes are not added to PET.

The reader reports and recovers consistently descending native PET slice
indices only with complete indexing and independently increasing frame times.
Original source indices and identities are unchanged; output ImageIndex follows
increasing physical slice position. Mixed/permuted/incomplete indexing is not
treated as a recoverable reversal. Exact MR spatial reorientation can preserve
volume-uniform acquisition facts even when planes change; conflicting plane
facts are omitted with warnings. See [the recovery decision](adr/0004-reference-recovery-and-volume-facts.md).

NIfTI volume order is preserved. Without per-frame provenance, this does not
prove an upstream tool left the original time order intact. Unknown temporal
spacing allows reference timing to be retained for matching frames; it is not
independent proof of temporal identity.

PET omits incomplete optional RadiopharmaceuticalCodeSequence entries from
output copies with a recorded warning. Other isotope fields, complete codes,
frame timing and source files are unchanged. PET and SEG share the basic
code-completeness policy; each writer names only the optional fields it may
omit. This is not full terminology or arbitrary nested-metadata validation.

The [2026-09-13 public PET audit](validation/2026-09-13-idc-pet.md) exposed this
copying defect. After the fix, all 3,126 regenerated instances passed unmodified
DICOM 2026c validation with unchanged voxel/geometry/time fidelity. The
[follow-up report](validation/2026-09-13-pet-metadata-fix.md) preserves both the
before/after evidence and the quantitative-precision concern subsequently
addressed by [per-image PET scaling](validation/2026-09-14-pet-precision.md).

## SEG contract

Input 0 is background. Nonzero integer values become consecutive SegmentNumber
values; the JSON report and SegmentDescription preserve the original mapping.
Finite, nonnegative values, nonempty foreground and truthful algorithm
provenance are required. Automatic and semiautomatic output records the creator
name and version, not the conversion library as the segmentation algorithm.

Segment descriptions include caller-supplied semantic codes and display colors.
Missing semantic codes are explicitly unspecified local codes with warnings;
anatomy is not guessed from label numbers or names. Both binary and multilabel
inputs produce BINARY SEG. Separate overlapping-mask inputs, probability maps,
LABELMAP output and time-varying SEG require future profiles.

Physical-grid alignment uses nearest-neighbor sampling. Foreground outside the
reference extent and segments lost during resampling are errors. After writing,
highdicom reconstructs the mask by source SOP Instance UID, including empty
planes, and the result must exactly equal the encoded label map.

Optional ProcedureCodeSequence, RequestingServiceCodeSequence and
ReasonForPerformedProcedureCodeSequence copied from the study are omitted with
a warning if a basic coded entry is incomplete. Complete entries are retained,
including URN codes that do not require a CodingSchemeDesignator. See
[General Study](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.2.html)
and [Code Sequence Macro](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_8.8.html).

## Validation evidence and remaining acceptance work

On 2026-09-11, synthetic classic CT, PET and RGB output passed external
dicom-validator 0.9.0 checks against DICOM 2026c. MR checks reported only the
declared rescale additions; 4D CT also reported the declared temporal additions.
Dynamic PET with slice-specific frame times was checked separately.

The stock validator reported six unexpected functional-group sequences in SEG.
Its specification parser expects section titles ending in "Functional Group
Macros", while 2026c A.51.5 is named "Segmentation Functional Groups"; the parsed
SEG macro table was consequently empty. Loading that same section with its own
table parser and supplying the seven macro definitions **in memory** yielded
zero errors for the unchanged SEG. The file, validator and cached standard were
not modified. This is a diagnosed validator parsing defect, not a reason to
remove SEG geometry or source references. These groups are specified by
[A.51.5](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.51.5.html)
and [C.8.20.3](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.3.html).

Automated tests additionally check decoded intensities, independent world
coordinates, temporal grouping, sparse labels and source-frame reconstruction.
They do not replace acceptance testing in the intended PACS/viewer. Before
release, validate representative real acquisitions and confirm quantitative
values, orientation, temporal navigation and SEG display in those systems.

The [baseline installed Slicer audit](validation/2026-09-14-slicer.md) found
receiver-side casting and temporal-grouping defects. The optional
[external integration](../integrations/slicer/README.md) now fixes them:
the [repair acceptance](validation/2026-09-14-slicer-fix.md) passes normal import
of static PET, both dynamic acquisitions as Sequences and MultiVolumes, and
binary/multilabel SEG. All 51.2 million distinct source-grid voxels retain their
exported quantitative values and physical/time identities. This applies to the
tested integrated Slicer 5.12.0 profile, not stock Slicer or every viewer/PACS.

The [2026-09-12 LMU real-data report](validation/2026-09-12-lmu-real-data.md)
records fresh LION and MOOSE masks with exact independent dcmqi reconstruction,
CT checks, validator versions and remaining acceptance limits. Older validators
can disagree with current requirements: the cached 2017 dicom3tools does not
recognize SegmentsOverlap and requires ContentCreatorName, which is now
[optional Type 3](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_10.9.3.html).
