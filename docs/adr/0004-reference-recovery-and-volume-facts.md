# 0004 — Recover ordering only when proven; distinguish volume facts from plane facts

Status: accepted, 2026-09-15.

## Context

Public IDC Siemens dynamic PET has complete native ImageIndex blocks but orders
each block opposite to increasing physical slice position. The source violates
[PS3.3 C.8.9.4.1.9](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.4.html).
Sorting filenames or InstanceNumber instead would not establish time identity.

Public sagittal dynamic MR undergoes an exact spatial axis permutation when
NIfTI is canonicalized. Its voxels still occupy the same physical locations,
but output planes no longer correspond one-to-one to acquisition planes.
Rejecting all metadata correspondence discarded valid, volume-wide acquisition
times, including an irregular gap. Copying arbitrary plane metadata would be
equally wrong.

## Decision

- The DICOM reader may recover a consistently reversed PET slice index only
  with complete native indexing, the same exact reversal in every timepoint,
  and complete finite FrameReferenceTime increasing at each physical slice.
  Mixed directions, arbitrary permutations, incomplete indices and time
  reversal remain errors. Recovery returns an explicit source-nonconformance
  warning. Original headers, SOP identities and pixel associations are intact.
- Fresh PET output always uses conformant increasing physical ImageIndex order.
  No vendor/model branches or source-header rewrites implement this recovery.
- Writers separately establish exact signed-permutation voxel correspondence
  and acquisition-plane correspondence. Existing dimension, discrete-offset
  and physical-corner tolerances are unchanged.
- When MR reorientation mixes acquisition planes, only an explicit allowlist
  of facts uniform throughout each contributing volume may be retained.
  Temporal identifiers and declared counts must establish complete volumes.
  Conflicting facts are omitted with warnings; incomplete required MR profile
  facts fail. Plane-specific source links and lossy-compression history are
  not copied through this route.
- Explicit NIfTI timing must agree with reference evidence before acquisition
  facts are retained. Empty source fields cannot overwrite explicit spacing.
  Irregular acquisition times do not imply a uniform TemporalResolution.
- Series-level Laterality and CT/MR PatientPosition are assigned once from
  proven uniform source facts, otherwise empty. Laterality is restricted to
  L/R. MR TriggerTime follows actual output heart-gating ScanOptions. Known
  CT AXIAL/LOCALIZER classification survives only complete exact frame matching
  and uniform source classification. Unknown anatomy/classification is not guessed.

## Consequences

The public PET/MR failures are handled without resampling or changing temporal
array order. Spatial matching is still not proof that an upstream application
preserved the NIfTI's temporal provenance. AcquisitionTime may denote acquisition
start rather than phase time; no generic chronological reorder is inferred.

Unrelated CT respiratory series are not silently combined into a 4D acquisition.
Selection remains explicit, and distinct Frame of Reference identities are not
rewritten. Public acceptance independently reconstructs source pixels and world
coordinates, then uses external validators and normal target-viewer loading.
