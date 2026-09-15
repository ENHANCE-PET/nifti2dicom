Supported profiles
==================

* Scalar 2D, 3D and 4D images: classic CT, MR or PET.
* Binary and integer multilabel masks: one BINARY DICOM SEG.
* 2D/3D RGB: Secondary Capture.
* Automatic dimensions, orientation, spatial units and unambiguous series selection.
* Native geometry by default; explicit physical-grid resampling.
* Structured errors, JSON reports, progress callbacks and staged publication.

Classic MR uses a standard-extended rescale mapping. 4D CT uses
standard-extended temporal attributes. Receiving systems must support those
attributes to preserve quantitative values and timepoint grouping.

Enhanced/multiframe references, diffusion/vector/tensor reconstruction,
temporal RGB, gated/reprojection PET, overlapping-mask inputs, probability
SEG and time-varying SEG are not supported yet.

See the repository's
`output profile specification <https://github.com/ENHANCE-PET/nifti2dicom/blob/v2-rewrite/docs/dicom-conformance.md>`_
for interoperability requirements.
