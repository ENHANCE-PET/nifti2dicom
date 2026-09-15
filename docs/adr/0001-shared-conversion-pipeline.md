# One conversion pipeline

Status: accepted, 2026-09-11.

The old image, SEG, RGB and PUMA paths used different orientation rules and
mixed reference metadata with newly encoded pixels. Passing shape checks did
not prove correct spatial alignment or intensities.

All public entry points now share readers, a geometry contract, pixel encoding
and staged publication. Writers receive explicit image meaning and validated
geometry. The CLI owns presentation. Compatibility wrappers retain signatures
but do not retain silent skips, clipping or vendor flips.

Scalar native-grid conversion is the default because resampling changes the
data. SEG uses the selected source grid and verifies correspondence after
serialization. Binary multi-segment SEG is the interoperability default;
probability maps, overlap stacks and LABELMAP need separate explicit profiles.

Consequences: a few unsafe historical behaviors become clear errors. Required
metadata must be supplied rather than guessed. The module count increases,
but each conversion rule has a single owner and the same tests cover all APIs.
