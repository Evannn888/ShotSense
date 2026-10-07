# Retained initial failures

The first isolated author-reference build using clang++ failed because this host could not resolve the C++ cmath header. The forward algorithm itself uses only C-compatible math; retry the unchanged function as C with math.h. No LUT inference/default was promoted on this failed parity build.

First focused run:2passed/1failed2.86s. Sampled resize differs by at most1.79e-7 from Torch; the1×1 uniform-color control amplifies float rounding through InstanceNorm and yields2.86e-6 weight difference, above the test's2e-6 bound. Same-input classifier architecture matches exactly. Preserve initial XML and predictor-parity.json. Use existing native F.interpolate directly for sources at most256²pixels, a bounded<1MiB float input, retaining sampled large-input resizing and the strict tolerance rather than loosening it. This fix precedes all frozen real-photo candidates.
