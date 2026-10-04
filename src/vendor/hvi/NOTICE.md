# HVI-CIDNet inference sources

Copyright (c) 2024 Yixu Feng. MIT license, retained in LICENSE.

Source: https://github.com/Fediory/HVI-CIDNet
Pinned revision: eb43d7d91e9a336c66856824ff9e4603ae41f408.

Only four inference architecture files are included. Trailing whitespace was normalized. Local modifications: package-relative imports; removed the optional Hugging Face download mixin; replaced four einops rearrangements with equivalent native Torch reshapes. Network mathematics and checkpoint keys are unchanged. No upstream training/data/demo framework is included.

The separately prepared author Generalization checkpoint is pinned and checksummed in artifacts/jpeg_restoration/manifest.json. Checkpoint training provenance is incomplete; previously inspected LOL images are diagnostic examples, not independent acceptance data.
