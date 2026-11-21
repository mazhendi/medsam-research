# Native SAM2 attribution

Official source: https://github.com/facebookresearch/sam2

Pinned dependency: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Apache-2.0 license is reproduced in LICENSE.

`NativeSAM2._decode_with_rcsr` adapts the token assembly and single-mask generation
sequence from `sam2/modeling/sam/mask_decoder.py:MaskDecoder.predict_masks`.
Changes: the first native transposed-convolution output receives a region-conditioned
resampling residual before the original high-resolution addition; only the native
single-mask output is assembled. Native decoder weights and transformer are reused.
The no-RCSR path calls the official decoder directly. SCDS/SBCA/RCSR are experimental
project additions, not established improvements or verified original contributions.
