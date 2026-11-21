# SAM2-UNet attribution

Architecture port: https://github.com/WZH0120/SAM2-UNet

Source commit: `01598e5e9912ffb23f965ecbebf4d1dfecbaa56e`.
License: Apache-2.0, reproduced in LICENSE in this directory.
Paper: https://doi.org/10.1007/s44267-025-00106-w .

`src/medsam_research/sam2_unet.py` adapts the upstream SAM2UNet.py.
We instantiate only the Hiera-L trunk and strictly extract its pretrained keys,
rather than constructing and deleting the SAM memory/prompt components.
An injected encoder supports lightweight contract tests. The Adapter/RFB/decoder
operations and three-output order are retained, including upstream's unused up4.
Our manifest handling, seeded workflow, non-overwriting outputs and provenance
are additions. This is a third-party baseline, not our proposed innovation.
