# Multi-scale context cross-attention A (candidate)

Replaces BOTH A1 and A2 in the original aligned three-head topology. Original A
and its main configuration remain unchanged. No random augmentation, original
B/C, no CFM, and auxiliary weights [0.5, 0.5] are retained.

A1: X=raw F1 (64 channels), G=raw F2 (128), attention width=32.
A2: X=raw F2 (128), G=raw F3 (320), attention width=64.
Independent parameters; four attention heads each. No dependency on enhanced
A2 or B3 outputs, so there is no forward cycle.

- Q = DW3(PW(LN(X))). LN normalizes channels at each spatial position.
- P = PW(G); H = LN(resize(PW(cat(DW3(P), DW5(P), DW7(P))))).
- K = PW(H), V = PW(H), with separate projections.
- For query i, attention weights are softmax_j(q_i dot k_j / sqrt(head_dim))
  over valid locations j in the centered 5x5 neighborhood.
- Output = X + PW(sum_j weight_ij * v_j).

Bilinear resize uses align_corners=False. Invalid padded neighbors are masked
before softmax. Half/bfloat16 scores use FP32 softmax. No FFN, layer scale,
extra gating, auxiliary supervision, or extra wrapper residual. The legacy
`a_residual: false` flag does not disable the intrinsic X residual in this A.

Local attention uses activation checkpointing during training to avoid keeping
unfolded K/V tensors for backward. This trades recomputation for memory; GPU
memory, throughput and training quality have not been measured here.

## Run

```bash
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_cross_attention_a.yaml
```

Uses the original 1450-image pool, base input 352, sequential scale updates
256/352/448, batch 16, 99 epochs, AdamW 1e-4, and no random augmentation.
Output: `../runs/polyp_pvt_slots_three_head_cross_attention_a` relative to config
resolution rules, separate from the original main run. Start a fresh training
run with the PVT backbone checkpoint, not a strict resume of the old A model.
Prediction must use this same config, since A parameter shapes and signatures
are different. No improvement is claimed before matched experiments.

The design hypothesis is that shallow local queries benefit from multi-scale
deep context while the identity residual preserves shallow detail. Upsampled
context does not recover information already lost in the encoder. Multi-scale
convolution and cross-attention are established components; novelty requires
comparison with related methods and controlled experiments.
