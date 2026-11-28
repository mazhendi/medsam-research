# Dual-source values and aligned G skip (candidate A)

Config: `configs/polyp_pvt_slots_three_head_dual_source_a.yaml`.
This replaces BOTH A1/A2 in the original aligned three-head topology.
Original A and the previously evaluated deep-value-only cross-attention A remain
available with unchanged configurations and state dictionaries.

## Exact graph

```
Nx = LN(X)
Q = DW3(PW(Nx))
Vx = PWx(Nx)
P = PW(G)
H = LN(resize(PW(cat(DW3(P), DW5(P), DW7(P)))))
K = PWk(H); Vg = PWv(H)
W[i,j] = softmax_j(Q[i] dot K[j] / sqrt(head_dim))
Ox[i] = sum_j W[i,j] Vx[j]
Og[i] = sum_j W[i,j] Vg[j]
Y = X + resize(PWskip(G)) + PWout(cat(Ox, Og))
```

j ranges over valid positions in the centered 5x5 neighborhood. Four heads;
A1 has width 32 with X=raw F1(64), G=raw F2(128); A2 has width 64
with X=raw F2(128), G=raw F3(320). Independent parameters at both stages.
One attention weight computation is reused for both value sources. The G skip
starts from raw G, with its own 1x1 projection to X channels, then bilinear
resize to X spatial dimensions. All new A resizing uses align_corners=False.

No FFN, learned residual scale, extra gating or extra loss. `a_residual: false`
only disables the legacy wrapper; this new A has intrinsic X and G skips.
Activation checkpointing recomputes local attention during training to reduce
saved neighborhood tensors. GPU peak memory, speed and accuracy are unmeasured.

## Training

```bash
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_dual_source_a.yaml
```

Start fresh from the PVT encoder checkpoint; do not resume an old A checkpoint.
Prediction must use this same new configuration. Separate run directory:
`runs/polyp_pvt_slots_three_head_dual_source_a`.
Original B/C, aligned 44-grid final fusion at base input 352, three heads,
auxiliary weights [0.5,0.5], 1450 images, 99 epochs, batch 16, AdamW 1e-4,
sequential scales 256/352/448 and no random augmentation are retained.

This candidate simultaneously introduces shallow values and a projected deep
skip. Its comparison with the previous A measures their combined effect, not
individual causal contributions. It cannot guarantee retaining Kvasir accuracy.
Separate ablations are needed before attributing any improvement to either path.
