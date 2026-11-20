# Coarse-to-fine B + existing RS (C)

This is the new agreed design, separate from the older scale-detail experiment.
Default old configs remain unchanged. No trained accuracy gain is established.

## Forward interfaces

1. Frozen encoder produces F1/F2/F3. The original baseline decoding produces P0.
2. B processes F3 using a residual gated Conv-FFN, producing F3*.
3. F2 queries local neighborhoods of F3* (K/V), producing F2* after projection,
   residual addition and gated FFN.
4. F1 queries local neighborhoods of F2* (K/V), producing F1* in the same way.
5. Existing C (region-shape head) consumes ALL three enhanced features and P0.
   Original query/shape/competition/region-update rules and prompts are retained.
6. Final SAM decoding consumes F3* image embeddings, F1*/F2* high-resolution
   features, and the existing dense/sparse prompts. No original feature bypass
   enters final decoding. Internal B residuals are intentional identity paths.

Each local attention uses channel LayerNorm and independent learned Q/K/V
projections. Fine query positions map to the native coarse pixel grid using
half-pixel alignment. A 3x3 coarse-pixel neighborhood is sampled bilinearly;
outside normalized image bounds is masked, with border extension inside bounds.
Fine query grids remain unchanged. The two attention blocks have independent
parameters. Default attention width=32, single attention head, no positional
bias or candidate-region bias. FFN expansion=2, depthwise 3x3 + GELU multiplied
by a separate linear channel branch. These are concrete implementation choices,
not claims of novel operators.

Output projections of both attention blocks and all FFNs are zero initialized.
Initially B is exactly identity in FP32, so warm-started RS predictions are
preserved (within precision tolerance). Internal Q/K/FFN expansion parameters
can have zero gradient on the first step; they activate after output projection
updates. Under autocast use numerical tolerances rather than bitwise equality.

## Stage one: train ONLY B

Load a trained BASELINE checkpoint AND its matching trained REGION-SHAPE checkpoint:

```bash
git pull origin main
medsam train-region-shape \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/YOUR_BASELINE/best.pt \
  --mechanism-config configs/region_shape_multiscale_b.yaml \
  --initialize-from outputs/YOUR_RS/best_region_shape.pt \
  --output outputs/region_shape_multiscale_b_seed42 \
  --epochs 50
```

Replace the two checkpoint paths with your own. Keep the original light data
config, baseline hash and original RS settings. If your RS used nondefault
settings (width, rounds, routing, grid, etc.), copy them into the new YAML.
B-only training rejects missing initialization: random frozen RS is invalid.
Warm start rejects legacy scale-detail checkpoints and already-refined models.
The optimizer is fresh; this is initialization, not optimizer resume.

Only `refinement.*` parameters are trainable. Encoder, decoder, RS head, prompt
adapters, candidate queries and task tokens stay fixed. Frozen modules STILL
participate in autograd on the B -> C/decoder -> loss paths. The initial baseline
pass uses original features and is performed without gradients. Both decoder
calls share the same weights; enhanced high-resolution paths stay enabled.
Existing auxiliary loss weights are unchanged. Gradient histories include B.

A 50-epoch run is a screening budget, not a known optimum. Match initialization,
additional training budget and validation selection for controls. Never pick an
epoch using the external test sets. CUDA memory/runtime and segmentation gains
need measurement on your 4090. Local attention memory scales with query pixels,
attention width and window area; reducing refinement_width is the first knob.

## Checkpoints and evaluation

Frozen trained RS parameters are saved alongside B; the decoder is also included
for this new mode so warm-started decoder changes survive reload. The frozen
encoder remains supplied by the baseline. Existing old checkpoints load normally.
New checkpoints require updated code. The baseline file is never overwritten.

```bash
medsam evaluate-region-shape \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/YOUR_BASELINE/best.pt \
  --checkpoint outputs/region_shape_multiscale_b_seed42/best_region_shape.pt \
  --manifest data/YOUR_TEST_MANIFEST.csv \
  --output outputs/region_shape_multiscale_b_test
```

For a later joint B+C experiment set `refinement_only: false` in a copied config.
Decoder training remains controlled separately by `train_decoder`. Do not change
these switches while evaluating an existing artifact; saved config is authoritative.

## Verification scope

Tests cover three native grids, identity initialization, semantic dependencies,
frozen parameter immutability, two-step gradients, trained-RS warm-start identity,
B-only train/save/reload/evaluate and the real SAM2 small-image decoder API.
These are implementation checks, not a trained performance reproduction.
