# Scale selection + region-constrained detail retrieval (experimental)

This extension preserves the original region-shape head, dense/sparse prompts,
SAM decoder and high-resolution skips. It does not replace the trained baseline.
The old `configs/region_shape.yaml` remains unchanged and default switches are off.

## Actual forward graph

One frozen encoder -> F1/F2/F3 -> initial SAM decoding -> P0.
The unchanged region-shape head reads the pyramid and P0, producing candidate
occupancies, global weights and dense/sparse prompts. B refines F3; C queries
native F2 neighborhoods using B's output and candidate affinities. Final SAM
decoding receives refined F3, original F1/F2 skips, and existing learned prompts.
There remain one encoder pass and two decoder passes. The earlier generated
figure omitted the prompt path and contained misconnected wires; it is not the
implementation specification. This document and source are authoritative.

B: 1x1 projection -> three DWConv3x3+GELU branches (dilations 1,2,3).
Concat of the THREE BRANCH OUTPUTS -> 1x1 -> per-location scale softmax.
Weighted sum -> zero-initialized output projection -> residual F3.

C: semantic 1x1 Q; native detail 1x1 K/V; bilinear 5x5 neighbor sampling.
Query centers map to native detail coordinates with align_corners=False.
Out-of-image neighbors are masked before softmax, retaining the center.
For candidate probabilities ci,cj, agreement=ci*cj+(1-ci)*(1-cj),
confidence=abs(2ci-1)*abs(2cj-1). Bias is log of the globally weighted sum
of `0.5 + confidence*(agreement-0.5)`, clamped at 1e-6.
Attention=softmax(QK/sqrt(width)+detail_bias*bias); weighted V -> zero output
projection -> residual semantic. No GT, hard candidate selection or text is used.
Both modules start as identity; internal controller/Q/K gradients can be zero
on step one, then become active after the output projections update.

## Run

From the repository root, use your existing light config and original baseline:

```bash
medsam train-region-shape \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/YOUR_BASELINE/best.pt \
  --mechanism-config configs/region_shape_scale_detail.yaml \
  --initialize-from outputs/YOUR_REGION_SHAPE/best_region_shape.pt \
  --output outputs/region_shape_scale_detail_seed42 --epochs 50
```

`--initialize-from` is optional. It copies the full existing region-shape model
(including trained decoder if enabled) after verifying baseline hash, evaluation
settings and original mechanism settings. Only the new zero-output feature
modules are freshly initialized. This is a fresh optimizer run, not resume.
If the old mechanism had nondefault settings, copy those settings into the new
YAML first; mismatches are rejected rather than silently ignored.

Use `region_shape_scale_only.yaml` and `region_shape_detail_only.yaml` for B/C
ablations. For a no-region-bias C control set `detail_bias: 0.0`. Give the old
region-shape control the same additional training budget from the same weights;
otherwise extra training and module effects cannot be separated. Keep data,
light augmentation, seeds and validation checkpoint rule identical. Choose
checkpoints on validation, never on the five test sets. Start with 50 epochs as
a screening budget, not a guaranteed optimum; inspect learning/overfit curves.

Evaluation uses the existing evaluate-region-shape command with the new saved
checkpoint; architecture settings are stored in the artifact. Gradient histories
now include scale_module and detail_module. Existing old checkpoints load with
both switches disabled. New checkpoints require this updated code.

CPU checks cover identity, two-step gradients, candidate permutation, actual
region-bias effects, border masking, warm-start identity and real SAM2 API.
No trained 4090 run or accuracy improvement has been established. Local sampling
allocates O(batch*width*25*H3*W3); measure CUDA peak memory/latency on your setup.
