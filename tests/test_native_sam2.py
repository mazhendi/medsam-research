import csv

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from medsam_research.native_sam2_workflow import load_native_config, predict_native


def test_native_configs_define_independent_ablation_outputs():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    configs = [load_native_config(root / f"native_sam2_{name}.yaml")
               for name in ("baseline", "scds", "sbca", "rcsr", "modules")]
    assert len({cfg["output"] for cfg in configs}) == 5
    assert [sum(cfg["modules"].values()) for cfg in configs] == [0, 1, 1, 1, 3]
    assert all(cfg["batch_size"] == 16 for cfg in configs)


@pytest.mark.integration
def test_real_native_352_parity_and_module_gradients():
    pytest.importorskip("sam2")
    from medsam_research.native_sam2 import build_native_model

    torch.set_num_threads(2)
    # Real native decoder and Hiera-T for a lightweight integration test.
    cfg = dict(sam_config="configs/sam2/sam2_hiera_t.yaml",
               modules=dict(scds=True, sbca=True, rcsr=True))
    model = build_native_model(cfg, allow_random=True).eval()
    images = torch.randn(1, 3, 352, 352)
    scds, sbca, rcsr = model.scds, model.sbca, model.rcsr
    with torch.no_grad():
        initial = model(images)
        model.scds = nn.ModuleList()
        model.decoder.transformer.layers[0].cross_attn_token_to_image = sbca.base
        model.rcsr = None
        baseline = model(images)
    assert initial.shape == (1, 1, 352, 352)
    assert torch.allclose(initial, baseline, atol=1e-6, rtol=1e-5)
    model.scds = scds
    model.decoder.transformer.layers[0].cross_attn_token_to_image = sbca
    model.rcsr = rcsr
    # Open residuals to exercise all branch interiors without an artificial huge step.
    for name, parameter in model.named_parameters():
        if name.endswith("gain"):
            parameter.data.fill_(.2)
    nn.init.normal_(rcsr.predict[-1].weight, std=.005)
    target = torch.zeros(1, 1, 352, 352)
    target[:, :, 100:180, 110:190] = 1
    logits = model(images)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss = loss + .1 * model.auxiliary_loss(target)
    loss.backward()
    assert model.encoder.blocks[0].prompt_learn[0].weight.grad.abs().sum() > 0
    assert model.scds[0].attention.in_proj_weight.grad.abs().sum() > 0
    assert sbca.background_query.grad.abs().sum() > 0
    assert rcsr.condition[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters()
               if not parameter.requires_grad)
    assert torch.isfinite(loss)


def test_native_prediction_needs_no_labels_or_pretrained_file(tmp_path, monkeypatch):
    import medsam_research.native_sam2_workflow as workflow

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 1)

        def forward(self, x):
            return self.conv(x)

    monkeypatch.setattr(workflow, "build_native_model", lambda *args, **kwargs: Probe())
    image = tmp_path / "image.png"
    Image.fromarray(np.zeros((17, 23, 3), np.uint8)).save(image)
    manifest = tmp_path / "test.csv"
    manifest.write_text("id,image\ncase,image.png\n")
    checkpoint = tmp_path / "native.pt"
    torch.save(dict(architecture="native_sam2_hiera_l", model=Probe().state_dict(),
                    config=dict(modules=dict(scds=False, sbca=False, rcsr=False)), epoch=20),
               checkpoint)
    out = tmp_path / "output"
    assert predict_native(checkpoint, manifest, out, "cpu")["count"] == 1
    with (out / "predictions.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["id"] == "case"
    with Image.open(out / row["prediction"]) as im:
        assert im.size == (23, 17)
    with pytest.raises(FileExistsError):
        predict_native(checkpoint, manifest, out, "cpu")


def test_probe_reports_native_attention_as_native_not_sbca(tmp_path, monkeypatch):
    import medsam_research.native_sam2_workflow as workflow

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.transformer = nn.Module()
            layer = nn.Module()
            layer.cross_attn_token_to_image = nn.Conv2d(3, 1, 1)
            self.decoder.transformer.layers = nn.ModuleList([layer])

        def forward(self, x):
            return self.decoder.transformer.layers[0].cross_attn_token_to_image(x)

        def auxiliary_loss(self, target):
            return target.new_zeros(())

    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(np.zeros((17, 23, 3), np.uint8)).save(image)
    y = np.zeros((17, 23), np.uint8)
    y[3:10, 5:15] = 255
    Image.fromarray(y).save(mask)
    manifest = tmp_path / "train.csv"
    manifest.write_text("id,image,mask\ncase,image.png,mask.png\n")
    monkeypatch.setattr(workflow, "load_native_config", lambda _: dict(
        train_manifest=str(manifest), device="cpu", modules=dict(scds=False, sbca=False, rcsr=False)))
    monkeypatch.setattr(workflow, "build_native_model", lambda *args, **kwargs: Probe())
    result = workflow.probe_native("unused", steps=2, batch_size=1)
    assert len(result["losses"]) == 2
    assert result["gradient_norms"]["native"] > 0
    assert "sbca" not in result["gradient_norms"]
