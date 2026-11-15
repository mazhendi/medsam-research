import csv
import json

import pytest
import torch
from PIL import Image
from test_data_metrics import make_manifest
from torch import nn

from medsam_research import engine
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig


class TinySegmenter(nn.Module):
    """Test fixture only: exercises workflow, not SAM accuracy or API compatibility."""

    image_size = 32

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        return self.conv(x)


def test_train_checkpoint_evaluate_and_label_free_prediction(tmp_path, monkeypatch):
    train = make_manifest(tmp_path, "train.csv", ["aa", "bb"])
    val = make_manifest(tmp_path, "val.csv", ["cc"])
    test = make_manifest(tmp_path, "test.csv", ["dd"])
    pretrained = tmp_path / "source.pt"
    pretrained.write_bytes(b"test-only pretrained identity")
    cfg = Config(
        ModelConfig(checkpoint=str(pretrained)),
        DataConfig(str(train), str(val)),
        TrainingConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            amp=False,
            output_dir=str(tmp_path / "run"),
        ),
    )
    monkeypatch.setattr(engine, "build_model", lambda cfg, device: TinySegmenter().to(device))
    out = engine.train(cfg)
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    assert len((out / "history.jsonl").read_text().splitlines()) == 2
    scores = engine.evaluate(cfg, out / "best.pt", test, tmp_path / "evaluation")
    assert scores["samples"] == 1 and 0 <= scores["dice"] <= 1
    # A nonexistent mask must not be needed by the inference manifest.
    inference = tmp_path / "inference.csv"
    with inference.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "image", "mask"])
        writer.writerow(["a/path-like-id", "dd.png", "deliberately_missing_mask.png"])
    predicted = engine.predict(cfg, out / "best.pt", inference, tmp_path / "predictions")
    with Image.open(predicted / "000000.png") as image:
        assert image.size == (19, 13)
    assert json.loads((out / "provenance.json").read_text())["data_audit"][0]["samples"] == 2
    with pytest.raises(FileExistsError):
        engine.train(cfg)
    cfg.training.threshold = 0.8
    with pytest.raises(ValueError, match="differs from training"):
        engine.predict(cfg, out / "best.pt", inference, tmp_path / "wrong")


def test_changed_pretrained_weights_are_rejected(tmp_path):
    model = TinySegmenter()
    source = tmp_path / "source.pt"
    source.write_bytes(b"changed")
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(), TrainingConfig())
    from medsam_research.config import SAM2_COMMIT

    artifact = {
        "format_version": 1,
        "sam2_commit": SAM2_COMMIT,
        "model_config": {"sam_config": cfg.model.sam_config, "prompt_tokens": 4},
        "pretrained_sha256": "different",
        "adapter": {},
    }
    torch.save(artifact, tmp_path / "adapter.pt")
    with pytest.raises(ValueError, match="does not match"):
        engine.load_adapter(model, tmp_path / "adapter.pt", cfg)


@pytest.mark.integration
def test_real_sam2_forward_backward_and_batching():
    """Real Hiera-Small, random weights, reduced grid: verifies gradients, not performance."""
    pytest.importorskip("sam2")
    from sam2.build_sam import build_sam2

    from medsam_research.config import SAM2_COMMIT
    from medsam_research.metrics import segmentation_loss
    from medsam_research.model import AutoSAM2, upstream_commit

    assert upstream_commit() == SAM2_COMMIT
    torch.set_num_threads(2)
    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        ckpt_path=None,
        device="cpu",
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.image_size=64"],
    )
    model = AutoSAM2(sam, prompt_tokens=4)
    model.train()
    images = torch.rand(2, 3, 64, 64)
    logits = model(images)
    assert logits.shape == (2, 1, 64, 64)
    loss = segmentation_loss(logits, torch.zeros_like(logits))
    loss.backward()
    assert model.task_tokens.grad is not None
    assert model.task_tokens.grad.abs().sum() > 0
    assert model.sam.sam_mask_decoder.conv_s0.weight.grad is not None
    assert model.sam.sam_mask_decoder.mask_tokens.weight.grad is not None
    assert all(p.grad is None for p in model.sam.image_encoder.parameters())
    assert not model.sam.image_encoder.training
    model.eval()
    with torch.no_grad():
        batch = model(images)
        singles = torch.cat([model(x[None]) for x in images])
    torch.testing.assert_close(batch, singles, atol=1e-4, rtol=1e-4)
