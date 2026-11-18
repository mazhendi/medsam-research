import argparse
import json

from .config import load_config
from .data import audit_manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAM 2.1 automatic binary segmentation baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    retina = sub.add_parser("prepare-retina", help="Convert DRIVE/STARE/CHASE_DB1 to paired PNGs")
    retina.add_argument("--dataset", required=True, choices=["DRIVE", "STARE", "CHASE_DB1"])
    retina.add_argument("--source", nargs="+", help="ZIP/TAR files or extracted directories")
    retina.add_argument("--output", required=True)
    retina.add_argument("--observer", type=int, choices=[1, 2], default=1)
    retina.add_argument("--seed", type=int, default=42)
    download = sub.add_parser("download-retina", help="Fetch STARE from its original public URLs")
    download.add_argument("--dataset", required=True, choices=["STARE"])
    download.add_argument("--output", required=True)
    manifest = sub.add_parser("manifest", help="Pair images and masks; does not split data")
    manifest.add_argument("--images", required=True)
    manifest.add_argument("--masks", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--mask-suffix", default="")
    audit = sub.add_parser("audit", help="Check binary masks, duplicate images and known groups")
    audit.add_argument("manifests", nargs="+")
    audit.add_argument("--mask-threshold", type=int, default=0)
    train = sub.add_parser("train", help="Train task tokens and mask decoder")
    train.add_argument("--config", required=True)
    for name in ("evaluate", "predict", "diagnose"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--checkpoint", required=True, help="Project best.pt or last.pt")
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True, help="Must not already exist")
        if name == "diagnose":
            command.add_argument("--grid", type=int, default=2)
            command.add_argument("--overlap", type=float, default=0.25)
            command.add_argument("--blend", type=float, default=0.5)
            command.add_argument("--thin-width", type=float, default=4.0)
            command.add_argument("--medium-width", type=float, default=8.0)
    correction = sub.add_parser("train-correction", help="Train shared detail/context correction")
    correction.add_argument("--config", required=True)
    correction.add_argument("--checkpoint", required=True, help="Existing baseline best.pt")
    correction.add_argument("--output", required=True)
    correction.add_argument("--epochs", type=int, default=20)
    correction.add_argument("--grid", type=int, default=2)
    correction.add_argument("--head-size", type=int, default=128)
    correction.add_argument("--learning-rate", type=float, default=0.001)
    correction.add_argument("--preservation", type=float, default=0.1)
    granularity = sub.add_parser("train-granularity", help="Train regional supervision prototype")
    granularity.add_argument("--config", required=True)
    granularity.add_argument("--checkpoint", required=True, help="Existing baseline best.pt")
    granularity.add_argument("--output", required=True)
    granularity.add_argument("--mode", choices=["pixel", "fixed", "curriculum", "meta", "reweight"],
                             default="meta")
    granularity.add_argument("--epochs", type=int, default=50)
    granularity.add_argument("--grid", type=int, default=8)
    granularity.add_argument("--learning-rate", type=float, default=None,
                             help="Default: Adam 0.001, SGD 0.01")
    granularity.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    granularity.add_argument("--controller-lr", type=float, default=0.001)
    granularity.add_argument("--minimum-pixel-weight", type=float, default=0.1)
    granular_eval = sub.add_parser("evaluate-granularity", help="Evaluate fixed adapter")
    granular_eval.add_argument("--config", required=True)
    granular_eval.add_argument("--checkpoint", required=True, help="Original baseline best.pt")
    granular_eval.add_argument("--adapter", required=True, help="best_granularity.pt")
    granular_eval.add_argument("--manifest", required=True)
    granular_eval.add_argument("--output", required=True)
    probe = sub.add_parser("probe-adapter", help="Training-only small-set fitting diagnostic")
    probe.add_argument("--config", required=True)
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--output", required=True)
    probe.add_argument("--steps", type=int, default=100)
    probe.add_argument("--samples", type=int, default=2)
    probe.add_argument("--sgd-rates", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    probe.add_argument("--adam-lr", type=float, default=0.001)
    backbone = sub.add_parser("train-backbone-control", help="Pure supervised backbone controls")
    backbone.add_argument("--config", required=True)
    backbone.add_argument("--checkpoint", required=True, help="Original baseline best.pt")
    backbone.add_argument("--output", required=True)
    backbone.add_argument("--mode", choices=["decoder", "deep", "pooled", "multiscale"],
                          default="multiscale")
    backbone.add_argument("--epochs", type=int, default=50)
    backbone.add_argument("--learning-rate", type=float, default=None)
    backbone.add_argument("--width", type=int, default=16)
    backbone_eval = sub.add_parser("evaluate-backbone-control", help="Evaluate fixed control")
    backbone_eval.add_argument("--config", required=True)
    backbone_eval.add_argument("--checkpoint", required=True, help="Original baseline best.pt")
    backbone_eval.add_argument("--control", required=True, help="best_control.pt")
    backbone_eval.add_argument("--manifest", required=True)
    backbone_eval.add_argument("--output", required=True)
    bundle = sub.add_parser("prepare-biomedclip", help="Download pinned official text/image weights")
    bundle.add_argument("--output", required=True)
    for name in ("train-text-localizer", "train-text-repair"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--checkpoint", required=True, help="Original SAM2 baseline best.pt")
        command.add_argument("--biomedclip-dir", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--epochs", type=int, default=30)
        command.add_argument("--learning-rate", type=float, default=0.001)
        if name == "train-text-localizer":
            command.add_argument("--prompt", default="retinal blood vessels")
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--prompt-map", help="Complete per-image JSON id-to-text mapping")
            source.add_argument("--task-only", action="store_true", help="Explicit uniform-text control")
            command.add_argument("--prompt-pairing", choices=["matched", "shuffle"], default="matched")
            command.add_argument("--width", type=int, default=16)
        else:
            command.add_argument("--localizer", required=True)
            command.add_argument("--repair-mode", choices=["full", "no-intervention", "no-direction", "residual", "balanced-residual"], default="full")
            command.add_argument("--direction-weight", type=float, default=0.5)
            command.add_argument("--consistency-weight", type=float, default=0.1)
    for name in ("evaluate-text-model", "predict-text-model"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--biomedclip-dir", required=True)
        command.add_argument("--localizer", required=True)
        command.add_argument("--repair", help="Optional fixed repair checkpoint")
        command.add_argument("--prompt-map", help="Per-image captions for new evaluation/inference images")
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True)
    captions = sub.add_parser("generate-image-prompts", help="Local VLM descriptions from original images only")
    captions.add_argument("--manifests", nargs="+", required=True)
    captions.add_argument("--output", required=True)
    captions.add_argument("--task", required=True)
    captions.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    captions.add_argument("--revision")
    captions.add_argument("--device", default="cuda")
    captions.add_argument("--cache-dir", default="checkpoints/caption-models")
    captions.add_argument("--max-new-tokens", type=int, default=256)
    captions.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate-image-prompts":
        from .image_captions import generate_image_prompts

        print(json.dumps(generate_image_prompts(args.manifests, args.output, args.task,
                         args.model_id, args.revision, args.device, args.cache_dir,
                         args.max_new_tokens, args.resume), indent=2))
        return
    if args.command == "prepare-biomedclip":
        from .text_evidence import prepare_biomedclip

        print(json.dumps(prepare_biomedclip(args.output), indent=2))
        return
    if args.command == "prepare-retina":
        from .retina import prepare_retina

        report = prepare_retina(args.dataset, args.source, args.output, args.observer, args.seed)
        print(json.dumps(report, indent=2))
        return
    if args.command == "download-retina":
        from .retina import download_stare

        print(json.dumps(download_stare(args.output), indent=2))
        return
    if args.command == "manifest":
        from .manifest import create_manifest

        count = create_manifest(
            args.images, args.masks, args.output, args.dataset, args.mask_suffix
        )
        print(f"Wrote {count} pairs. Fill group metadata before making train/validation splits.")
        return
    if args.command == "audit":
        print(json.dumps(audit_manifests(args.manifests, args.mask_threshold), indent=2))
        return
    from . import engine

    cfg = load_config(args.config)
    if args.command == "train-text-localizer":
        from .text_training import train_text_model

        train_text_model(cfg, args.checkpoint, args.biomedclip_dir, args.output,
                         prompt=args.prompt, prompt_map=args.prompt_map, epochs=args.epochs,
                         learning_rate=args.learning_rate, width=args.width, task_only=args.task_only,
                         prompt_pairing=args.prompt_pairing)
    elif args.command == "train-text-repair":
        from .text_training import train_text_model

        train_text_model(cfg, args.checkpoint, args.biomedclip_dir, args.output, stage="repair",
                         localizer_path=args.localizer, epochs=args.epochs,
                         learning_rate=args.learning_rate, repair_mode=args.repair_mode,
                         direction_weight=args.direction_weight, consistency_weight=args.consistency_weight)
    elif args.command in ("evaluate-text-model", "predict-text-model"):
        from . import text_training

        function = (text_training.evaluate_text_model if args.command == "evaluate-text-model"
                    else text_training.predict_text_model)
        function(cfg, args.checkpoint, args.biomedclip_dir, args.localizer, args.manifest,
                 args.output, args.repair, args.prompt_map)
    elif args.command == "train-backbone-control":
        from .backbone_controls import train_backbone_control

        train_backbone_control(cfg, args.checkpoint, args.output, args.mode, args.epochs,
                               args.learning_rate, args.width)
    elif args.command == "evaluate-backbone-control":
        from .backbone_controls import evaluate_backbone_control

        evaluate_backbone_control(cfg, args.checkpoint, args.control, args.manifest, args.output)
    elif args.command == "probe-adapter":
        from .adapter_probe import probe_adapter

        probe_adapter(cfg, args.checkpoint, args.output, args.steps, args.samples,
                      args.sgd_rates, args.adam_lr)
    elif args.command == "train-granularity":
        from .granularity import train_granularity

        train_granularity(cfg, args.checkpoint, args.output, args.mode, args.epochs, args.grid,
                          args.learning_rate, args.controller_lr, args.minimum_pixel_weight, args.optimizer)
    elif args.command == "evaluate-granularity":
        from .granularity import evaluate_granularity

        evaluate_granularity(cfg, args.checkpoint, args.adapter, args.manifest, args.output)
    elif args.command == "train-correction":
        from .correction import train_correction

        train_correction(cfg, args.checkpoint, args.output, args.epochs, args.grid,
                         args.head_size, args.learning_rate, args.preservation)
    elif args.command == "diagnose":
        from .diagnostics import diagnose

        diagnose(cfg, args.checkpoint, args.manifest, args.output, args.grid, args.overlap,
                 args.blend, args.thin_width, args.medium_width)
    elif args.command == "train":
        engine.train(cfg)
    else:
        getattr(engine, args.command)(cfg, args.checkpoint, args.manifest, args.output)


if __name__ == "__main__":
    main()
