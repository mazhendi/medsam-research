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
    for name in ("train-region-shape", "probe-region-shape", "evaluate-region-shape",
                 "predict-region-shape", "diagnose-region-shape"):
        command = sub.add_parser(name, help="Region–shape co-inference prototype")
        command.add_argument("--config", required=True, help="Existing baseline/light YAML")
        command.add_argument("--baseline-checkpoint", required=True, help="Trained baseline best.pt")
        command.add_argument("--output", required=True, help="New output directory")
        if name in ("train-region-shape", "probe-region-shape"):
            command.add_argument("--mechanism-config", required=True)
            if name == "train-region-shape":
                command.add_argument("--epochs", type=int, default=None)
                command.add_argument("--initialize-from", help="Existing region-shape checkpoint")
            else:
                command.add_argument("--steps", type=int, default=100)
                command.add_argument("--samples", type=int, default=2)
                command.add_argument("--learning-rate", type=float, default=.001)
        else:
            command.add_argument("--checkpoint", required=True, help="best_region_shape.pt")
            command.add_argument("--manifest", required=True)
            if name == "diagnose-region-shape":
                command.add_argument("--window-size", type=int, default=8)
    args = parser.parse_args(argv)
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
    if args.command in ("train-region-shape", "probe-region-shape", "evaluate-region-shape",
                        "predict-region-shape", "diagnose-region-shape"):
        from . import region_shape_training as workflow
        from .region_shape import load_mechanism_config

        if args.command in ("train-region-shape", "probe-region-shape"):
            mechanism = load_mechanism_config(args.mechanism_config)
            if args.command == "train-region-shape":
                workflow.train_region_shape(cfg, args.baseline_checkpoint, mechanism,
                                            args.output, args.epochs, initialize_from=args.initialize_from)
            else:
                workflow.probe_region_shape(cfg, args.baseline_checkpoint, mechanism,
                    args.output, args.steps, args.samples, args.learning_rate)
        else:
            if args.command == 'diagnose-region-shape':
                from .region_shape_diagnostics import diagnose_region_shape
                print(json.dumps(diagnose_region_shape(cfg, args.baseline_checkpoint,
                    args.checkpoint, args.manifest, args.output, args.window_size), indent=2))
                return
            function = (workflow.evaluate_region_shape if args.command == "evaluate-region-shape"
                        else workflow.predict_region_shape)
            function(cfg, args.baseline_checkpoint, args.checkpoint, args.manifest, args.output)
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
