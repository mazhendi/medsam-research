import argparse
import json

from .config import load_config
from .data import audit_manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAM 2.1 automatic binary segmentation baseline")
    sub = parser.add_subparsers(dest="command", required=True)
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
    for name in ("evaluate", "predict"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--checkpoint", required=True, help="Project best.pt or last.pt")
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True, help="Must not already exist")
    args = parser.parse_args(argv)
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
    if args.command == "train":
        engine.train(cfg)
    else:
        getattr(engine, args.command)(cfg, args.checkpoint, args.manifest, args.output)


if __name__ == "__main__":
    main()
