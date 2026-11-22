"""Obtain an isolated, pinned official source checkout; does not download weights."""
import argparse
import subprocess
from pathlib import Path

REVISION = "cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4"
URL = "https://github.com/DengPingFan/Polyp-PVT.git"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path,
                        default=Path(__file__).resolve().parents[1] / "vendor/Polyp-PVT")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"],
                                       text=True).strip()
        if head != REVISION:
            raise ValueError(f"Existing checkout is {head}, expected {REVISION}; use a new directory")
        if subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=no"]):
            raise ValueError("Existing checkout has local modifications; use a new directory")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--no-checkout", URL,
                        str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "config", "core.autocrlf", "false"],
                       check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", REVISION], check=True)
    print(f"Official Polyp-PVT source: {destination}\nRevision: {REVISION}")


if __name__ == "__main__":
    main()
