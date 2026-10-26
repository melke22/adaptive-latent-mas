import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Download outputs from a Kaggle kernel."
    )
    parser.add_argument(
        "--kernel", required=True, help="owner/kernel-slug or owner/kernel-slug/version"
    )
    parser.add_argument("--destination", default=str(ROOT / "artifacts/kaggle"))
    parser.add_argument(
        "--pattern", default=None, help="Optional Kaggle output regex file pattern"
    )
    args = parser.parse_args()

    dest = Path(args.destination)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "kernels",
        "output",
        args.kernel,
        "-p",
        str(dest),
        "-o",
    ]
    if args.pattern:
        cmd += ["--file-pattern", args.pattern]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Downloaded Kaggle outputs to:", dest)


if __name__ == "__main__":
    main()
