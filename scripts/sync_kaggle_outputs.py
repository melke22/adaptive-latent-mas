
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Download outputs from a Kaggle kernel.")
parser.add_argument("--kernel", required=True, help="owner/kernel-slug or owner/kernel-slug/version")
parser.add_argument("--destination", default=str(ROOT / "artifacts/kaggle"))
parser.add_argument("--pattern", default=None, help="Optional Kaggle output regex file pattern")
args = parser.parse_args()

exe = shutil.which("kaggle")
if not exe:
    raise SystemExit("Kaggle CLI not found. Install with: python -m pip install kaggle")

dest = Path(args.destination)
dest.mkdir(parents=True, exist_ok=True)

cmd = [exe, "kernels", "output", args.kernel, "-p", str(dest), "-o"]
if args.pattern:
    cmd += ["--file-pattern", args.pattern]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Downloaded Kaggle outputs to:", dest)
