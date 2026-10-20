import subprocess
import sys


def run_kaggle(*args):
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        capture_output=True,
        text=True,
    )


print("Checking Kaggle CLI...")

p = run_kaggle("--version")

print(p.stdout.strip() or p.stderr.strip())

if p.returncode != 0:
    print("ERROR: Kaggle CLI is not available.")
    sys.exit(p.returncode)


print("\nChecking Kaggle authentication...")

p = run_kaggle(
    "kernels",
    "list",
    "-m",
    "--page-size",
    "1",
)

print(p.stdout.strip() or p.stderr.strip())

if p.returncode != 0:
    print("\nERROR: Kaggle authentication failed.")
    print("Authenticate with:")
    print("python -m kaggle auth login")
    sys.exit(p.returncode)


print("\nKAGGLE_CONNECTION_OK")