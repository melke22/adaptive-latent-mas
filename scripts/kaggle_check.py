import subprocess
import sys


def run_kaggle(*args):
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main():
    print("Checking Kaggle CLI...")
    result = run_kaggle("--version")
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        print("ERROR: Kaggle CLI is not available.")
        return result.returncode

    print("\nChecking Kaggle authentication...")
    result = run_kaggle("kernels", "list", "-m", "--page-size", "1")
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        print("\nERROR: Kaggle authentication failed.")
        print("Authenticate with: python -m kaggle auth login")
        return result.returncode

    print("\nKAGGLE_CONNECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
