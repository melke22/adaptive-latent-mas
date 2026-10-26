from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "kaggle_job" / "live_inference"
RUNTIME_DIR = ROOT / ".kaggle_runtime" / "live_inference"
ARTIFACT_DIR = ROOT / "artifacts" / "live_inference"
KERNEL_REF = "malkee/adaptive-latent-mas-live-inference"


def run_kaggle(*args, check=True):
    cmd = [sys.executable, "-m", "kaggle", *args]
    env = os.environ.copy()
    # Prevent Windows cp1252/charmap failures when Kaggle output contains Unicode.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if p.stdout.strip():
        print(p.stdout.strip())
    if p.stderr.strip():
        print(p.stderr.strip(), file=sys.stderr)

    if check and p.returncode != 0:
        raise RuntimeError(f"Kaggle command failed ({p.returncode}): {' '.join(cmd)}")
    return p


def get_status_text():
    p = run_kaggle("kernels", "status", KERNEL_REF, check=False)
    return (p.stdout + "\n" + p.stderr).strip().lower(), p.returncode


def prepare_runtime(payload: dict):
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        TEMPLATE_DIR / "kernel-metadata.json",
        RUNTIME_DIR / "kernel-metadata.json",
    )

    template = (TEMPLATE_DIR / "remote_inference.py.template").read_text(
        encoding="utf-8"
    )
    request_json = json.dumps(payload, ensure_ascii=False)
    replacement = repr(request_json)

    marker = "__REQUEST_JSON_LITERAL__"
    if template.count(marker) != 1:
        raise RuntimeError(
            "Remote inference template placeholder is missing or duplicated."
        )

    source = template.replace(marker, replacement)
    (RUNTIME_DIR / "remote_inference.py").write_text(source, encoding="utf-8")


def wait_for_completion(timeout_minutes: int, poll_seconds: int):
    deadline = time.time() + timeout_minutes * 60

    while time.time() < deadline:
        status_text, code = get_status_text()
        print("REMOTE STATUS:", status_text or f"returncode={code}")

        if re.search(r"\bcomplete(d)?\b|\bstatus[^\n]*complete", status_text):
            return
        if re.search(r"\berror\b|\bfailed\b|\bcancel", status_text):
            raise RuntimeError(
                f"Kaggle kernel did not complete successfully: {status_text}"
            )

        time.sleep(poll_seconds)

    raise TimeoutError(f"Kaggle kernel exceeded {timeout_minutes} minute timeout.")


def download_response():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the active UI log and previous result intact while downloading.
    with tempfile.TemporaryDirectory(prefix="download-", dir=ARTIFACT_DIR) as temporary:
        download_dir = Path(temporary)
        run_kaggle(
            "kernels",
            "output",
            KERNEL_REF,
            "-p",
            str(download_dir),
            "-o",
            "--file-pattern",
            r".*response\.json$",
        )

        matches = list(download_dir.rglob("response.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one response.json; found {len(matches)}. "
                "Inspect the Kaggle kernel logs."
            )
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        response_path = ARTIFACT_DIR / "response.json"
        matches[0].replace(response_path)

    print("\n" + "=" * 80)
    print("REMOTE RESPONSE")
    print("=" * 80)
    # ASCII-safe terminal output; the JSON file itself remains UTF-8.
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    print("=" * 80)
    print("Saved locally:", response_path)
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Submit one adaptive latent MAS inference request to Kaggle GPU."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["gsm8k", "arc_challenge"],
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="Custom question to solve.")
    source.add_argument(
        "--benchmark-example-id",
        help=(
            "Pre-approved Phase-9 evaluated example ID for fidelity verification. "
            "Arbitrary test IDs are blocked to protect the sealed remainder."
        ),
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional gold answer for custom questions only.",
    )
    parser.add_argument("--id", default="live_request")
    parser.add_argument("--timeout-minutes", type=int, default=40)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()

    if args.benchmark_example_id and args.reference is not None:
        parser.error(
            "--reference must not be supplied with --benchmark-example-id; "
            "the official record provides its own reference."
        )

    payload = {
        "id": args.id,
        "dataset": args.dataset,
        "question": args.question,
        "reference_answer": args.reference,
        "benchmark_example_id": args.benchmark_example_id,
    }

    print("=" * 80)
    print("ADAPTIVE LATENT MAS — KAGGLE REMOTE JOB")
    print("=" * 80)
    print("Kernel :", KERNEL_REF)
    print("Dataset:", args.dataset)
    print("Request:", args.id)
    print(
        "Source :",
        (
            f"Phase-9 evaluated benchmark {args.benchmark_example_id}"
            if args.benchmark_example_id
            else "custom question"
        ),
    )
    print("Mode   : frozen learned adaptive 10/20 latent inference")
    print("=" * 80)

    prepare_runtime(payload)
    run_kaggle("kernels", "push", "-p", str(RUNTIME_DIR))
    wait_for_completion(args.timeout_minutes, args.poll_seconds)
    response = download_response()

    if response.get("status") != "SUCCESS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
