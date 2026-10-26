from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "kaggle_remote_infer.py"
RESPONSE_ROOT = ROOT / "artifacts" / "live_inference"
VERIFIED_PATH = ROOT / "data" / "verified_live_example.json"
LOG_PATH = RESPONSE_ROOT / "ui_background_kaggle.log"


def _find_response_path(*, since: float | None = None) -> Path | None:
    if not RESPONSE_ROOT.exists():
        return None
    matches = sorted(
        RESPONSE_ROOT.rglob("response.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if since is not None:
        matches = [p for p in matches if p.stat().st_mtime >= since - 1.0]
    return matches[0] if matches else None


def load_latest_response(*, since: float | None = None) -> dict | None:
    path = _find_response_path(since=since)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_verified_example() -> dict | None:
    if not VERIFIED_PATH.exists():
        return None
    try:
        return json.loads(VERIFIED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _build_command(
    *,
    dataset: str,
    question: str | None,
    reference: str | None,
    benchmark_example_id: str | None,
    request_id: str,
    timeout_minutes: int,
) -> list[str]:
    command = [
        sys.executable,
        str(LAUNCHER),
        "--dataset",
        dataset,
        "--id",
        request_id,
        "--timeout-minutes",
        str(timeout_minutes),
    ]

    if benchmark_example_id:
        command += ["--benchmark-example-id", benchmark_example_id]
    else:
        if not question or not question.strip():
            raise ValueError("Question is required for custom inference.")
        command += ["--question", question.strip()]
        if reference and reference.strip():
            command += ["--reference", reference.strip()]

    return command


def start_live_inference(
    *,
    dataset: str,
    question: str | None = None,
    reference: str | None = None,
    benchmark_example_id: str | None = None,
    request_id: str = "ui_live_request",
    timeout_minutes: int = 40,
) -> dict:
    if not LAUNCHER.exists():
        return {
            "ok": False,
            "error": f"Kaggle launcher not found: {LAUNCHER}",
        }

    try:
        command = _build_command(
            dataset=dataset,
            question=question,
            reference=reference,
            benchmark_example_id=benchmark_example_id,
            request_id=request_id,
            timeout_minutes=timeout_minutes,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    RESPONSE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Start a fresh UI orchestration log. The Kaggle launcher itself owns the remote job.
    handle = LOG_PATH.open("w", encoding="utf-8", errors="replace")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=creationflags,
        )
    except OSError as exc:
        handle.close()
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "ok": True,
        "process": process,
        "_log_handle": handle,
        "log_path": str(LOG_PATH),
        "started_at": time.time(),
        "request_id": request_id,
        "finished": False,
    }


def _read_log(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _stage_from_log(log: str) -> str:
    lower = log.lower()
    if "remote response" in lower:
        return "response downloaded"
    if "kernelworkerstatus.complete" in lower:
        return "Kaggle run complete; retrieving result"
    if "kernelworkerstatus.running" in lower:
        return "GPU runtime active"
    if "kernel version" in lower and "successfully pushed" in lower:
        return "kernel submitted; waiting for GPU"
    if "adaptive latent mas" in lower:
        return "preparing request"
    return "starting remote job"


def _elapsed_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def poll_live_inference(job: dict) -> dict:
    process = job.get("process")
    if process is None:
        return {
            "state": "error",
            "error": "Background process handle is unavailable.",
            "log_tail": "",
        }

    log = _read_log(job.get("log_path", LOG_PATH))
    tail = "\n".join(log.splitlines()[-30:])
    elapsed = time.time() - float(job.get("started_at", time.time()))
    returncode = process.poll()

    if returncode is None:
        return {
            "state": "running",
            "stage": _stage_from_log(log),
            "elapsed_text": _elapsed_text(elapsed),
            "log_tail": tail,
        }

    handle = job.get("_log_handle")
    if handle is not None and not getattr(handle, "closed", True):
        try:
            handle.close()
        except OSError:
            pass

    response = load_latest_response(since=float(job.get("started_at", 0)))

    if returncode == 0 and response and response.get("status") == "SUCCESS":
        return {
            "state": "complete",
            "response": response,
            "elapsed_text": _elapsed_text(elapsed),
            "log_tail": tail,
        }

    if response and response.get("status") == "ERROR":
        message = (
            f"{response.get('error_type', 'Remote error')}: "
            f"{response.get('error_message', 'Unknown remote failure')}"
        )
    elif returncode != 0:
        message = f"Kaggle launcher exited with code {returncode}."
    else:
        message = "Kaggle job ended but a fresh response.json was not found."

    return {
        "state": "error",
        "error": message,
        "elapsed_text": _elapsed_text(elapsed),
        "log_tail": tail,
        "response": response,
    }
