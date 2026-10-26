import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_live_ui_files_exist():
    assert (ROOT / "ui/services/kaggle_live.py").exists()
    assert (ROOT / "scripts/kaggle_remote_infer.py").exists()


def test_ui_contains_live_page():
    text = (ROOT / "ui/app.py").read_text(encoding="utf-8")
    assert '"Live Kaggle Inference"' in text
    assert "Run on Kaggle GPU" in text
    assert "gsm8k_test_00198" in text


def test_remote_kernel_private_gpu():
    payload = json.loads(
        (ROOT / "kaggle_job/live_inference/kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["is_private"] is True
    assert payload["enable_gpu"] is True
