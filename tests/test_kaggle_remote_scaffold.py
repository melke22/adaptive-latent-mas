import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_live_metadata():
    payload = json.loads(
        (ROOT / "kaggle_job/live_inference/kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["id"] == "malkee/adaptive-latent-mas-live-inference"
    assert payload["enable_gpu"] is True
    assert "malkee/adaptive-latent-mas-phase8-complete" in payload["dataset_sources"]


def test_remote_template_restricts_benchmark_verification():
    text = (ROOT / "kaggle_job/live_inference/remote_inference.py.template").read_text(
        encoding="utf-8"
    )
    assert '"gsm8k": {"gsm8k_test_00198"}' in text
    assert "sealed test remainder" in text
    assert text.count("__REQUEST_JSON_LITERAL__") == 1


def test_launcher_downloads_only_response():
    text = (ROOT / "scripts/kaggle_remote_infer.py").read_text(encoding="utf-8")
    assert "--file-pattern" in text
    assert r".*response\.json$" in text
    assert "ensure_ascii=True" in text
    assert "PYTHONIOENCODING" in text
