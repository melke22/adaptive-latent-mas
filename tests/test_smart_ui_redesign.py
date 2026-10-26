import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smart_ui_theme_exists():
    text = (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")
    assert 'base = "light"' in text
    assert 'primaryColor = "#0D5C80"' in text


def test_verified_live_example_is_real_and_complete():
    payload = json.loads(
        (ROOT / "data/verified_live_example.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "SUCCESS"
    assert payload["prediction"] == "320"
    assert payload["correct"] is True
    assert payload["total_latent_steps"] == 50
    assert payload["agents"]["planner"]["latent_steps"] == 20
    assert payload["agents"]["critic"]["latent_steps"] == 20
    assert payload["agents"]["solver"]["latent_steps"] == 10


def test_live_page_has_instant_and_fresh_modes():
    text = (ROOT / "ui/app.py").read_text(encoding="utf-8")
    assert "Verified demo — instant" in text
    assert "Fresh Kaggle run" in text
    assert "Run on Kaggle GPU" in text
    assert "run_every=3" in text


def test_live_service_is_background_nonblocking():
    text = (ROOT / "ui/services/kaggle_live.py").read_text(encoding="utf-8")
    assert "subprocess.Popen" in text
    assert "poll_live_inference" in text
    assert "process.poll()" in text
