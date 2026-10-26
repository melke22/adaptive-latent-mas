"""Exercise the local presentation without submitting remote jobs."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from ui.services import kaggle_live

APP = Path(__file__).resolve().parents[1] / "ui/app.py"
PAGES = [
    "Research Dashboard",
    "Live Kaggle Inference",
    "Benchmark Explorer",
    "Controller Analysis",
    "Statistical Evidence",
    "Architecture",
    "Reproducibility",
    "Kaggle Connection",
]


@pytest.fixture
def app(monkeypatch):
    def reject_remote(*args, **kwargs):
        pytest.fail("Local UI checks must not launch external commands.")

    monkeypatch.setattr(kaggle_live, "start_live_inference", reject_remote)
    monkeypatch.setattr(kaggle_live.subprocess, "Popen", reject_remote)
    return AppTest.from_file(str(APP), default_timeout=15).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception


def test_saved_demo_renders(app):
    app.sidebar.radio[0].set_value("Live Kaggle Inference").run()
    assert not app.exception
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Prediction"] == "320"
    assert values["Latent steps"] == "50"


def test_running_job_monitor(app, tmp_path):
    app.sidebar.radio[0].set_value("Live Kaggle Inference").run()
    log_path = tmp_path / "running.log"
    log_path.write_text("KernelWorkerStatus.RUNNING", encoding="utf-8")
    app.session_state["live_background_job"] = {
        "process": Mock(poll=Mock(return_value=None)),
        "log_path": str(log_path),
        "finished": False,
    }
    app.run()
    assert not app.exception
    assert any(status.label == "Kaggle GPU job in progress" for status in app.status)
    assert any(metric.label == "Elapsed" for metric in app.metric)
