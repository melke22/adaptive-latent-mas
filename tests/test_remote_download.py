"""Local download regression checks; the Kaggle CLI is replaced by a fixture."""

import json
from pathlib import Path

import pytest

from scripts import kaggle_remote_infer


@pytest.mark.parametrize("result", ["valid", "missing", "malformed"])
def test_download_preserves_log_and_previous_result(monkeypatch, tmp_path, result):
    log_path = tmp_path / "ui_background_kaggle.log"
    response_path = tmp_path / "response.json"
    response_path.write_text('{"status": "OLD"}', encoding="utf-8")
    monkeypatch.setattr(kaggle_remote_infer, "ARTIFACT_DIR", tmp_path)

    def download(*args):
        destination = Path(args[args.index("-p") + 1])
        if result == "valid":
            (destination / "response.json").write_text(
                '{"status": "SUCCESS"}', encoding="utf-8"
            )
        elif result == "malformed":
            (destination / "response.json").write_text("{", encoding="utf-8")

    monkeypatch.setattr(kaggle_remote_infer, "run_kaggle", download)
    # Keep the log open, as the UI does during a download on Windows.
    with log_path.open("w", encoding="utf-8") as log:
        log.write("active orchestration log")
        log.flush()
        if result == "valid":
            assert kaggle_remote_infer.download_response() == {"status": "SUCCESS"}
        else:
            with pytest.raises((RuntimeError, json.JSONDecodeError)):
                kaggle_remote_infer.download_response()
        log.write("\nfinished")

    assert log_path.read_text(encoding="utf-8").endswith("finished")
    expected = "SUCCESS" if result == "valid" else "OLD"
    assert json.loads(response_path.read_text(encoding="utf-8"))["status"] == expected
    assert not list(tmp_path.glob("download-*"))
