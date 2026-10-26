from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_recovered_phase9_record_count():
    df = pd.read_csv(ROOT / "data/phase9_recovered_console_records.csv")
    assert len(df) == 512
    assert set(df["dataset"]) == {"gsm8k", "arc_challenge"}


def test_final_phase9_summary():
    df = pd.read_csv(ROOT / "data/phase9_final_performance_summary.csv")
    combined = df[df["dataset"] == "combined"].set_index("system")
    assert int(combined.loc["adaptive", "correct_n"]) == 320
    assert abs(combined.loc["adaptive", "accuracy"] - 0.625) < 1e-9
    assert abs(combined.loc["fixed_20", "accuracy"] - 0.5625) < 1e-9


def test_integrity_fields_exist():
    df = pd.read_csv(ROOT / "data/phase9_recovered_console_records.csv")
    required = {
        "planner_decision",
        "critic_decision",
        "solver_decision",
        "planner_utility",
        "critic_utility",
        "solver_utility",
        "adaptive_steps",
        "adaptive_correct",
        "saving_vs_fixed20_pct",
    }
    assert required.issubset(df.columns)


def test_controller_decisions_are_binary():
    df = pd.read_csv(ROOT / "data/phase9_recovered_console_records.csv")
    for col in ["planner_decision", "critic_decision", "solver_decision"]:
        assert set(df[col]) <= {"STOP", "CONTINUE"}
