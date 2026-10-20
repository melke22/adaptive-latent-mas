
# Adaptive Latent Communication for Efficient Multi-Agent LLM Reasoning

M.Tech 3rd Semester research interface for the **Planner → Critic → Solver** adaptive latent
multi-agent reasoning project.

## What is included

- A local Streamlit professor-facing UI.
- 512 real Phase-9 per-question console records recovered from the completed Kaggle run.
- Final performance, controller, and paired-statistics tables.
- Research-integrity and reproducibility views.
- Kaggle CLI connection and output-sync scripts.
- GitHub CI tests.
- Completed Phase-9 notebook preserved under `notebooks/`.

## Operating modes

### 1. Results Mode — ready now
Runs fully on the local machine and uses the real completed Phase-9 outputs.

### 2. Kaggle Remote Compute — scaffold included
The local machine uses the Kaggle CLI to authenticate, start/pull Kaggle jobs, and download
artifacts. Heavy Qwen3-4B inference remains on Kaggle GPU rather than on the laptop.

## Windows quick start

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd adaptive-latent-mas
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1
.\scripts\run_ui.ps1
```

Then open the URL printed by Streamlit, normally `http://localhost:8501`.

## Verify Kaggle

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_check.py
```

Expected final line:

```text
KAGGLE_CONNECTION_OK
```

## Security

Never commit:

- `kaggle.json`
- Kaggle API tokens
- Hugging Face tokens
- `.env` secrets
- model checkpoints
- private Kaggle downloads

The `.gitignore` already excludes these.

## Phase-9 result headline

| System | Accuracy | Mean latent steps | Mean latency |
|---|---:|---:|---:|
| Fixed-10 | 61.91% | 30.00 | 9.54 s |
| Fixed-20 | 56.25% | 60.00 | 10.54 s |
| Adaptive | **62.50%** | **49.04** | 10.31 s |

Adaptive vs Fixed-20:

- accuracy difference: **+6.25 percentage points**
- 95% bootstrap interval: **+2.15 to +10.55**
- exact McNemar p: **0.004455**
- mean latent-step reduction: **18.26%**

## Repository policy

Source code, UI, tests, documentation and compact result tables belong in GitHub.
Large model weights and large experimental outputs belong in Kaggle persistent storage or local
archives, not in ordinary Git history.
