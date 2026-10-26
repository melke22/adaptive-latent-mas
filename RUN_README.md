# How to run the project

The dashboard runs locally with Python 3.11 and Streamlit. The bundled research
results and saved inference demo work without a GPU or Kaggle credentials.
Fresh inference is optional and runs remotely on Kaggle.

## 1. Open a terminal in the project folder

Open the `Adaptive_Latent_MAS` folder in your terminal. All commands below must
run from this folder, where `requirements.txt` and the `ui/` directory are located.

## 2. First-time setup on Windows (PowerShell)

Install Python 3.11 if it is not already available, then run:

```powershell
# Check that Python 3.11 is installed.
py -3.11 --version

# Create a local virtual environment.
py -3.11 -m venv .venv

# Install the project dependencies.
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the `py` launcher is unavailable but `python --version` reports Python 3.11,
use `python -m venv .venv` to create the environment instead.
Installing dependencies requires internet access. These commands do not require
activating the virtual environment or changing PowerShell's execution policy.

## 3. Start the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Open **http://localhost:8501** in your browser, or use the local URL printed in
the terminal. Keep the terminal running while using the application.

- Explore the **Research Dashboard** and other pages to view the bundled results.
- Open **Live Kaggle Inference** and select **Verified demo — instant** to view
  the saved inference example. This does not submit a new job.

Press **Ctrl+C** in the terminal to stop the application. To run it again, open
a terminal in the project folder and repeat the start command above. You only
need to install dependencies again if they change or you recreate `.venv`.

### Alternative: use the provided PowerShell scripts

With Python 3.11 configured as your default Python, you can use:

```powershell
# Allow scripts in this terminal session only.
Set-ExecutionPolicy -Scope Process Bypass

# Run once to create the environment and install dependencies.
.\scripts\bootstrap_windows.ps1

# Run whenever you want to start the dashboard.
.\scripts\run_ui.ps1
```

The bootstrap script uses the default `py` or `python` interpreter. Use the
explicit `py -3.11` setup above if you have several Python versions installed.

## Linux or macOS

With Python 3.11 installed, run from the project folder:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run ui/app.py
```

Open **http://localhost:8501**. Press **Ctrl+C** to stop. For subsequent runs,
only the final command is needed.

## Optional: run fresh inference on Kaggle

Fresh inference requires Kaggle authentication, available remote GPU capacity,
and access to the external Phase-8 research backend, trained controller, and
alignment weights. These artifacts are not bundled in this repository.

Follow the [Kaggle configuration instructions](README.md#optional-kaggle-inference)
to configure credentials, the kernel, and dataset access. The application does
not automatically load `.env` files.

Then check the connection from PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_check.py
```

After it reports `KAGGLE_CONNECTION_OK`, start the dashboard and select
**Live Kaggle Inference → Fresh Kaggle run** to submit a question. Authentication
alone does not verify access to the required research artifacts. Run one fresh
job at a time; queueing and model loading can take several minutes.

## Check the local installation

Run the existing checks from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux or macOS, use `.venv/bin/python -m pytest -q`. These checks do not
submit a Kaggle GPU job.

For code maintenance, install `requirements-dev.txt` and run the lint and
formatting checks documented in [Validation](README.md#validation).

## Troubleshooting

| Problem | What to do |
|---|---|
| `py` or `python` is not recognized | Install Python 3.11, ensure its launcher or executable is available on PATH, and reopen the terminal. |
| `.venv\Scripts\python.exe` is missing | Complete the first-time setup from the project folder. |
| `No module named streamlit` or another missing dependency | Rerun `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. |
| Editor warns that imports cannot be resolved | Select `.venv\Scripts\python.exe` with **Python: Select Interpreter** in VS Code. |
| PowerShell blocks a `.ps1` script | Use the direct Python commands above, or the process-scoped policy shown with the scripts. |
| Port 8501 is already in use | Run `.\.venv\Scripts\python.exe -m streamlit run ui/app.py --server.port 8502`, then open `http://localhost:8502`. |
| The browser does not open automatically | Open the local URL printed by Streamlit manually. |
| A CSV or saved demo file is missing | Restore the bundled `data/` folder from the project checkout. |
| Fresh Kaggle inference fails | Check authentication, kernel configuration, dataset permissions, remote logs, and GPU availability using the main README's Kaggle instructions. |

For research details, results, and architecture, see the [main README](README.md).
