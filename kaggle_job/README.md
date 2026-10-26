
# Kaggle Remote Compute

Fresh inference runs as a Kaggle batch job. The local dashboard and saved
demonstration remain usable without Kaggle access.

Recommended use:

1. The UI runs locally.
2. Heavy inference requests are packaged as Kaggle kernel jobs.
3. `kaggle kernels push` starts the remote run.
4. The launcher polls `kaggle kernels status`; the UI shows the orchestration log.
5. `kaggle kernels output` downloads the result.
6. The UI refreshes the downloaded result.

This is appropriate for demonstration/research jobs, but the latency is job-level rather than
interactive web-service latency.

The implemented launcher is `scripts/kaggle_remote_infer.py`. It embeds the
request as a JSON literal in `live_inference/remote_inference.py.template`, stages
the private GPU kernel under `.kaggle_runtime/live_inference`, and submits it.
The remote script loads the frozen external Phase-8 artifacts and produces
`response.json`. Only that result is downloaded; the active UI log is preserved.

See the [main project's Kaggle instructions](../README.md#optional-kaggle-inference)
for credentials, required artifacts, approved fidelity checks, and launch commands.
Run one request at a time: the kernel and local runtime paths are shared.
