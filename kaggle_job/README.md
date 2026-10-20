
# Kaggle Remote Compute

The local UI should not pretend Kaggle is a real-time inference server.

Recommended use:

1. The UI runs locally.
2. Heavy inference requests are packaged as Kaggle kernel jobs.
3. `kaggle kernels push` starts the remote run.
4. `kaggle kernels status` / `kaggle kernels logs --follow` monitor it.
5. `kaggle kernels output` downloads the result.
6. The UI refreshes the downloaded result.

This is appropriate for demonstration/research jobs, but the latency is job-level rather than
interactive web-service latency.

The next implementation step is to convert the verified single-question Phase-9 inference path
into a dedicated Kaggle script kernel that reads `request.json` and writes `response.json`.
