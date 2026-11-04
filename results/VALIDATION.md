# Validation evidence

Executed locally on macOS ARM64 with Python 3.12.9, CPU only.
Frozen dependency installation, unit tests, lint/format checks, and end-to-end
smoke training completed. Full synthetic reference runs are in `metrics.json`
(except serving, which records a classifier manifest and ASGI benchmark).

GitHub Actions runs Python 3.11–3.13 on Ubuntu. See the current hosted run
for remote status; local completion alone does not establish hosted success.

Kubernetes and live GPU/vLLM serving were not executed.
