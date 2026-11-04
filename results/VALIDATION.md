# Review validation

Reviewed on November 4, 2026. This repository has 10 local tests; the four-repository suite
now totals 57 tests. Regression coverage includes missing NaNs, adapter reload without
labels, template-cluster uncertainty, semantic confidence/slot guards, unlabelled
clinical inference, event-loop responsiveness, overload/cancellation, and model
failure recovery. Exact test counts per repository appear in the CI run.

Reference training and the affected evaluations were rerun. Language projects
also report a separate familiar-template slice, with its limits stated explicitly.
The serving image was built from the frozen lock and tested over real HTTP.

See the current CI badge/run for the published head; local tests alone do not
establish hosted success. Kubernetes and live GPU/vLLM serving remain unexecuted.

The hosted startup probe also retries transient TCP resets while the server
starts; a regression test reproduces that condition. Prediction failures are
not retried or suppressed by this readiness loop.
