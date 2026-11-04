# Review validation

Reviewed on November 4, 2026. The local suite now totals 56 tests across the four
repositories. Regression coverage includes missing NaNs, adapter reload without
labels, template-cluster uncertainty, semantic confidence/slot guards, unlabelled
clinical inference, event-loop responsiveness, overload/cancellation, and model
failure recovery. Exact test counts per repository appear in the CI run.

Reference training and the affected evaluations were rerun. Language projects
also report a separate familiar-template slice, with its limits stated explicitly.
The serving image was built from the frozen lock and tested over real HTTP.

See the current CI badge/run for the published head; local tests alone do not
establish hosted success. Kubernetes and live GPU/vLLM serving remain unexecuted.
