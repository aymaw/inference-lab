# Serving choices

| System | Relevant capability | Evidence in this repository |
|---|---|---|
| FastAPI + PyTorch | Explicit model lifecycle, validation, custom batching | Executed CPU service/tests and in-process load experiment |
| vLLM | Generative-model serving, continuous batching, paged KV memory | HTTP client contract mock-tested; GPU serving not executed |
| Kubernetes | Replicas, probes, resource limits, rolling workloads | Authored local manifest; no cluster deployment |
| TorchServe | Model archives, handlers, inference/management endpoints | Comparison only; no deployment |

The [TorchServe upstream notice](https://github.com/pytorch/serve/blob/master/README.md)
states that active maintenance and planned security fixes have ended. It is not
used for a new demonstration. A model archive/handler workflow is different from
training and exporting weights, and neither establishes deployment experience.

This project's static MLP microbatcher concatenates independent vectors. vLLM's
continuous batching schedules token generation with differing sequence lengths
and manages KV cache; the two should not be described as equivalent.
