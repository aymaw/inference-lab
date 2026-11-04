# Inference Lab

A small model-serving reference with bounded admission, microbatching, versioned predictions, and container packaging.

[![CI](https://github.com/aymaw/inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/aymaw/inference-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
![Data](https://img.shields.io/badge/data-entirely_synthetic-teal)
[![License: MIT](https://img.shields.io/badge/license-MIT-black)](LICENSE)

**New personal engineering work, November 2026.** Original, AI-assisted portfolio
implementation with synthetic task data. This is not historical research code,
employer software, or a deployed product. See [provenance](PROVENANCE.md).

## Implemented behavior

A seeded PyTorch MLP learns a synthetic eight-feature, three-class task. The
FastAPI service loads its weights once during lifespan startup, checks a SHA-256
manifest, accepts finite bounded vectors, verifies the requested model version,
and batches concurrent requests. The queue has a fixed capacity; full admission
or deadlines return 503. Cancellation and shutdown have explicit handling.

```mermaid
flowchart LR
    R[Validated request] --> V[Version check]
    V --> Q[Bounded queue]
    Q --> B[Microbatch worker]
    B --> M[PyTorch inference mode]
    M --> P[Versioned probabilities]
```
## Reproduce

```bash
# Python 3.11–3.13; uv.lock pins the complete dependency graph.
uv sync --frozen --extra dev
uv run pytest -q
```

Alternatively, `python -m pip install -e '.[dev]'` installs the exact direct
dependencies from `pyproject.toml`, while resolving transitive dependencies.
CPU is the reference execution device. Full runs and checkpoints go to ignored
`runs/`; committed `results/` contains measured reports and provenance only.

```bash
uv run python -m inferlab.train
uv run uvicorn inferlab.app:app --host 127.0.0.1 --port 8000 --no-access-log
# In another terminal:
curl http://127.0.0.1:8000/v1/predict   -H 'Content-Type: application/json'   -d '{"features":[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],"expected_version":"synthetic-mlp-v1"}'
```

```bash
docker build -t inference-lab:local .
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges   -p 127.0.0.1:8000:8000 inference-lab:local
```

The image trains only its fictional toy task at build time, runs as UID 10001,
and has a readiness health check. No model download or training occurs on an
inference request. `/health/live`, `/health/ready`, and `/metrics` distinguish
process availability, model availability, and batching counters. Request vectors
are not logged; no remote model-upload or model-management endpoint is exposed.

## Measured CPU benchmark

`uv run python -m inferlab.benchmark` measures **in-process ASGI**, excluding one
warmup request. The committed [report](results/benchmark.json) covers 200 requests,
concurrency 16, all 200 successful. These measurements exclude sockets, proxying,
TLS, authentication, and GPU execution; they are not production capacity claims.
Timing depends on the host. [The training manifest](results/model-manifest.json)
records the actual synthetic held-out classifier accuracy and checkpoint checksum.

## Kubernetes and GPU integration

[A local-cluster manifest](deploy/kubernetes.yaml) demonstrates replicas,
ClusterIP networking, probes, CPU/memory requests and limits, and non-root
read-only containers. It uses `imagePullPolicy: Never` and the local image;
it must be loaded into an isolated development cluster first. No cluster or
cloud resources are created automatically. This example has not been deployed
to Kubernetes and does not demonstrate operating a production cluster.

The optional [vLLM client](src/inferlab/vllm_client.py) uses the OpenAI-compatible
chat API with bounded output tokens and HTTP failure propagation. Its request
contract is mock-tested. [GPU setup notes](deploy/VLLM.md) explain what to run on
a compatible GPU host and what remains unverified. This CPU service uses a
standard MLP and does **not** implement paged KV cache or continuous LLM batching.

TorchServe is covered in [the serving decision note](deploy/SERVING.md), including
its upstream limited-maintenance status. It is not falsely claimed as deployed.

## Deliberate limits

This is a local engineering example, not a production deployment. Before external
use it would need authenticated ingress, measured SLOs, structured error metrics,
distributed tracing, load tests through the actual network path, model governance,
artifact signing, and platform-specific review. The checksum detects corruption;
it does not authenticate an untrusted manifest. Only locally generated trusted
weights are loaded with `weights_only=True`.

## References

- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/).
- [Kubernetes probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/).
- [vLLM Docker serving](https://docs.vllm.ai/en/latest/deployment/docker/).

Original code and synthetic data: [MIT](LICENSE).
