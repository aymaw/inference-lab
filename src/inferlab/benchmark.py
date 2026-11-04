"""In-process ASGI measurements. These are not GPU, network, or production load tests."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

from .app import create_app


async def benchmark(model_dir, requests=200, concurrency=16):
    app = create_app(model_dir)
    gate = asyncio.Semaphore(concurrency)
    latencies, codes = [], []
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            # Exclude warmup from time and report both counters explicitly.
            await client.post("/v1/predict", json={"features": [0.2] * 8})
            started = time.perf_counter()

            async def call():
                async with gate:
                    begin = time.perf_counter()
                    response = await client.post("/v1/predict", json={"features": [0.2] * 8})
                    latencies.append(1000 * (time.perf_counter() - begin))
                    codes.append(response.status_code)

            await asyncio.gather(*(call() for _ in range(requests)))
            elapsed = time.perf_counter() - started
            counters = (await client.get("/metrics")).json()
    return {
        "scope": "in-process ASGI on CPU",
        "requests": requests,
        "concurrency": concurrency,
        "successes": codes.count(200),
        "seconds": elapsed,
        "requests_per_second": requests / elapsed,
        "latency_ms_p50_p95_p99": np.quantile(latencies, [0.5, 0.95, 0.99]).tolist(),
        "counters_including_one_warmup": counters,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="runs/model")
    p.add_argument("--out", default="runs/benchmark.json")
    args = p.parse_args()
    result = asyncio.run(benchmark(args.model_dir))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
