import asyncio
import json

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from inferlab.app import create_app
from inferlab.batching import MicroBatcher, Overloaded
from inferlab.train import architecture, train
from inferlab.vllm_client import complete


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("model")
    train(path)
    return path


def test_validation_version_and_prediction(model_dir):
    with TestClient(create_app(model_dir)) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.post("/v1/predict", json={"features": [0.0] * 7}).status_code == 422
        assert client.post("/v1/predict", json={"features": [99.0] * 8}).status_code == 422
        assert (
            client.post("/v1/predict", json={"features": [0.0] * 8, "extra": 1}).status_code == 422
        )
        assert (
            client.post(
                "/v1/predict", json={"features": [0.0] * 8, "expected_version": "wrong"}
            ).status_code
            == 409
        )
        response = client.post("/v1/predict", json={"features": [0.0] * 8}).json()
        assert sum(response["probabilities"]) == pytest.approx(1.0)
        assert response["model_version"] == "synthetic-mlp-v1"


def test_checksum_fails_closed(model_dir, tmp_path):
    (tmp_path / "manifest.json").write_bytes((model_dir / "manifest.json").read_bytes())
    (tmp_path / "weights.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        with TestClient(create_app(tmp_path)):
            pass


def test_batching_preserves_order_and_matches_direct():
    async def scenario():
        torch.manual_seed(4)
        model = architecture().eval()
        batcher = MicroBatcher(model, batch_size=8)
        await batcher.start()
        rows = torch.randn(20, 8)
        output = await asyncio.gather(*(batcher.predict(row.tolist()) for row in rows))
        assert torch.allclose(torch.tensor(output), model(rows).softmax(-1), atol=1e-6)
        assert batcher.batches == 3 and batcher.examples == 20
        await batcher.close()
        with pytest.raises(Overloaded):
            await batcher.predict([0.0] * 8)

    asyncio.run(scenario())


def test_overload_and_cancellation():
    async def scenario():
        batcher = MicroBatcher(architecture(), capacity=1, wait=0.1)
        # Before worker starts, admission must reject the second queued request.
        first = asyncio.create_task(batcher.predict([0.0] * 8))
        await asyncio.sleep(0)
        with pytest.raises(Overloaded):
            await batcher.predict([0.0] * 8)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await batcher.start()
        await asyncio.sleep(0.01)
        await batcher.close()

    asyncio.run(scenario())


def test_mock_vllm_request_contract():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["max_tokens"] == 64 and body["temperature"] == 0
        return httpx.Response(200, json={"choices": [{"message": {"content": "synthetic reply"}}]})

    assert (
        asyncio.run(
            complete(
                "http://local", "test", "fictional prompt", "test", httpx.MockTransport(handler)
            )
        )
        == "synthetic reply"
    )


def test_mock_vllm_propagates_http_failure():
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            complete(
                "http://local",
                "test",
                "fictional prompt",
                "test",
                httpx.MockTransport(lambda r: httpx.Response(503)),
            )
        )
