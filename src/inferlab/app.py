import asyncio
import hashlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .batching import InferenceFailed, MicroBatcher, Overloaded
from .train import architecture

FiniteFeature = Annotated[float, Field(ge=-20, le=20, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: Annotated[list[FiniteFeature], Field(min_length=8, max_length=8)]
    expected_version: str | None = None


def create_app(model_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(1)
        directory = Path(model_dir or os.getenv("MODEL_DIR", "runs/model"))
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / "weights.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise RuntimeError("Model checksum mismatch")
        if manifest["architecture"] != "8-32-3-relu" or not manifest["synthetic_only"]:
            raise RuntimeError("Unsupported model manifest")
        model = architecture()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        app.state.version = manifest["version"]
        app.state.batcher = MicroBatcher(model)
        await app.state.batcher.start()
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await app.state.batcher.close()

    app = FastAPI(title="Inference Lab", version="0.1.0", lifespan=lifespan)

    @app.get("/health/live")
    def live():
        return {"live": True}

    @app.get("/health/ready")
    def ready():
        if not getattr(app.state, "ready", False) or not app.state.batcher.running:
            raise HTTPException(503, "Model unavailable")
        return {"ready": True, "version": app.state.version}

    @app.get("/metrics")
    def metrics():
        batcher = app.state.batcher
        return {
            "batches": batcher.batches,
            "examples": batcher.examples,
            "queued": batcher.queue.qsize(),
        }

    @app.post("/v1/predict")
    async def predict(body: PredictRequest):
        if not getattr(app.state, "ready", False) or not app.state.batcher.running:
            raise HTTPException(503, "Model unavailable")
        if body.expected_version is not None and body.expected_version != app.state.version:
            raise HTTPException(409, "Model version differs")
        try:
            probabilities = await app.state.batcher.predict(body.features)
        except (Overloaded, InferenceFailed, asyncio.TimeoutError) as error:
            raise HTTPException(
                503, "Capacity unavailable", headers={"Retry-After": "1"}
            ) from error
        return {
            "request_id": str(uuid.uuid4()),
            "model_version": app.state.version,
            "label": max(range(len(probabilities)), key=probabilities.__getitem__),
            "probabilities": probabilities,
        }

    return app


app = create_app()
