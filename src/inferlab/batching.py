"""Bounded async admission and microbatching. No request bodies are logged."""

import asyncio
import math

import torch


class Overloaded(Exception):
    pass


class InferenceFailed(Exception):
    pass


class MicroBatcher:
    def __init__(self, model, capacity=64, batch_size=16, wait=0.005):
        if (
            not isinstance(capacity, int)
            or capacity < 1
            or not isinstance(batch_size, int)
            or batch_size < 1
            or not math.isfinite(wait)
            or wait < 0
        ):
            raise ValueError("Use positive capacity/batch_size and a finite nonnegative wait")
        self.model = model
        self.queue = asyncio.Queue(maxsize=capacity)
        self.batch_size, self.wait = batch_size, wait
        self.worker = None
        self.closed = False
        self.batches = self.examples = 0

    async def start(self):
        if self.worker is not None or self.closed:
            raise RuntimeError("Batcher can only be started once")
        self.worker = asyncio.create_task(self._work())

    @property
    def running(self):
        return not self.closed and self.worker is not None and not self.worker.done()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.worker is not None:
                self.worker.cancel()
                try:
                    await self.worker
                except asyncio.CancelledError:
                    pass
        finally:
            while not self.queue.empty():
                _, future = self.queue.get_nowait()
                if not future.done():
                    future.set_exception(Overloaded("service shutting down"))

    async def predict(self, row):
        if not self.running:
            raise Overloaded("service unavailable")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((row, future))
        except asyncio.QueueFull as error:
            raise Overloaded("queue full") from error
        return await asyncio.wait_for(future, timeout=2.0)

    def _infer(self, rows):
        with torch.inference_mode():
            logits = self.model(torch.tensor(rows, dtype=torch.float32))
            if logits.ndim != 2 or logits.shape[0] != len(rows) or not torch.isfinite(logits).all():
                raise ValueError("Model returned invalid logits")
            return logits.softmax(-1).tolist()

    async def _work(self):
        pending = []
        try:
            while True:
                pending = [await self.queue.get()]
                await asyncio.sleep(self.wait)
                while len(pending) < self.batch_size and not self.queue.empty():
                    pending.append(self.queue.get_nowait())
                active = [(row, f) for row, f in pending if not f.done()]
                if active:
                    try:
                        # One worker keeps model calls serialized while health checks and
                        # request cancellation can run on the event loop during inference.
                        probs = await asyncio.to_thread(self._infer, [r for r, _ in active])
                        self.batches += 1
                        self.examples += len(active)
                        for (_, future), value in zip(active, probs, strict=True):
                            if not future.done():
                                future.set_result(value)
                    except Exception:
                        for _, future in active:
                            if not future.done():
                                future.set_exception(InferenceFailed("model inference failed"))
                pending = []
        finally:
            for _, future in pending:
                if not future.done():
                    future.set_exception(Overloaded("service shutting down"))
