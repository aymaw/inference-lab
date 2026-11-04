"""Bounded async admission and microbatching. No request bodies are logged."""

import asyncio

import torch


class Overloaded(Exception):
    pass


class MicroBatcher:
    def __init__(self, model, capacity=64, batch_size=16, wait=0.005):
        self.model = model
        self.queue = asyncio.Queue(maxsize=capacity)
        self.batch_size, self.wait = batch_size, wait
        self.worker = None
        self.closed = False
        self.batches = self.examples = 0

    async def start(self):
        self.worker = asyncio.create_task(self._work())

    async def close(self):
        self.closed = True
        self.worker.cancel()
        try:
            await self.worker
        except asyncio.CancelledError:
            pass
        while not self.queue.empty():
            _, future = self.queue.get_nowait()
            if not future.done():
                future.set_exception(Overloaded("service shutting down"))

    async def predict(self, row):
        if self.closed:
            raise Overloaded("service shutting down")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((row, future))
        except asyncio.QueueFull as error:
            raise Overloaded("queue full") from error
        return await asyncio.wait_for(future, timeout=2.0)

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
                        with torch.inference_mode():
                            probs = (
                                self.model(torch.tensor([r for r, _ in active]))
                                .softmax(-1)
                                .tolist()
                            )
                        self.batches += 1
                        self.examples += len(active)
                        for (_, future), value in zip(active, probs, strict=True):
                            if not future.done():
                                future.set_result(value)
                    except Exception as error:
                        for _, future in active:
                            if not future.done():
                                future.set_exception(error)
                pending = []
        finally:
            for _, future in pending:
                if not future.done():
                    future.set_exception(Overloaded("service shutting down"))
