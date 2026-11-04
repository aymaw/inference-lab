# Optional GPU experiment — not executed

Use a compatible GPU host, a public model whose license you have reviewed, and
the [official vLLM Docker documentation](https://docs.vllm.ai/en/latest/deployment/docker/).
Select an explicit compatible release image and model revision for a real run;
record both before reporting results. Do not present the mock test as GPU evidence.

Example shape (replace environment variables with your reviewed image/model):

```bash
docker run --rm --gpus all -p 127.0.0.1:8001:8000 --shm-size 1g   "$VLLM_IMAGE" --model "$MODEL_ID" --revision "$MODEL_REVISION" --max-model-len 2048
```

The loopback binding is deliberate. vLLM's API-key option does not protect every
endpoint; it is not a substitute for an authenticated proxy/network boundary.
See the [upstream API documentation](https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/openai_compatible_server.md).

After startup, call `inferlab.vllm_client.complete` against `http://127.0.0.1:8001`
with a fictional prompt. Record GPU type/VRAM, model/revision, quantization,
concurrency, input/output token distributions, time to first token, inter-token
latency, throughput, failures, and peak memory. No such measurements are included
in this repository because no compatible GPU run was performed.
