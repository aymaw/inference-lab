"""Exercise the actual HTTP listener, rather than an in-process ASGI transport."""

import argparse
import json
import math
import time
import urllib.error
import urllib.request


def check(base_url):
    deadline = time.monotonic() + 45
    while True:
        try:
            with urllib.request.urlopen(base_url + "/health/ready", timeout=2) as response:
                ready = json.load(response)
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
    body = json.dumps({"features": [0.2] * 8, "expected_version": ready["version"]}).encode()
    request = urllib.request.Request(
        base_url + "/v1/predict", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        result = json.load(response)
    assert result["model_version"] == ready["version"]
    assert len(result["probabilities"]) == 3
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in result["probabilities"])
    assert math.isclose(sum(result["probabilities"]), 1.0, abs_tol=1e-6)
    return {"scope": "HTTP loopback to running container", "ready": ready, "prediction": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    print(json.dumps(check(args.url), indent=2))
