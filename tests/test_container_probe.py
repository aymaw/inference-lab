import io
import json

from container_smoke import check


def test_probe_retries_connection_reset_during_startup(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise ConnectionResetError("proxy accepted before server startup")
        if len(calls) == 2:
            return io.BytesIO(json.dumps({"version": "test"}).encode())
        return io.BytesIO(
            json.dumps({"model_version": "test", "probabilities": [0.2, 0.3, 0.5]}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    result = check("http://local")
    assert len(calls) == 3
    assert result["prediction"]["model_version"] == "test"
