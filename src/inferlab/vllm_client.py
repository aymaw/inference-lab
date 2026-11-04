"""Optional HTTP adapter. Tested with a mock transport, not a GPU server."""

import httpx


async def complete(base_url, model, prompt, api_key, transport=None):
    if not prompt.strip() or len(prompt) > 4000:
        raise ValueError("Use 1–4000 characters of synthetic input")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0, transport=transport) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0,
            },
        )
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("Expected text completion")
        return text
