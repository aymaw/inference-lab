ARG PYTHON_IMAGE=python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
FROM ${PYTHON_IMAGE} AS builder
WORKDIR /build
RUN pip install --no-cache-dir uv==0.12.5
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
RUN /opt/venv/bin/python -m inferlab.train --out /model

FROM ${PYTHON_IMAGE}
RUN useradd --uid 10001 --create-home runner
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=10001:10001 /model /app/model
ENV PATH="/opt/venv/bin:$PATH" MODEL_DIR=/app/model PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["uvicorn", "inferlab.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
