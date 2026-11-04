FROM python:3.12-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN python -m inferlab.train --out /model

FROM python:3.12-slim
RUN useradd --uid 10001 --create-home runner
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=10001:10001 /model /app/model
ENV PATH="/opt/venv/bin:$PATH" MODEL_DIR=/app/model PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["uvicorn", "inferlab.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
