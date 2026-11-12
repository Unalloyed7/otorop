FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/app/data \
    WEB_STATE_DIR=/state \
    OTOROP_KNOWLEDGE_CACHE_DIR=/state/knowledge \
    OTOROP_EMBEDDING_CACHE_DIR=/tmp/otorop/embeddings
WORKDIR /app
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt \
    && useradd --uid 10001 --create-home appuser \
    && mkdir /state && chown 10001:10001 /state && chmod 700 /state
COPY *.py ./
COPY web ./web
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["python", "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
