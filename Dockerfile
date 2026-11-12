FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache
WORKDIR /app

# CPU-only torch keeps the image ~1GB instead of ~5GB.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"
COPY serve ./serve

RUN useradd -m app && mkdir -p /app/.cache && chown -R app /app
USER app

# Required at runtime: -e MODEL_ID=YOUR_HF_USERNAME/jobbert-skill-extractor
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
