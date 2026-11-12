"""FastAPI inference service.

    MODEL_ID=YOUR_USER/jobbert-skill-extractor uvicorn serve.app:app --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from skillex.predict import SkillExtractor

log = logging.getLogger("skillex")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.environ.get("MODEL_ID")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "20000"))
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_ID:
        raise RuntimeError("Set the MODEL_ID env var (Hub id or local path).")
    state["extractor"] = SkillExtractor(MODEL_ID)
    log.info("loaded model %s", MODEL_ID)
    yield
    state.clear()


app = FastAPI(title="Skill Extractor", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # Prometheus metrics at /metrics


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)


class Span(BaseModel):
    type: str
    text: str
    start: int
    end: int
    score: float


class ExtractResponse(BaseModel):
    skills: list[str]
    knowledge: list[str]
    spans: list[Span]
    model: str
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, f"text longer than {MAX_CHARS} characters")
    t0 = time.perf_counter()
    result = state["extractor"].extract(req.text)
    latency = (time.perf_counter() - t0) * 1000
    log.info("extract chars=%d spans=%d latency_ms=%.1f", len(req.text), len(result["spans"]), latency)
    return ExtractResponse(**result, model=MODEL_ID, latency_ms=round(latency, 1))
