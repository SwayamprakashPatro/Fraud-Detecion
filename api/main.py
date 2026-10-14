"""HTTP API for the GraphSAGE fraud model, plus the web UI.

    python run_app.py          then open http://127.0.0.1:8000
Interactive API docs:          http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.service import ModelService

svc = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc.load()
    yield


app = FastAPI(title="GraphSAGE Fraud API", version="1.0", lifespan=lifespan)


class NewPayment(BaseModel):
    sender: str          # e.g. "account_12"
    receiver: str        # e.g. "account_44"
    features: Dict[str, float] = {}   # cleaned (scaled) feature values; missing ones use the dataset mean


@app.get("/api/health")
def health():
    return svc.health()


@app.get("/api/data")
def data():
    """Everything the UI needs: scores for every payment, history, report, facts and stress tests."""
    return JSONResponse(svc.payload())


@app.get("/api/payments/{tid}")
def payment(tid: int):
    p = svc.payment(tid)
    if p is None:
        raise HTTPException(404, f"No payment with ID {tid}")
    return p


@app.post("/api/score")
def score(body: NewPayment):
    try:
        return svc.score_new(body.sender, body.receiver, body.features)
    except KeyError as e:
        raise HTTPException(422, str(e).strip("'\""))


@app.get("/api/metrics")
def metrics(threshold: Optional[float] = Query(None, ge=0, le=1)):
    """Held-out test metrics at any alert threshold (defaults to the one saved with the checkpoint)."""
    return svc.metrics_at(svc.threshold if threshold is None else threshold)


@app.post("/api/reload")
def reload():
    """Reload the checkpoint and logs from disk, for example right after retraining."""
    svc.load()
    return svc.health()


FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
