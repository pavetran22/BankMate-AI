"""FastAPI app: chat API, health/metrics, audit trail and the web UI.

Authentication is simulated with an X-Customer-Id header picked from a demo
dropdown. In production this would come from a verified identity token
(e.g. OIDC/JWT from the bank's login), never from the client directly.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import db
from app.config import ROOT, settings
from app.rag.retriever import get_retriever
from app.service import BankMateService


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
log = logging.getLogger("bankmate")

METRICS = {"requests": 0, "blocked": 0, "errors": 0, "total_latency_ms": 0, "tool_calls": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.seed()
    retriever = get_retriever()  # build the index at startup, not on the first request
    app.state.service = BankMateService()
    log.info("BankMate ready", extra={"extra_fields": {
        "model": app.state.service.model_name, "retrieval": retriever.mode,
        "chunks": len(retriever.chunks)}})
    yield


app = FastAPI(title="BankMate AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=settings.max_input_chars)
    session_id: str | None = None


def _require_customer(customer_id: str | None) -> str:
    if not customer_id:
        raise HTTPException(401, "Missing X-Customer-Id header")
    with db.connect() as conn:
        if not conn.execute("SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)).fetchone():
            raise HTTPException(401, "Unknown customer")
    return customer_id


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health(request: Request):
    svc: BankMateService = request.app.state.service
    return {"status": "ok", "model": svc.model_name, "offline_mode": settings.offline_mode,
            "retrieval": get_retriever().mode}


@app.get("/metrics")
def metrics():
    n = METRICS["requests"] or 1
    return {**METRICS, "avg_latency_ms": round(METRICS["total_latency_ms"] / n, 1)}


@app.get("/api/customers")
def customers():
    """Demo-only: lets the UI pretend to sign in as different customers."""
    with db.connect() as conn:
        rows = conn.execute("SELECT customer_id, name, card_product FROM customers").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request, x_customer_id: str | None = Header(None)):
    customer_id = _require_customer(x_customer_id)
    svc: BankMateService = request.app.state.service
    started = time.perf_counter()
    try:
        result = await svc.chat(customer_id, body.message, body.session_id)
    except Exception:
        METRICS["errors"] += 1
        log.exception("chat failed")
        raise HTTPException(502, "The assistant is unavailable right now. Please try again.")

    METRICS["requests"] += 1
    METRICS["blocked"] += int(result.blocked)
    METRICS["total_latency_ms"] += result.latency_ms
    for t in result.tools_used:
        METRICS["tool_calls"][t["tool"]] = METRICS["tool_calls"].get(t["tool"], 0) + 1
    log.info("chat", extra={"extra_fields": {
        "customer": customer_id, "session": result.session_id,
        "tools": [t["tool"] for t in result.tools_used], "flags": result.guardrail_flags,
        "latency_ms": int((time.perf_counter() - started) * 1000)}})
    return result.to_dict()


@app.get("/api/audit")
def audit_log(x_customer_id: str | None = Header(None), limit: int = 20):
    """The signed-in customer's own audit trail (tool calls, guardrail events)."""
    customer_id = _require_customer(x_customer_id)
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT ts, event, detail FROM audit_log WHERE customer_id = ? ORDER BY id DESC LIMIT ?",
            (customer_id, max(1, min(limit, 100))),
        ).fetchall()
    return [dict(r) for r in rows]
