"""Chat orchestration: guardrails -> ADK agent run -> output checks -> response.

Returns a structured result the UI and the eval harness both use: the answer,
the sources it cited, which tools ran (the agent's "trace"), guardrail flags
and latency.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app import db
from app.agent import build_agent
from app.config import settings
from app.guardrails import check_input, check_output

CITATION_RE = re.compile(r"\[([a-z0-9-]+#[a-z0-9-]+)\]")


@dataclass
class ChatResult:
    session_id: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)
    guardrail_flags: list[str] = field(default_factory=list)
    blocked: bool = False
    latency_ms: int = 0
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class BankMateService:
    def __init__(self, model=None):
        self.agent = build_agent(model)
        self.sessions = InMemorySessionService()
        self.runner = Runner(app_name=settings.app_name, agent=self.agent,
                             session_service=self.sessions)
        self.model_name = getattr(self.agent.model, "model", None) or str(self.agent.model)

    async def _session(self, customer_id: str, session_id: str | None) -> str:
        if session_id:
            existing = await self.sessions.get_session(
                app_name=settings.app_name, user_id=customer_id, session_id=session_id)
            # Sessions are looked up per user, so another customer's session id
            # simply isn't found here and a fresh session is created instead.
            if existing and existing.state.get("customer_id") == customer_id:
                return session_id
        session = await self.sessions.create_session(
            app_name=settings.app_name, user_id=customer_id,
            state={"customer_id": customer_id}, session_id=str(uuid.uuid4()))
        return session.id

    async def chat(self, customer_id: str, message: str, session_id: str | None = None) -> ChatResult:
        started = time.perf_counter()
        session_id = await self._session(customer_id, session_id)

        guard = check_input(message)
        if guard.flags:
            db.audit("guardrail", json.dumps(guard.flags), session_id, customer_id)
        if not guard.allowed:
            return ChatResult(session_id, guard.message or "", guardrail_flags=guard.flags,
                              blocked=True, model=self.model_name,
                              latency_ms=int((time.perf_counter() - started) * 1000))

        tools_used: list[dict] = []
        retrieved: dict[str, dict] = {}
        answer_parts: list[str] = []

        content = types.Content(role="user", parts=[types.Part(text=guard.text)])
        async for event in self.runner.run_async(
            user_id=customer_id, session_id=session_id, new_message=content
        ):
            for call in event.get_function_calls() or []:
                tools_used.append({"tool": call.name, "args": dict(call.args or {})})
            for resp in event.get_function_responses() or []:
                if resp.name == "search_bank_policies":
                    for hit in (resp.response or {}).get("results", []):
                        retrieved.setdefault(hit["citation"], hit)
            if event.is_final_response() and event.content and event.content.parts:
                answer_parts.extend(p.text for p in event.content.parts if p.text)

        answer = "\n".join(answer_parts).strip() or "Sorry, I couldn't produce an answer."
        answer, out_flags = check_output(answer)

        cited = list(dict.fromkeys(CITATION_RE.findall(answer)))
        citations = [
            {"citation": c, "document": retrieved[c]["document"], "section": retrieved[c]["section"],
             "text": retrieved[c]["text"]}
            for c in cited if c in retrieved
        ]
        # A citation the retriever never returned is a hallucinated source: flag it.
        flags = guard.flags + out_flags
        if any(c not in retrieved for c in cited):
            flags.append("unverified_citation")
        if guard.message:  # e.g. credential warning, shown before the answer
            answer = f"{guard.message}\n\n{answer}"

        result = ChatResult(
            session_id=session_id, answer=answer, citations=citations, tools_used=tools_used,
            guardrail_flags=flags, model=self.model_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        db.audit("chat", json.dumps({"tools": [t["tool"] for t in tools_used],
                                     "latency_ms": result.latency_ms, "flags": flags}),
                 session_id, customer_id)
        return result
