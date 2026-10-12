"""A scripted stand-in for Gemini so the app, tests and evals run with no API key.

It is NOT an LLM. It routes with keyword rules, emits real ADK function calls,
and stitches tool results into an answer. That exercises the whole pipeline
(agent loop, tools, access control, audit log, citations, API, UI) offline.
Set GOOGLE_API_KEY to use Gemini for real.
"""
from __future__ import annotations

import re
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.models._capabilities import LlmCapabilities
from google.genai import types

TXN_RE = re.compile(r"\b(TX-C\d{3}-\d{4}|T\d{3}-\d{4})\b", re.IGNORECASE)


def _route(text: str) -> list[tuple[str, dict]]:
    t = text.lower()
    calls: list[tuple[str, dict]] = []
    m = TXN_RE.search(text)
    if m:
        calls.append(("assess_transaction_risk", {"transaction_id": m.group(1).upper()}))
    elif re.search(r"suspicious|unusual|fraud check|anything wrong|look(s)? (odd|weird|strange)|scan", t):
        calls.append(("scan_account_for_suspicious_activity", {"days": 30}))
    elif re.search(r"\b(recent|last|latest)\b.*\btransactions?\b|\bwhat did i (buy|spend)", t):
        calls.append(("get_recent_transactions", {"days": 30, "limit": 10}))
    elif re.search(r"\b(summary|my account|my card|how much (have|did) i spen[dt])\b", t):
        calls.append(("get_account_summary", {}))

    needs_policy = not calls or re.search(
        r"\b(how (do|can) i|dispute|report|fee|charge|limit|policy|rate|liable|block)\b", t)
    if needs_policy:
        calls.append(("search_bank_policies", {"query": text}))
    return calls


def _last_user_text(req: LlmRequest) -> str:
    for content in reversed(req.contents or []):
        if content.role == "user":
            for part in content.parts or []:
                if part.text:
                    return part.text
    return ""


def _render(responses: list[types.FunctionResponse]) -> str:
    out: list[str] = []
    for fr in responses:
        r = fr.response or {}
        if fr.name == "search_bank_policies":
            if not r.get("found"):
                out.append("I don't have information on that in Nusa Bank's policies. "
                           "Please call Nusa Bank on 1-300-88-6872.")
            else:
                for hit in r["results"][:2]:
                    out.append(f"{hit['text']} [{hit['citation']}]")
        elif fr.name == "assess_transaction_risk":
            if "error" in r:
                out.append(r["error"])
            else:
                t = r["transaction"]
                out.append(f"{t['txn_id']} - RM{t['amount_myr']:,.2f} at {t['merchant']} "
                           f"({t['country']}): risk {r['risk_score']}/100 ({r['risk_level']}).")
                out.extend(f"- {reason}" for reason in r["reasons"])
                out.append(r["recommended_action"])
        elif fr.name == "scan_account_for_suspicious_activity":
            if not r.get("flagged_count"):
                out.append(f"I checked {r.get('transactions_scanned', 0)} transactions from the "
                           f"last {r.get('days_scanned')} days and nothing looks unusual.")
            else:
                out.append(f"{r['flagged_count']} transaction(s) look unusual:")
                for f in r["flagged"][:5]:
                    t = f["transaction"]
                    out.append(f"- {t['txn_id']}: RM{t['amount_myr']:,.2f} at {t['merchant']} - "
                               f"{f['risk_level']} risk ({f['risk_score']}/100). {f['reasons'][0]}")
        elif fr.name == "get_recent_transactions":
            out.append(f"Your last {r.get('count', 0)} transactions:")
            for t in r.get("transactions", []):
                out.append(f"- {t['ts'][:16].replace('T', ' ')}  RM{t['amount_myr']:,.2f}  {t['merchant']}")
        elif fr.name == "get_account_summary":
            s = r.get("last_30_days", {})
            out.append(f"{r.get('name')}, {r.get('card')}: {s.get('transactions')} transactions "
                       f"totalling RM{s.get('total_spend_myr', 0):,.2f} in the last 30 days.")
    return "\n".join(out) + "\n\n_(Offline demo mode: set GOOGLE_API_KEY to use Gemini.)_"


class OfflineScriptedLlm(BaseLlm):
    model: str = "offline-scripted"

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        last = (llm_request.contents or [None])[-1]
        responses = [p.function_response for p in (last.parts if last else []) or []
                     if p.function_response]
        if responses:
            parts = [types.Part(text=_render(responses))]
        else:
            parts = [types.Part.from_function_call(name=n, args=a)
                     for n, a in _route(_last_user_text(llm_request))]
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            turn_complete=True,
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=0, candidates_token_count=0, total_token_count=0),
        )
