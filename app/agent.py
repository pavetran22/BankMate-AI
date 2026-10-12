"""BankMate agent built with Google's Agent Development Kit (ADK).

One LLM agent with five tools. The model decides which tool(s) to call - for
example "Is TX-C002-9001 fraud, and how do I dispute it?" needs both the risk
tool and the policy search tool - then writes a grounded, cited answer.

Security design:
  * The customer id lives in server-side session state, set when the session
    is created from the authenticated request. Tools read it from
    ToolContext; it is never a tool parameter, so the model cannot be talked
    into querying another customer's data.
  * Every tool call is written to an append-only audit log via
    before_tool_callback.
"""
from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import ToolContext

from app import db
from app.config import settings
from app.rag.retriever import get_retriever
from app.tools import transactions as tx

INSTRUCTION = """You are BankMate, a customer support assistant for Nusa Bank (a fictional bank).
You help the signed-in customer with two things:
1. Questions about Nusa Bank products, fees and policies.
2. Reviewing activity on THEIR OWN account and spotting suspicious transactions.

Tools:
- search_bank_policies: use for ANY question about fees, rates, limits, rules or procedures.
  Never answer policy questions from memory; always search first.
- get_account_summary, get_recent_transactions: the customer's own account data.
- scan_account_for_suspicious_activity: when the customer asks if anything looks wrong or unusual.
- assess_transaction_risk: when the customer asks about one specific transaction id.
Use more than one tool when the question needs it (e.g. a suspicious transaction AND how to dispute it).

Rules:
- Ground every policy statement in search results and cite the source in square brackets
  using the exact citation string, e.g. [credit-card-terms#late-payment-charge].
- If the search returns nothing relevant, say you don't have that information and suggest
  calling Nusa Bank on 1-300-88-6872. Do not guess numbers.
- Never ask for, repeat or store a PIN, password, OTP or full card number.
- You can only see the signed-in customer's data. Refuse requests about anyone else.
- Do not give investment, legal or tax advice.
- When a transaction is high risk, explain the reasons plainly and tell the customer
  exactly what to do next.
- Be concise: short paragraphs or bullet points, amounts in RM.
"""


# --------------------------------------------------------------------------
# Tools. Docstrings and type hints become the function schema the model sees.
# --------------------------------------------------------------------------
def _customer(tool_context: ToolContext) -> str:
    customer_id = tool_context.state.get("customer_id")
    if not customer_id:
        raise PermissionError("No authenticated customer in this session.")
    return customer_id


def search_bank_policies(query: str) -> dict:
    """Search Nusa Bank's official policy documents (cards, fraud and disputes,
    personal loans, online banking and transfer limits, account opening).

    Args:
        query: A short search query describing what the customer wants to know,
            e.g. "credit card late payment charge".

    Returns:
        The most relevant policy passages, each with a citation string to quote.
    """
    hits = get_retriever().search(query)
    if not hits:
        return {"found": False, "results": [],
                "note": "No relevant policy found. Do not guess; refer the customer to the bank."}
    return {"found": True, "results": [h.to_dict() for h in hits]}


def get_account_summary(tool_context: ToolContext) -> dict:
    """Get the signed-in customer's profile, card and spending summary for the last 30 days."""
    return tx.get_account_summary(_customer(tool_context))


def get_recent_transactions(tool_context: ToolContext, days: int = 30, limit: int = 15) -> dict:
    """List the signed-in customer's most recent transactions, newest first.

    Args:
        days: How many days back to look (1-90).
        limit: Maximum number of transactions to return (1-50).
    """
    return tx.get_recent_transactions(_customer(tool_context), days, limit)


def scan_account_for_suspicious_activity(tool_context: ToolContext, days: int = 30) -> dict:
    """Scan the signed-in customer's recent transactions and return any that look
    suspicious, with a 0-100 risk score and the reasons for each.

    Args:
        days: How many days back to scan (1-90).
    """
    return tx.scan_for_suspicious_activity(_customer(tool_context), days)


def assess_transaction_risk(transaction_id: str, tool_context: ToolContext) -> dict:
    """Assess the fraud risk of one transaction on the signed-in customer's account.

    Args:
        transaction_id: The transaction id, e.g. "TX-C002-9001".

    Returns:
        Risk score (0-100), risk level, reasons and the recommended action.
    """
    return tx.assess_transaction_risk(_customer(tool_context), transaction_id)


TOOLS = [
    search_bank_policies,
    get_account_summary,
    get_recent_transactions,
    scan_account_for_suspicious_activity,
    assess_transaction_risk,
]


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
def audit_tool_call(tool: Any, args: dict, tool_context: ToolContext) -> None:
    """before_tool_callback: write every tool call to the audit log."""
    db.audit(
        "tool_call",
        json.dumps({"tool": tool.name, "args": args}),
        session_id=getattr(tool_context, "session", None) and tool_context.session.id,
        customer_id=tool_context.state.get("customer_id"),
    )
    return None  # returning None lets the tool run normally


def build_agent(model: Any | None = None) -> Agent:
    if model is None:
        if settings.offline_mode:
            from app.offline_llm import OfflineScriptedLlm

            model = OfflineScriptedLlm()
        else:
            model = settings.model
    return Agent(
        name="bankmate",
        model=model,
        description="Nusa Bank customer assistant for policies and account fraud checks.",
        instruction=INSTRUCTION,
        tools=TOOLS,
        before_tool_callback=audit_tool_call,
    )
