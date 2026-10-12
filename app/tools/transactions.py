"""Account and fraud-analysis tools.

These are plain functions that take the *authenticated* customer_id as an
argument. The agent layer injects that id from the server-side session, never
from the model, so a prompt like "show me C004's transactions" cannot reach
another customer's data (see app/agent.py).

Risk scoring is rule-based and explainable on purpose: every point added has
a human-readable reason that maps to the bank's fraud-monitoring policy, which
matters for auditability in a regulated environment.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta

from app.db import connect

HIGH_RISK_CATEGORIES = {"gift cards", "cryptocurrency", "online gambling"}


class AccessDenied(Exception):
    pass


def _rows(sql: str, params: tuple) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_account_summary(customer_id: str) -> dict:
    cust = _rows("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    if not cust:
        return {"error": "Customer not found."}
    c = cust[0]
    since = (datetime.now() - timedelta(days=30)).isoformat()
    stats = _rows(
        """SELECT COUNT(*) AS n, ROUND(COALESCE(SUM(amount_myr),0),2) AS total
           FROM transactions WHERE customer_id = ? AND ts >= ?""",
        (customer_id, since),
    )[0]
    top = _rows(
        """SELECT category, ROUND(SUM(amount_myr),2) AS total FROM transactions
           WHERE customer_id = ? AND ts >= ? GROUP BY category ORDER BY total DESC LIMIT 3""",
        (customer_id, since),
    )
    return {
        "name": c["name"],
        "segment": c["segment"],
        "card": f"{c['card_product']} ending {c['card_last4']}",
        "last_30_days": {"transactions": stats["n"], "total_spend_myr": stats["total"],
                         "top_categories": top},
    }


def get_recent_transactions(customer_id: str, days: int = 30, limit: int = 15) -> dict:
    days = max(1, min(int(days), 90))
    limit = max(1, min(int(limit), 50))
    since = (datetime.now() - timedelta(days=days)).isoformat()
    rows = _rows(
        """SELECT txn_id, ts, amount_myr, merchant, category, country, channel
           FROM transactions WHERE customer_id = ? AND ts >= ?
           ORDER BY ts DESC LIMIT ?""",
        (customer_id, since, limit),
    )
    return {"days": days, "count": len(rows), "transactions": rows}


def _history(customer_id: str) -> list[dict]:
    return _rows(
        "SELECT * FROM transactions WHERE customer_id = ? ORDER BY ts", (customer_id,)
    )


def _score(txn: dict, history: list[dict]) -> dict:
    ts = datetime.fromisoformat(txn["ts"])
    prior = [h for h in history if h["ts"] < txn["ts"]]
    amounts = [h["amount_myr"] for h in prior] or [txn["amount_myr"]]
    mean = statistics.fmean(amounts)
    stdev = statistics.pstdev(amounts) or 1.0
    median = statistics.median(amounts)

    score, reasons = 0, []

    z = (txn["amount_myr"] - mean) / stdev
    # Ignore small purchases: a RM300 online order is not suspicious even if it
    # is unusual for a customer whose average spend is RM50.
    if z > 3 and txn["amount_myr"] >= 500 and len(prior) >= 10:
        score += 30
        reasons.append(f"Amount RM{txn['amount_myr']:,.2f} is far above usual spending "
                       f"(typical RM{median:,.2f}, {z:.1f} std devs above average).")

    countries = {h["country"] for h in prior}
    if txn["country"] not in countries:
        score += 20
        reasons.append(f"First transaction in country {txn['country']}.")

    if 0 <= ts.hour < 5:
        score += 15
        reasons.append(f"Made at {ts:%H:%M}, between midnight and 5am.")

    if txn["category"] in HIGH_RISK_CATEGORIES:
        score += 20
        reasons.append(f"High-risk merchant category: {txn['category']}.")

    window = [h for h in history
              if abs((datetime.fromisoformat(h["ts"]) - ts).total_seconds()) <= 600]
    if len(window) >= 4:
        small = sum(1 for h in window if h["amount_myr"] < 5)
        score += 25
        reasons.append(f"{len(window)} transactions within 10 minutes"
                       + (f", {small} of them under RM5 (possible card testing)." if small >= 3 else "."))

    merchants = {h["merchant"] for h in prior}
    if txn["merchant"] not in merchants and txn["amount_myr"] > 1000:
        score += 10
        reasons.append(f"New merchant ({txn['merchant']}) with a large amount.")

    score = min(score, 100)
    level = "high" if score >= 60 else "medium" if score >= 30 else "low"
    action = {
        "high": "Treat as likely fraud: advise the customer to block the card (Kill Switch "
                "or fraud hotline 1-300-88-6872) and lodge a dispute.",
        "medium": "Ask the customer to confirm whether they made this transaction.",
        "low": "No action needed; consistent with normal activity.",
    }[level]
    return {
        "transaction": {k: txn[k] for k in ("txn_id", "ts", "amount_myr", "merchant",
                                             "category", "country", "channel")},
        "risk_score": score,
        "risk_level": level,
        "reasons": reasons or ["Consistent with the customer's usual spending pattern."],
        "recommended_action": action,
    }


def assess_transaction_risk(customer_id: str, transaction_id: str) -> dict:
    txn = _rows("SELECT * FROM transactions WHERE txn_id = ?", (transaction_id.strip(),))
    # Same response for "not found" and "belongs to someone else" so the tool
    # cannot be used to probe which transaction ids exist.
    if not txn or txn[0]["customer_id"] != customer_id:
        return {"error": f"Transaction {transaction_id} was not found on this account."}
    return _score(txn[0], _history(customer_id))


def scan_for_suspicious_activity(customer_id: str, days: int = 30) -> dict:
    days = max(1, min(int(days), 90))
    history = _history(customer_id)
    since = (datetime.now() - timedelta(days=days)).isoformat()
    flagged = []
    for txn in history:
        if txn["ts"] < since:
            continue
        result = _score(txn, history)
        if result["risk_level"] != "low":
            flagged.append(result)
    flagged.sort(key=lambda r: -r["risk_score"])
    return {
        "days_scanned": days,
        "transactions_scanned": sum(1 for t in history if t["ts"] >= since),
        "flagged_count": len(flagged),
        "flagged": flagged[:10],
    }
