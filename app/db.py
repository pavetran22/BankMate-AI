"""SQLite store with synthetic customers and transactions.

The data is generated deterministically (fixed random seed) relative to the
day it is seeded, so demos always look "recent". A few customers have
planted suspicious patterns so the fraud tools have something to find:

  C001 Aisyah  - normal spending only
  C002 Daniel  - large overseas purchase at 3am (new country, new merchant)
  C003 Mei Ling - card-testing burst of tiny charges, then a gift-card purchase
  C004 Ravi    - large transfer to a cryptocurrency exchange
  C005 Siti    - normal spending only
"""
from __future__ import annotations

import random
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    segment       TEXT NOT NULL,
    home_country  TEXT NOT NULL,
    card_last4    TEXT NOT NULL,
    card_product  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id        TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
    ts            TEXT NOT NULL,          -- ISO 8601, local time
    amount_myr    REAL NOT NULL,
    merchant      TEXT NOT NULL,
    category      TEXT NOT NULL,
    country       TEXT NOT NULL,          -- ISO 3166 alpha-2
    channel       TEXT NOT NULL,          -- card_present | online | transfer
    status        TEXT NOT NULL DEFAULT 'posted'
);
CREATE INDEX IF NOT EXISTS idx_txn_customer_ts ON transactions(customer_id, ts);
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    session_id    TEXT,
    customer_id   TEXT,
    event         TEXT NOT NULL,          -- tool_call | guardrail | chat
    detail        TEXT NOT NULL
);
"""

CUSTOMERS = [
    ("C001", "Aisyah Rahman", "Premier", "MY", "4821", "Nusa Platinum"),
    ("C002", "Daniel Tan", "Personal", "MY", "1934", "Nusa Platinum"),
    ("C003", "Lim Mei Ling", "Personal", "MY", "7702", "Nusa Classic"),
    ("C004", "Ravi Kumar", "Personal", "MY", "3358", "Nusa Classic"),
    ("C005", "Siti Nurhaliza Osman", "Premier", "MY", "6610", "Nusa Platinum"),
]

# (merchant, category, typical min, typical max, channel)
NORMAL_MERCHANTS = [
    ("FreshMart Supermarket", "groceries", 40, 260, "card_present"),
    ("Kopi Corner", "dining", 8, 35, "card_present"),
    ("Nasi Kandar Seri", "dining", 12, 45, "card_present"),
    ("Petrol Plus Station", "fuel", 50, 120, "card_present"),
    ("ShopMart Online", "e-commerce", 25, 350, "online"),
    ("RideGo", "transport", 9, 40, "online"),
    ("StreamFlix", "subscriptions", 45, 45, "online"),
    ("Tenaga Utility Bill", "utilities", 90, 260, "online"),
    ("PharmaCare", "health", 15, 120, "card_present"),
    ("CineMax", "entertainment", 20, 70, "card_present"),
]


@contextmanager
def connect(db_path: Path | None = None):
    conn = sqlite3.connect(db_path or settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _normal_history(rng: random.Random, customer_id: str, now: datetime) -> list[tuple]:
    rows = []
    n = 0
    for day in range(90, 0, -1):
        for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):
            merchant, cat, lo, hi, channel = rng.choice(NORMAL_MERCHANTS)
            ts = (now - timedelta(days=day)).replace(
                hour=rng.randint(8, 22), minute=rng.randint(0, 59), second=0, microsecond=0
            )
            n += 1
            rows.append(
                (f"T{customer_id[1:]}-{n:04d}", customer_id, ts.isoformat(),
                 round(rng.uniform(lo, hi), 2), merchant, cat, "MY", channel, "posted")
            )
    return rows


def _planted(customer_id: str, now: datetime) -> list[tuple]:
    """Deliberately suspicious transactions with stable, memorable ids."""
    d = lambda days, h, m, s=0: (now - timedelta(days=days)).replace(  # noqa: E731
        hour=h, minute=m, second=s, microsecond=0).isoformat()
    if customer_id == "C002":
        return [("TX-C002-9001", "C002", d(2, 3, 12), 4850.00, "LuxWatch Bangkok",
                 "luxury goods", "TH", "online", "posted")]
    if customer_id == "C003":
        burst = [(f"TX-C003-90{i:02d}", "C003", d(1, 1, 40 + i // 2, (i * 17) % 60),
                  round(1.0 + i * 0.37, 2), "QuickPay Digital", "digital goods", "GB",
                  "online", "posted") for i in range(1, 7)]
        burst.append(("TX-C003-9010", "C003", d(1, 1, 52), 2300.00, "GiftCardHub",
                      "gift cards", "GB", "online", "posted"))
        return burst
    if customer_id == "C004":
        return [("TX-C004-9001", "C004", d(3, 23, 5), 8000.00, "CoinVault Exchange",
                 "cryptocurrency", "SG", "transfer", "posted")]
    return []


def seed(db_path: Path | None = None, force: bool = False) -> Path:
    path = Path(db_path or settings.db_path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(42)
    now = datetime.now().replace(second=0, microsecond=0)
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", CUSTOMERS)
        for cust in CUSTOMERS:
            cid = cust[0]
            rows = _normal_history(rng, cid, now) + _planted(cid, now)
            conn.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return path


def audit(event: str, detail: str, session_id: str | None = None,
          customer_id: str | None = None, db_path: Path | None = None) -> None:
    """Append-only audit trail of tool calls and guardrail decisions."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, session_id, customer_id, event, detail) VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), session_id, customer_id, event, detail[:2000]),
        )


if __name__ == "__main__":
    p = seed(force=True)
    print(f"Seeded {p}")
