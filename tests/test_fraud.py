from app.tools import transactions as tx


def test_planted_overseas_purchase_is_high_risk():
    r = tx.assess_transaction_risk("C002", "TX-C002-9001")
    assert r["risk_level"] == "high"
    assert any("country TH" in reason for reason in r["reasons"])
    assert any("midnight" in reason for reason in r["reasons"])


def test_card_testing_burst_detected():
    r = tx.scan_for_suspicious_activity("C003", 30)
    ids = [f["transaction"]["txn_id"] for f in r["flagged"]]
    assert ids[0] == "TX-C003-9010"
    assert any("card testing" in reason for f in r["flagged"] for reason in f["reasons"])


def test_clean_customer_has_no_flags():
    assert tx.scan_for_suspicious_activity("C001", 30)["flagged_count"] == 0
    assert tx.scan_for_suspicious_activity("C005", 30)["flagged_count"] == 0


def test_cannot_read_other_customers_transaction():
    r = tx.assess_transaction_risk("C001", "TX-C002-9001")
    assert "error" in r and "not found" in r["error"]


def test_limits_are_clamped():
    r = tx.get_recent_transactions("C001", days=10_000, limit=10_000)
    assert r["days"] == 90 and r["count"] <= 50
