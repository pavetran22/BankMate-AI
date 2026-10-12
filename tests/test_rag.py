from app.config import settings
from app.rag.chunking import chunk_directory, slugify
from app.rag.retriever import get_retriever, tokenize


def test_chunks_keep_section_and_citation():
    chunks = chunk_directory(settings.policies_dir)
    assert len(chunks) > 20
    c = next(c for c in chunks if c.section == "Late Payment Charge")
    assert c.citation == "credit-card-terms#late-payment-charge"
    assert "RM100" in c.text


def test_disclaimer_is_not_indexed():
    assert not any("fictional bank" in c.text for c in chunk_directory(settings.policies_dir))


def test_slugify_and_tokenize():
    assert slugify("Interest and Finance Charges") == "interest-and-finance-charges"
    assert tokenize("What is the RM250 fee?") == ["rm250", "fee"]


def test_retrieves_right_section():
    hits = get_retriever().search("early settlement fee for personal loan")
    assert hits[0].chunk.citation == "personal-loan-faq#early-settlement"


def test_irrelevant_query_returns_nothing():
    assert get_retriever().search("weather forecast tokyo") == []
