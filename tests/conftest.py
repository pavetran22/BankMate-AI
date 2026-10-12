import os
import sys
from pathlib import Path

# Tests always run offline against a throwaway database.
os.environ.pop("GOOGLE_API_KEY", None)
os.environ["OFFLINE_MODE"] = "1"
os.environ["BANKMATE_DB"] = str(Path(__file__).parent / ".test_bank.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    from app import db
    from app.config import settings

    db.seed(force=True)
    yield
    settings.db_path.unlink(missing_ok=True)
