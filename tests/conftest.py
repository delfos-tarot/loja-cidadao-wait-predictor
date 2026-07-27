"""Ensures the SQLite history DB and trained model artifact exist before the
test session runs, so `pytest` is runnable standalone on a clean checkout
without requiring the full real-data pipeline (dados.gov.pt download,
Nominatim geocoding) first.

Uses a small, bounded synthetic subset (few branches/services) rather than
the full real branch registry (78 branches, up to ~40 services each) so this
fixture stays fast and fully offline. If a real production DB/model already
exist (as they do in this project's own working copy after running the real
pipeline), this fixture is a no-op and tests exercise the real artifacts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model() -> None:
    from config import BRANCHES, DEFAULT_DB_PATH, DEFAULT_MODEL_PATH
    from pipeline.db import count_samples

    if count_samples(DEFAULT_DB_PATH) < 500:
        from pipeline.db import get_connection, insert_queue_samples, upsert_branch
        from pipeline.synthetic_bootstrap import generate_synthetic_bootstrap

        small_branch_subset = BRANCHES[:3]
        readings = generate_synthetic_bootstrap(days=14, branches=small_branch_subset, max_services_per_branch=3)
        with get_connection(DEFAULT_DB_PATH) as connection:
            for branch in small_branch_subset:
                upsert_branch(connection, branch.branch_id, branch.name, branch.district, branch.latitude, branch.longitude)
            insert_queue_samples(connection, readings)

    if not Path(DEFAULT_MODEL_PATH).exists():
        subprocess.run([sys.executable, "-m", "pipeline.train"], cwd=PROJECT_ROOT, check=True)
