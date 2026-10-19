import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from automation import vm_agent

client = TestClient(vm_agent.app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_jobs_unauth():
    r = client.post("/jobs/discover-manifest", json={"job_type": "discover-manifest", "configuration": {}})
    assert r.status_code == 403 or r.status_code == 401
