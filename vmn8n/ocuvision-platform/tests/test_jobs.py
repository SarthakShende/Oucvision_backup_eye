from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OCUVISION_API_TOKEN','testtoken')
from fastapi.testclient import TestClient
from automation import vm_agent

client = TestClient(vm_agent.app)

def test_jobs_list_auth():
    r = client.get('/jobs', headers={'Authorization':'Bearer testtoken'})
    assert r.status_code == 200

def test_train_requires_approval():
    r = client.post('/jobs/train', json={'job_type':'train','configuration':{}}, headers={'Authorization':'Bearer testtoken'})
    assert r.status_code == 200
    assert r.json().get('requires_approval') == True

def test_cancel_404():
    r = client.post('/jobs/nonexist/cancel', headers={'Authorization':'Bearer testtoken'})
    assert r.status_code == 404
