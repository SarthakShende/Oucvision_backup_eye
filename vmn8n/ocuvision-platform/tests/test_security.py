from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_classes_ids():
    import json
    p = ROOT/'config/classes.json'
    d = json.load(open(p))
    assert d['classes']['0']=='normal'
    assert len(d['classes'])==13

def test_env_example_has_token():
    p = ROOT/'.env.example'
    s = p.read_text()
    assert 'OCUVISION_API_TOKEN' in s
