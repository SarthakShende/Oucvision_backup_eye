import json
from pathlib import Path

CLS_PATH = Path(__file__).resolve().parent.parent / 'config' / 'classes.json'

def load_classes():
    with open(CLS_PATH) as f:
        return json.load(f)

def map_label(label: str):
    cls = load_classes()
    cid = cls['class_ids'].get(str(label).strip().lower())
    if cid is None:
        return None, None
    return cid, cls['classes'][str(cid)]
