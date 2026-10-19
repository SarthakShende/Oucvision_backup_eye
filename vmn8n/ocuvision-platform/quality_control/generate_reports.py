import json
from pathlib import Path

def write_report(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f:
        json.dump(data,f,indent=2)
