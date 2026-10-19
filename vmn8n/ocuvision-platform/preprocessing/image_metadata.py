import csv
import json
from pathlib import Path
from typing import List, Dict

FIELDNAMES = [
 'image_id','case_id','dataset_id','source_image_id','class_id','class_name',
 'original_label','original_image_path','processed_image_path','width','height','format',
 'sha256','perceptual_hash','split','diagnosis_evidence','annotation_confidence',
 'image_modality','licence','quality_status'
]

def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,'') for k in FIELDNAMES})
