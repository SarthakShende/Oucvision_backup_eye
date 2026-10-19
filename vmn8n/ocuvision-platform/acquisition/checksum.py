import hashlib
from pathlib import Path
from typing import Tuple

def sha256_file(path: Path, chunk_size: int = 1024*1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def verify_sha256(path: Path, expected: str) -> Tuple[bool, str]:
    actual = sha256_file(path)
    return (actual.lower() == str(expected).lower()), actual
