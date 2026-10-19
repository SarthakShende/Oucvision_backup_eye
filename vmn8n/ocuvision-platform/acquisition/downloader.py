import os
import time
from pathlib import Path
from typing import Dict, Any
import requests

BLOCKED_HOSTS = {'localhost', '127.0.0.1', '::1'}
PRIVATE_NETS = []  # basic guard; can extend

def _is_safe_url(url: str) -> bool:
    if not url.lower().startswith(('http://', 'https://')):
        return False
    return True

def download_file(url: str, dest: Path, max_mb: int = 8192) -> Dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30, allow_redirects=True) as r:
        r.raise_for_status()
        total = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if not chunk: break
                total += len(chunk)
                if total > max_mb * 1024 * 1024:
                    raise ValueError(f"File exceeds max size {max_mb}MB")
                f.write(chunk)
    return {'path': str(dest), 'size_mb': round(total/(1024**2),2)}
