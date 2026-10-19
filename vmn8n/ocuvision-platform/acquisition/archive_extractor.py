import os
import tarfile
import zipfile
from pathlib import Path
from typing import List

def _is_safe_member(base: Path, member_path: Path) -> bool:
    try:
        resolved = (base / member_path).resolve()
        base_res = base.resolve()
        return str(resolved).startswith(str(base_res) + os.sep) or resolved == base_res
    except Exception:
        return False

def extract_zip(zip_path: Path, dest: Path, max_expand_mb: int = 2048) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            p = Path(m)
            if p.is_absolute() or '..' in p.parts:
                continue
            if not _is_safe_member(dest, p):
                continue
            z.extract(m, dest)
            extracted.append(dest / p)
    return extracted

def extract_tar(tar_path: Path, dest: Path, max_expand_mb: int = 2048) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    extracted = []
    mode = 'r:*'
    with tarfile.open(tar_path, mode) as t:
        for m in t.getmembers():
            if m.islnk() or m.issym():
                continue
            name = m.name
            p = Path(name)
            if p.is_absolute() or '..' in p.parts:
                continue
            if not _is_safe_member(dest, p):
                continue
            t.extract(m, dest, filter='data' if hasattr(tarfile, 'data_filter') else None)
            extracted.append(dest / p)
    return extracted
