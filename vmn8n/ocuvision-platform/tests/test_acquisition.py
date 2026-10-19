from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acquisition import archive_extractor, checksum

def test_sha256(tmp_path):
    p = tmp_path / 'a.txt'
    p.write_text('hello')
    h = checksum.sha256_file(p)
    assert len(h) == 64

def test_zip_safe(tmp_path):
    import zipfile
    z = tmp_path / 'bad.zip'
    with zipfile.ZipFile(z, 'w') as f:
        f.writestr('../evil.txt', 'x')
    out = tmp_path / 'out'
    extracted = archive_extractor.extract_zip(z, out)
    # should not extract outside
    assert not (tmp_path / 'evil.txt').exists()
