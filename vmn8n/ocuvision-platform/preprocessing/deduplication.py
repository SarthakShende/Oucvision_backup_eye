from pathlib import Path
import imagehash
from PIL import Image

def phash_image(path: Path, hash_size: int = 8) -> str:
    img = Image.open(path)
    h = imagehash.phash(img, hash_size=hash_size)
    return str(h)

def dhash_image(path: Path, hash_size: int = 8) -> str:
    img = Image.open(path)
    h = imagehash.dhash(img, hash_size=hash_size)
    return str(h)
