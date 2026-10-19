from PIL import Image
from pathlib import Path

def normalize_save(src: Path, dst: Path, size: int = 224, preserve_aspect: bool = True):
    img = Image.open(src).convert('RGB')
    if preserve_aspect:
        w,h = img.size
        scale = size / max(w,h)
        nw,nh = int(w*scale), int(h*scale)
        img = img.resize((nw,nh))
        # pad to square
        from PIL import ImageOps
        img = ImageOps.pad(img, (size,size), color=(0,0,0))
    else:
        img = img.resize((size,size))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, 'JPEG', quality=90)
    return str(dst)
