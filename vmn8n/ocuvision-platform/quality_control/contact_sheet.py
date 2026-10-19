from PIL import Image, ImageDraw
from pathlib import Path
from typing import List

def make_sheet(images: List[Path], out: Path, cols: int = 4, rows: int = 2, thumb: int = 128):
    out.parent.mkdir(parents=True, exist_ok=True)
    w,h = cols*thumb, rows*thumb
    sheet = Image.new('RGB',(w,h),(255,255,255))
    for i,p in enumerate(images[:cols*rows]):
        im = Image.open(p).convert('RGB').resize((thumb,thumb))
        sheet.paste(im, ((i%cols)*thumb,(i//cols)*thumb))
    sheet.save(out)
    return str(out)
