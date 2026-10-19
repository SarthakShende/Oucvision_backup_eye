from pathlib import Path
from typing import Dict, Any
from PIL import Image
import cv2
import numpy as np

SUPPORTED = {'jpg','jpeg','png','tiff','bmp'}

def validate_image(path: Path, min_size: int = 16) -> Dict[str, Any]:
    try:
        img = Image.open(path)
        img.load()
        w,h = img.size
        if w < min_size or h < min_size:
            return {'ok': False, 'error': 'too_small'}
        arr = np.array(img)
        return {'ok': True, 'width': w, 'height': h, 'format': img.format, 'channels': arr.shape[2] if arr.ndim==3 else 1}
    except Exception as e:
        return {'ok': False, 'error': str(e)}
