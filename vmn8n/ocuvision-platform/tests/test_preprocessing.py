from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from preprocessing import image_validator, label_mapper
from PIL import Image

def test_validate(tmp_path):
    p = tmp_path/'t.jpg'
    Image.new('RGB',(32,32)).save(p)
    r = image_validator.validate_image(p)
    assert r['ok']

def test_map():
    cid,name = label_mapper.map_label('normal')
    assert cid==0 and name=='normal'
    assert label_mapper.map_label('unknown') == (None,None)
