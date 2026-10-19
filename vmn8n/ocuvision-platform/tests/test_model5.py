from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.model5 import train

def test_build():
    import torch
    m = train.build_model(13)
    x = torch.randn(2,3,224,224)
    y = m(x)
    assert y.shape == (2,13)
