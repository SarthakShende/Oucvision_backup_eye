from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quality_control import leakage_check

def test_no_leak():
    rows=[{'case_id':'c1','split':'train'},{'case_id':'c2','split':'val'}]
    assert leakage_check.check_leakage(rows)['potential_leaks']==[]

def test_leak():
    rows=[{'case_id':'c1','split':'train'},{'case_id':'c1','split':'test'}]
    assert 'c1' in leakage_check.check_leakage(rows)['potential_leaks']
