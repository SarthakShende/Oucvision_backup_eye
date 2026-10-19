import os
import sys
import shutil

RESERVE_GB = int(os.environ.get('OCUVISION_DISK_RESERVE_GB','25'))
def free_gb(path='/'):
    s = shutil.disk_usage(path)
    return s.free/(1024**3)
if free_gb() < RESERVE_GB:
    print(f'ERR: less than {RESERVE_GB}GB free', file=sys.stderr)
    sys.exit(1)
print('OK')
