def score_quality(w,h,sharpness=None,exposure=None,contrast=None):
    # stub scoring 0-3
    if w<64 or h<64: return 1
    if w>=256 and h>=256: return 3
    return 2
