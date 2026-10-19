from collections import defaultdict

def check_leakage(rows):
    # group by case_id? stub
    cases = defaultdict(list)
    for r in rows:
        cases[r.get('case_id')].append(r.get('split'))
    leaks = []
    for cid, splits in cases.items():
        s = set(splits)
        if len(s)>1:
            leaks.append(cid)
    return {'potential_leaks': leaks}
