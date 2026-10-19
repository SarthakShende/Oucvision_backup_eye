def audit(rows):
    by_cls = {}
    by_split = {'train':0,'val':0,'test':0}
    for r in rows:
        by_cls[r['class_id']] = by_cls.get(r['class_id'],0)+1
        by_split[r['split']] = by_split.get(r['split'],0)+1
    return {'counts_by_class': by_cls, 'counts_by_split': by_split}
