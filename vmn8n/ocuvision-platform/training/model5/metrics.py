from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np

def compute_metrics(y_true,y_pred):
    acc = accuracy_score(y_true,y_pred)
    prec,rec,f1,_ = precision_recall_fscore_support(y_true,y_pred,average='macro', zero_division=0)
    return {'accuracy': acc, 'macro_precision': prec, 'macro_recall': rec, 'macro_f1': f1}
