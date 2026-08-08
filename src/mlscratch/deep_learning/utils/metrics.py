from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlscratch.deep_learning.utils.utils import save_metrics

def compute_metrics_and_confmat(y_true, y_pred, output_dir, threshold=0.5):
  
    # y_pred = (y_probs >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0, average='macro'))
    rec = float(recall_score(y_true, y_pred, zero_division=0, average='macro'))
    f1 = float(f1_score(y_true, y_pred, zero_division=0, average='macro'))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    tnr = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'true_negative_rate': float(tnr),
        'recall': rec,
        'f1_score': f1
    }, {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    
    return cm, metrics