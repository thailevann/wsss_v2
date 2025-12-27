"""
Classification metrics and threshold tuning
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix


def _to_threshold_array(threshold, C):
    """Convert threshold to array format"""
    if np.isscalar(threshold):
        return np.full((C,), float(threshold), dtype=float)
    thr = np.asarray(threshold, dtype=float).reshape(-1)
    assert thr.shape[0] == C
    return thr


def compute_cls_metrics(y_true, y_prob, threshold=0.5, class_names=None):
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels [N, C]
        y_prob: Predicted probabilities [N, C]
        threshold: Threshold(s) for binary classification
        class_names: List of class names
        
    Returns:
        summary: Dictionary with summary metrics
        df: DataFrame with per-class metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(y_prob.shape[1])]
    
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    C = y_true.shape[1]
    thr = _to_threshold_array(threshold, C)
    y_pred = (y_prob >= thr[None, :]).astype(int)

    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    auc_per_class, ap_per_class = [], []
    for c in range(C):
        try:
            auc = roc_auc_score(y_true[:, c], y_prob[:, c])
        except Exception:
            auc = 0.5
        try:
            ap = average_precision_score(y_true[:, c], y_prob[:, c])
        except Exception:
            ap = 0.0
        auc_per_class.append(float(auc))
        ap_per_class.append(float(ap))

    summary = {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "auc_macro": float(np.mean(auc_per_class)),
        "map_macro": float(np.mean(ap_per_class)),
    }

    rows = []
    for c in range(C):
        cm = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        rows.append({
            "class": class_names[c],
            "threshold": float(thr[c]),
            "f1": float(f1),
            "auroc": float(auc_per_class[c]),
            "ap": float(ap_per_class[c]),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        })

    return summary, pd.DataFrame(rows)


def tune_thresholds_per_class(y_true, y_prob, grid_points=101):
    """
    Tune thresholds per class using grid search
    
    Args:
        y_true: Ground truth labels [N, C]
        y_prob: Predicted probabilities [N, C]
        grid_points: Number of grid points to search
        
    Returns:
        best_thr: Best threshold for each class [C]
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    C = y_true.shape[1]
    best_thr = np.full((C,), 0.5, dtype=float)
    grid = np.linspace(0.0, 1.0, int(grid_points))

    for c in range(C):
        yt, yp = y_true[:, c], y_prob[:, c]
        if yt.min() == yt.max():
            continue
        best_s, best_t = -1.0, 0.5
        for t in grid:
            pred = (yp >= t).astype(int)
            s = f1_score(yt, pred, zero_division=0)
            if s > best_s:
                best_s, best_t = float(s), float(t)
        best_thr[c] = best_t
    return best_thr

