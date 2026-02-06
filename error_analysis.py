import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
def get_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    return pd.DataFrame(
        cm,
        index=[f"Actual_{l}" for l in labels],
        columns=[f"Pred_{l}" for l in labels]
    )
def compute_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics
def error_breakdown(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }
def classification_summary(y_true, y_pred, y_prob=None):
    return {
        "confusion_matrix": get_confusion_matrix(y_true, y_pred),
        "metrics": compute_classification_metrics(y_true, y_pred, y_prob),
        "error_breakdown": error_breakdown(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred)
    }