"""
Metric functions cho T1 và T4.

T1 (mục 10): Accuracy, Macro-F1, F1 per class, Confusion matrix, Precision, Recall
T4 (mục 11): Micro-F1, Macro-F1, Per-label F1, Exact match ratio, Precision/Recall per label
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import T1_LABELS, TACTIC_LABELS


# ============================================================
# T1 Metrics – Multi-class, Single-label
# ============================================================
def compute_t1_metrics(eval_pred) -> dict:
    """
    Compute metrics cho Trainer callback (T1).
    eval_pred: EvalPrediction(predictions, label_ids)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(T1_LABELS))), zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

    for i, label_name in enumerate(T1_LABELS):
        metrics[f"f1_{label_name}"] = f1_per_class[i]
        metrics[f"precision_{label_name}"] = precision[i]
        metrics[f"recall_{label_name}"] = recall[i]

    return metrics


def print_t1_report(labels: np.ndarray, preds: np.ndarray):
    """In báo cáo chi tiết cho T1."""
    print("\n" + "=" * 60)
    print("T1 – SCAM DETECTION REPORT")
    print("=" * 60)
    print(classification_report(
        labels, preds,
        target_names=T1_LABELS,
        digits=4,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    cm = confusion_matrix(labels, preds, labels=list(range(len(T1_LABELS))))
    print(f"{'':>12s}", end="")
    for name in T1_LABELS:
        print(f"{name:>12s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{T1_LABELS[i]:>12s}", end="")
        for val in row:
            print(f"{val:>12d}", end="")
        print()


# ============================================================
# T4 Metrics – Multi-label
# ============================================================
def compute_t4_metrics(eval_pred, threshold: float = 0.5) -> dict:
    """
    Compute metrics cho Trainer callback (T4).
    Dùng sigmoid + threshold để chuyển logits → nhãn dự đoán.
    """
    logits, labels = eval_pred

    # Sigmoid → xác suất
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    # Micro-F1 và Macro-F1
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # Exact match ratio (subset accuracy)
    exact_match = np.mean(np.all(preds == labels, axis=1))

    metrics = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "exact_match": exact_match,
    }

    # Per-label metrics
    for i, label_name in enumerate(TACTIC_LABELS):
        if labels.shape[0] > 0:
            label_f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
            label_prec = precision_recall_fscore_support(
                labels[:, i], preds[:, i], average="binary", zero_division=0
            )
            metrics[f"f1_{label_name}"] = label_f1
            metrics[f"precision_{label_name}"] = label_prec[0]
            metrics[f"recall_{label_name}"] = label_prec[1]

    return metrics


def print_t4_report(labels: np.ndarray, preds: np.ndarray, threshold: float = 0.5):
    """In báo cáo chi tiết cho T4."""
    if preds.max() <= 1.0 and preds.min() >= 0:
        # Đã là probability
        probs = preds
    else:
        # Còn là logits
        probs = 1 / (1 + np.exp(-preds))
    preds_binary = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    print("\n" + "=" * 60)
    print(f"T4 – TACTIC CLASSIFICATION REPORT (threshold={threshold})")
    print("=" * 60)

    micro_f1 = f1_score(labels, preds_binary, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds_binary, average="macro", zero_division=0)
    exact_match = np.mean(np.all(preds_binary == labels, axis=1))

    print(f"  Micro-F1:     {micro_f1:.4f}")
    print(f"  Macro-F1:     {macro_f1:.4f}")
    print(f"  Exact Match:  {exact_match:.4f}")
    print()

    print(f"{'Label':>15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 55)
    for i, label_name in enumerate(TACTIC_LABELS):
        p, r, f, s = precision_recall_fscore_support(
            labels[:, i], preds_binary[:, i],
            average="binary", zero_division=0,
        )
        support = labels[:, i].sum()
        print(f"{label_name:>15s} {p:>10.4f} {r:>10.4f} {f:>10.4f} {support:>10.0f}")
