import numpy as np
from sklearn.metrics import (auc, precision_recall_curve, roc_curve,
                             accuracy_score, f1_score, recall_score, precision_score)


def count_accuracy(B_true, B_prob, ignore_diag=True):
    """Compute causal discovery metrics: AUROC, AUPRC, F1, Precision, Recall, ACC, SHD."""
    if hasattr(B_true, 'cpu'):
        B_true = B_true.cpu().numpy()
    if hasattr(B_prob, 'cpu'):
        B_prob = B_prob.cpu().numpy()

    # Handle dynamic GT of shape (N, N, T) or (T, N, N)
    if B_true.ndim == 3:
        if B_true.shape[0] == B_true.shape[1]:  # (N, N, T)
            N, _, T = B_true.shape
            mask2d = ~np.eye(N, dtype=bool) if ignore_diag else np.ones((N, N), dtype=bool)
            mask = np.broadcast_to(mask2d[..., None], (N, N, T))
        else:  # (T, N, N)
            T, N, _ = B_true.shape
            mask2d = ~np.eye(N, dtype=bool) if ignore_diag else np.ones((N, N), dtype=bool)
            mask = np.broadcast_to(mask2d, (T, N, N))
        true_flat = B_true[mask].flatten()
        prob_flat = B_prob[mask].flatten()
    else:
        n = B_true.shape[0]
        mask = ~np.eye(n, dtype=bool) if ignore_diag else np.ones((n, n), dtype=bool)
        true_flat = B_true[mask].flatten()
        prob_flat = B_prob[mask].flatten()

    true_flat = true_flat.astype(int)

    if len(np.unique(true_flat)) < 2:
        auroc = 0.5
        auprc = 0.0
    else:
        fpr, tpr, _ = roc_curve(true_flat, prob_flat)
        auroc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(true_flat, prob_flat)
        auprc = auc(recall_curve, precision_curve)

    # Find best threshold
    is_binary = np.all(np.isin(prob_flat, [0, 1]))
    best = {'F1': 0.0, 'Precision': 0.0, 'Recall': 0.0,
            'ACC': 0.0, 'SHD': 0, 'Threshold': 0.5}

    if is_binary:
        pred_binary = prob_flat.astype(int)
        best = _compute_binary_metrics(true_flat, pred_binary)
        best['Threshold'] = 0.5
    else:
        thresholds = np.unique(prob_flat) if len(prob_flat) < 1000 \
                     else np.linspace(prob_flat.min(), prob_flat.max(), 100)
        thresholds = thresholds[1:-1] if len(thresholds) > 2 else thresholds
        if len(thresholds) == 0:
            thresholds = [0.5]

        max_f1 = -1
        for th in thresholds:
            pred_binary = (prob_flat > th).astype(int)
            f1 = f1_score(true_flat, pred_binary, zero_division=0)
            if f1 > max_f1:
                max_f1 = f1
                current = _compute_binary_metrics(true_flat, pred_binary)
                current['Threshold'] = th
                best = current

    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'F1': float(best['F1']),
        'Precision': float(best['Precision']),
        'Recall': float(best['Recall']),
        'ACC': float(best['ACC']),
        'SHD': int(best['SHD']),
        'Best_Threshold': float(best['Threshold'])
    }


def _compute_binary_metrics(true_flat, pred_flat):
    return {
        'F1': float(f1_score(true_flat, pred_flat, zero_division=0)),
        'Precision': float(precision_score(true_flat, pred_flat, zero_division=0)),
        'Recall': float(recall_score(true_flat, pred_flat, zero_division=0)),
        'ACC': float(accuracy_score(true_flat, pred_flat)),
        'SHD': int(np.sum(np.abs(true_flat - pred_flat)))
    }
