"""Fairness metrics computation.
This module provides functions to compute common group fairness metrics for binary classification. Metrics include statistical
parity difference, equal opportunity difference, equalised odds difference and predictive parity difference. These metrics help
evaluate disparities across groups defined by sensitive attributes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import confusion_matrix, precision_score
from fairness import group_fairness_metrics


def group_fairness_metrics(y_true, y_pred, protected_attribute): #Computes group-specific fairness metrics for each protected group.

      results = {}  # dictionary to store metrics for each group
    groups = np.unique(protected_attribute)  # find unique groups (e.g. male/female)
   
for group in groups:
        
        fairness_df = group_fairness_metrics(
    y_test.values,      # actual values
    y_pred,             # predicted values
    protected_attribute # sensitive attribute
)
 
        idx = (protected_attribute == group) #select rows
        y_true_g = y_true[idx] 
        y_pred_g = y_pred[idx]

        tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel() #confusion matrix for this group

        base_rate = y_true_g.mean() #base rate real positive
        positive_rate = y_pred_g.mean() #positive prediction rate
        tpr = tp / (tp + fn) if tp + fn > 0 else 0 #true positive rate
        fpr = fp / (fp + tn) if fp + tn > 0 else 0 #false positive rate
        precision = precision_score(y_true_g, y_pred_g, zero_division=0)

        results[group] = {
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Base Rate": base_rate,
            "Positive Rate": positive_rate,
            "TPR": tpr,
            "FPR": fpr,
            "Precision": precision
        } #store metrics 

    return pd.DataFrame(results).T

protected_attribute = df["sex"].values   # or "race"



"""
def _compute_group_rates(y_true: pd.Series, y_pred: pd.Series, sensitive: pd.Series) -> Dict[str, Dict[str, float]]:
    
    Returns per‑group metrics: positive rate, true positive rate (TPR),
    false positive rate (FPR) and positive predictive value (PPV).

    Parameters
    ----------
    y_true: pd.Series
        True binary labels.
    y_pred: pd.Series
        Predicted binary labels.
    sensitive: pd.Series
        Sensitive attribute values for each instance. Should align
        with y_true/y_pred.

    Returns
    -------
    dict
        A dictionary keyed by each group value containing metric
        values.
    
    metrics = {}
    # Ensure input is 1‑D arrays
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    sens_arr = np.asarray(sensitive)
    unique_groups = np.unique(sens_arr)
    for group in unique_groups:
        mask = sens_arr == group
        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]
        
        tp = np.sum((yp == 1) & (yt == 1))
        fp = np.sum((yp == 1) & (yt == 0))
        tn = np.sum((yp == 0) & (yt == 0))
        fn = np.sum((yp == 0) & (yt == 1))
        total = len(yt)
        # Positive rate: fraction of predicted positives
        positive_rate = np.mean(yp == 1) if total > 0 else 0.0
        # True positive rate (recall)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # Positive predictive value (precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[group] = {
            "positive_rate": positive_rate,
            "TPR": tpr,
            "FPR": fpr,
            "PPV": ppv,
        }
    return metrics


def statistical_parity_difference(y_true: pd.Series, y_pred: pd.Series, sensitive: pd.Series) -> float:
    Calculate the statistical parity difference.

    This is defined as the difference between the maximum and minimum positive rates across sensitive groups. A value close to zero
    indicates that the model predicts positive outcomes at similar rates across groups.
    rates = _compute_group_rates(y_true, y_pred, sensitive)
    positive_rates = [m["positive_rate"] for m in rates.values()]
    return max(positive_rates) - min(positive_rates)


def equal_opportunity_difference(y_true: pd.Series, y_pred: pd.Series, sensitive: pd.Series) -> float:
    Compute the equal opportunity difference.

    Equal opportunity requires that the true positive rate (TPR) be equal across sensitive groups. This metric returns the difference
     between the maximum and minimum TPR. 
    rates = _compute_group_rates(y_true, y_pred, sensitive)
    tprs = [m["TPR"] for m in rates.values()]
    return max(tprs) - min(tprs)


def equalised_odds_difference(y_true: pd.Series, y_pred: pd.Series, sensitive: pd.Series) -> float:
    Compute the equalised odds difference.

    Equalised odds requires that both TPR and FPR be equal across sensitive groups. This implementation computes the largest
    absolute disparity between any two groups for these two rates and returns the maximum of TPR and FPR differences.
    rates = _compute_group_rates(y_true, y_pred, sensitive)
    tprs = [m["TPR"] for m in rates.values()]
    fprs = [m["FPR"] for m in rates.values()]
    diff_tpr = max(tprs) - min(tprs)
    diff_fpr = max(fprs) - min(fprs)
    return max(diff_tpr, diff_fpr)


def predictive_parity_difference(y_true: pd.Series, y_pred: pd.Series, sensitive: pd.Series) -> float:
    Compute the predictive parity difference.

    Predictive parity requires that the positive predictive value(precision) be equal across sensitive groups. The metric
    returns the difference between the maximum and minimum PPV.
    rates = _compute_group_rates(y_true, y_pred, sensitive)
    ppvs = [m["PPV"] for m in rates.values()]
    return max(ppvs) - min(ppvs)
"""