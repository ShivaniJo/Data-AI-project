"""Model evaluation functions.

This module encapsulates evaluation logic for classification models,including standard performance metrics and group fairness metrics."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from . import fairness, config


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Compute standard performance metrics for a classifier.Returns a dictionary containing accuracy, precision, recall,F1 and 
    ROC‑AUC scores. For ROC‑AUC, the positive class is assumed to be encoded as 1."""
    y_pred = model.predict(X_test)
    # Some metrics require probability estimates
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = np.nan
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc,
    }
    return metrics


def compute_confusion(model, X_test, y_test) -> np.ndarray:
    """Compute the confusion matrix for the test set.
    Returns a 2x2 array [[TN, FP], [FN, TP]]."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def evaluate_fairness(model, X_test, y_test, df_test, dataset_name: str) -> Dict[str, Dict[str, float]]:
    """Evaluate fairness metrics for all protected attributes in a dataset.

    Parameters
    ----------
    model:
        Fitted classifier.
    X_test:
        Preprocessed features for the test set.
    y_test:
        True binary labels for the test set.
    df_test:
        Original (unprocessed) test DataFrame, containing sensitive
        attributes.
    dataset_name: str
        Name of the dataset to lookup protected attributes in config.

    Returns
    -------
    dict
        A nested dictionary mapping each sensitive attribute to its
        fairness metrics.
    """
    y_pred = model.predict(X_test)
    info = config.DATASET_INFO[dataset_name]
    results: Dict[str, Dict[str, float]] = {}
    for attr in info['protected_attrs']:
        sensitive_values = df_test[attr]
        spd = fairness.statistical_parity_difference(y_test, y_pred, sensitive_values)
        eod = fairness.equal_opportunity_difference(y_test, y_pred, sensitive_values)
        eodds = fairness.equalised_odds_difference(y_test, y_pred, sensitive_values)
        ppd = fairness.predictive_parity_difference(y_test, y_pred, sensitive_values)
        results[attr] = {
            'statistical_parity_difference': spd,
            'equal_opportunity_difference': eod,
            'equalised_odds_difference': eodds,
            'predictive_parity_difference': ppd,
        }
    return results
