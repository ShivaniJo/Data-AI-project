"""Model training utilities.

This module provides simple wrappers for fitting scikit‑lear classifiers used in the thesis. It also exposes helpers to
extract feature importances from fitted models and to build pipelines combining preprocessing with estimators."""

from __future__ import annotations

from typing import Tuple, List
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


"""def train_logistic_regression(X_train, y_train, **kwargs) -> LogisticRegression:
    Train a logistic regression classifier.

    The classifier uses L2 regularisation and the saga solver for efficiency. Class weights are balanced by default to handle
    imbalanced datasets. Additional keyword arguments are passed tothe scikit‑learn constructor.

    Returns
    -------
    LogisticRegression
        The fitted logistic regression model.
    
    params = {
        'penalty': 'l2',
        'solver': 'saga',
        'max_iter': 500,
        'class_weight': 'balanced',
        'n_jobs': -1,
    }
    params.update(kwargs)
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model
"""
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

"""def train_random_forest(X_train, y_train, **kwargs) -> RandomForestClassifier:
    Train a random forest classifier.

    By default, the forest uses 200 trees and balances class weights. Additional hyperparameters can be supplied via kwargs.

    Returns
    -------
    RandomForestClassifier
        The fitted random forest model.
    
    params = {
        'n_estimators': 200,
        'n_jobs': -1,
        'class_weight': 'balanced',
        'random_state': 42,
    }
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model
"""
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def get_feature_names(preprocessor) -> List[str]:
    """Retrieve the transformed feature names from a ColumnTransformer.
    This helper extracts feature names from the fitted preprocessor by querying the underlying transformers. Numeric column names are
    returned as is; categorical columns are expanded to include one‑hot encoded categories."""
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'remainder':
            continue
        if hasattr(transformer, 'named_steps'):
            # Pipeline: extract from the last step
            encoder = transformer.named_steps.get('encoder')
            if encoder is not None:
                encoded_cols = encoder.get_feature_names_out(cols)
                feature_names.extend(encoded_cols)
            else:
                # Numeric pipeline: just use the original column names
                feature_names.extend(cols)
        else:
            # Single transformer
            feature_names.extend(cols)
    return feature_names


def get_feature_importance(model, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Compute feature importance for a fitted model.

    For logistic regression, the absolute value of the coefficient is used as importance. For random forests, the `feature_importances_`
    attribute is used. Returns a list of (feature_name, importance)sorted in descending order of importance."""
    if hasattr(model, 'coef_'):
        # Logistic regression: model.coef_ is shape (1, n_features)
        importances = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model type does not support feature importance extraction")
    pairs = list(zip(feature_names, importances))
    return sorted(pairs, key=lambda x: x[1], reverse=True)

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy, preds