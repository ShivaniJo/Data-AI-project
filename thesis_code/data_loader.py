"""Data loading and preprocessing utilities.

This module defines functions to load tabular datasets from CSV files,
handle missing values, encode categorical variables, scale numeric
features and split the data into training and test sets. It relies on
pandas for I/O and scikit‑learn for preprocessing. The metadata for
each dataset (file names, targets and protected attributes) is
defined in `config.py`.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from . import config


def _build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    """Construct a ColumnTransformer for the given dataframe.

    The preprocessor imputes missing values, one‑hot encodes
    categorical features and scales numerical features. It
    automatically determines which columns are numeric and which are
    categorical based on pandas dtypes. The target column is
    excluded from preprocessing.

    Parameters
    ----------
    df: pd.DataFrame
        The raw dataset.
    target: str
        Name of the target column.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A fitted preprocessor ready to transform feature matrices.
    """
    # Identify categorical and numeric columns excluding the target
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c != target]
    numeric_cols = [c for c in df.columns if df[c].dtype != 'object' and c != target]

    # Pipelines for numeric and categorical columns
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def load_and_preprocess(name: str, test_size: float = 0.3, random_state: int = 42):
    """Load a dataset by name and prepare train/test splits.

    This function reads the dataset CSV file specified in
    `config.DATASET_INFO`, constructs a preprocessing pipeline
    appropriate for the feature types, fits the preprocessor on
    training data and transforms both training and test sets. The
    test split is stratified on the target to preserve class
    distribution.

    Parameters
    ----------
    name: str
        The dataset identifier defined in `config.DATASET_INFO`.
    test_size: float, default=0.3
        Fraction of the data to reserve for testing.
    random_state: int, default=42
        Seed for the random train/test split.

    Returns
    -------
    X_train: pd.DataFrame or numpy.ndarray
        Preprocessed training features.
    X_test: pd.DataFrame or numpy.ndarray
        Preprocessed test features.
    y_train: pd.Series
        Training target labels.
    y_test: pd.Series
        Test target labels.
    preprocessor: ColumnTransformer
        The fitted preprocessing transformer.
    df_test: pd.DataFrame
        The original test dataframe (before preprocessing), useful for
        computing fairness metrics on sensitive attributes.
    """
    info = config.DATASET_INFO.get(name)
    if info is None:
        raise KeyError(f"Unknown dataset {name}; available datasets: {list(config.DATASET_INFO.keys())}")
    path = config.get_dataset_path(name)
    # Load CSV; assume first row is header and comma‑separated.
    df = pd.read_csv(path)
    target = info["target"]
    # Drop rows with missing target values
    df = df.dropna(subset=[target])
    y = df[target]
    # Convert non-numeric binary target values to 0/1 automatically.
    # Some datasets encode the target as strings (e.g. '<=50K' vs '>50K').
    # We map the lexicographically sorted unique values to 0 and 1.
    if not pd.api.types.is_numeric_dtype(y):
        unique_vals = y.unique()
        if len(unique_vals) == 2:
            # Sort values to ensure consistent ordering
            sorted_vals = sorted(list(unique_vals))
            mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
            y = y.map(mapping)
        else:
            raise ValueError(
                f"Target column '{target}' has non-numeric values with {len(unique_vals)} unique classes. "
                "Please ensure it is binary or encode it manually."
            )
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Build and fit preprocessor on training data
    preprocessor = _build_preprocessor(X_train, target=None)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    # Return original test df for fairness metrics
    df_test = X_test.copy()
    df_test[target] = y_test
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, df_test
