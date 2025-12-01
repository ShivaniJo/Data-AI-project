"""Responsible AI thesis code package.

This package exposes the core functionality used in the thesis,
organised into separate modules:

* ``config`` – central configuration for datasets and paths
* ``data_loader`` – functions to load and preprocess datasets
* ``fairness`` – group fairness metrics
* ``models`` – model training and feature importance utilities
* ``evaluation`` – model evaluation and fairness assessment
* ``plot_utils`` – helper functions for plotting figures
* ``main`` – script to run the full analysis pipeline

Import the modules as needed, for example:

    >>> from thesis_code.data_loader import load_and_preprocess
    >>> from thesis_code.models import train_logistic_regression

The package structure allows you to reuse individual components or
run the entire pipeline via ``python -m thesis_code.main``.
"""

__all__ = [
    "config",
    "data_loader",
    "fairness",
    "models",
    "evaluation",
    "plot_utils",
    "main",
]