"""Plotting utilities for thesis figures.

This module contains helper functions for creating and saving
confusion matrix heatmaps and feature importance bar charts.
Matplotlib is used to generate publication‑quality figures. Colors
are not explicitly set to comply with guidelines.
"""

from __future__ import annotations

# Use non-interactive backend to avoid GUI-related errors (e.g., Tkinter
# "main thread is not in main loop" exceptions on some systems).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, filename: str):
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm: np.ndarray
        A 2x2 confusion matrix [[TN, FP], [FN, TP]].
    class_names: list of str
        Names of the negative and positive classes in order.
    title: str
        Plot title.
    filename: str
        Path to save the PNG image.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_feature_importance(importances: List[Tuple[str, float]], title: str, filename: str, top_n: int = 20):
    """Plot and save the top_n feature importances as a bar chart.

    Parameters
    ----------
    importances: list of (feature_name, importance)
        A list of tuples sorted by importance descending.
    title: str
        Title of the plot.
    filename: str
        Path to save the plot.
    top_n: int, default 20
        Number of top features to display.
    """
    top_features = importances[:top_n]
    names = [f for f, _ in top_features]
    values = [v for _, v in top_features]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top_features) * 0.3)))
    y_positions = np.arange(len(names))
    ax.barh(y_positions, values)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
