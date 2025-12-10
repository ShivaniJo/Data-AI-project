"""Command‑line entry point for running the thesis analysis pipeline.

This script orchestrates loading datasets, training models, computing performance and fairness metrics and saving plots and CSV outputs.
It can be executed directly ("python -m thesis_code.main") to process all datasets defined in "config.DATASET_INFO".

Usage:
    python -m thesis_code.main

Outputs are saved into the directories defined in "config.py" """

from __future__ import annotations

import csv
from pathlib import Path
import json
import numpy as np

from . import config, data_loader, models, evaluation, plot_utils


def run_pipeline():
    
    # Prepare containers for aggregated results
    overall_metrics = []
    overall_fairness = []
    for dataset_name in config.DATASET_INFO.keys():
        print(f"Processing dataset: {dataset_name}")
        # Load data
        X_train, X_test, y_train, y_test, preprocessor, df_test = data_loader.load_and_preprocess(dataset_name)
        # Train logistic regression
        lr_model = models.train_logistic_regression(X_train, y_train)
        rf_model = models.train_random_forest(X_train, y_train)
        for model_name, model in [("LogisticRegression", lr_model), ("RandomForest", rf_model)]:
            # Evaluate performance
            perf = evaluation.evaluate_model(model, X_test, y_test)
            perf['dataset'] = dataset_name
            perf['model'] = model_name
            overall_metrics.append(perf)
            # Confusion matrix
            cm = evaluation.compute_confusion(model, X_test, y_test)
            cm_filename = config.FIGURES_DIR / f"{dataset_name}_{model_name}_confusion_matrix.png"
            plot_utils.plot_confusion_matrix(
                cm,
                class_names=["Negative", "Positive"],
                title=f"{dataset_name} {model_name} Confusion Matrix",
                filename=str(cm_filename),
            )
            # Feature importances
            feature_names = models.get_feature_names(preprocessor)
            importances = models.get_feature_importance(model, feature_names)
            fi_filename = config.FIGURES_DIR / f"{dataset_name}_{model_name}_feature_importance.png"
            plot_utils.plot_feature_importance(
                importances,
                title=f"{dataset_name} {model_name} Feature Importance",
                filename=str(fi_filename),
            )
            # Fairness metrics per protected attribute
            fairness_results = evaluation.evaluate_fairness(model, X_test, y_test, df_test, dataset_name, )
            for attr, metrics in fairness_results.items():
                overall_fairness.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'attribute': attr,
                    **metrics,
                })
                
                # === Group-specific fairness analysis ===
            from .fairness import group_fairness_metrics

            y_pred = model.predict(X_test)
            protected_attribute = df_test["sex"].values  # or "race"

            fairness_group_df = group_fairness_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            protected_attribute=protected_attribute
            )

            fairness_output_path = config.RESULTS_DIR / f"{dataset_name}_{model_name}_group_fairness.csv"
            fairness_group_df.to_csv(fairness_output_path)
            print(f"Saved group fairness metrics to {fairness_output_path}")


            metrics_csv = config.RESULTS_DIR / "performance_metrics.csv"
            fairness_csv = config.RESULTS_DIR / "fairness_metrics.csv"
            _write_dicts_to_csv(overall_metrics, metrics_csv)
            _write_dicts_to_csv(overall_fairness, fairness_csv)
    
            with open(config.RESULTS_DIR / "performance_metrics.json", "w") as f:
                json.dump(overall_metrics, f, indent=2)
            with open(config.RESULTS_DIR / "fairness_metrics.json", "w") as f:
                json.dump(overall_fairness, f, indent=2)
            print(f"Analysis complete. Results saved to {config.RESULTS_DIR}")


            def _write_dicts_to_csv(dict_list, path: Path):
   
                if not dict_list:
                 return
                 fieldnames = list(dict_list[0].keys())
                with open(path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                for row in dict_list:
                    writer.writerow(row)


                    if __name__ == "__main__":
                     run_pipeline()
