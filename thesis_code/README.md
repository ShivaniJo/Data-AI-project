# Responsible AI Thesis Code

This repository contains a modular codebase for the *Responsible AI* thesis.
The goal of the code is to analyse tabular datasets for predictive
modelling, evaluate traditional performance metrics, compute
group‑based fairness metrics and visualise model behaviour.

## Structure

```
thesis_code/
  ├── __init__.py        # Package initialisation
  ├── config.py          # Dataset paths and project configuration
  ├── data_loader.py     # Data reading and preprocessing
  ├── fairness.py        # Fairness metric implementations
  ├── models.py          # Model training and feature importance
  ├── evaluation.py      # Performance and fairness evaluation
  ├── plot_utils.py      # Plotting helper functions
  ├── main.py            # Pipeline orchestration script
  └── README.md          # This file
```

## Prerequisites

- Python 3.8 or later
- [pandas](https://pandas.pydata.org/)
- [scikit‑learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/)

You can install the required packages via pip:

```bash
pip install pandas scikit-learn matplotlib
```

## Usage

1. **Add your datasets**: Place your CSV files in the `data/` directory
   located in the repository root. The expected filenames and
   protected attributes are defined in `config.py`. Update
   `config.DATASET_INFO` if your filenames or target column names
   differ.

2. **Run the pipeline**:

  run the below command in root directory:
   ```bash
   python -m thesis_code.main
   ```

   The script will iterate over all datasets listed in
   `config.DATASET_INFO`, perform preprocessing, train a logistic
   regression and a random forest model, evaluate performance and
   fairness, and save results into the `results/` directory. Plots
   will be saved under `results/figures/`.

3. **Inspect outputs**: After running, you will find:

   - `performance_metrics.csv` and `.json`: tables of accuracy,
     precision, recall, F1 and ROC‑AUC for each dataset and model.
   - `fairness_metrics.csv` and `.json`: group fairness metrics
     (statistical parity difference, equal opportunity difference,
     equalised odds difference, predictive parity difference) for
     each protected attribute.
   - Confusion matrix and feature importance plots saved as PNG files.

## Extending the Code

- **New datasets**: Add an entry to `config.DATASET_INFO` with the
  filename, target column and list of protected attribute columns.
- **Different models**: Implement additional training functions in
  `models.py` and update `main.py` to include them.
- **Fairness metrics**: Extend `fairness.py` with additional
  definitions and modify `evaluation.evaluate_fairness` to call
  them.

This modular design enables you to experiment with different
configurations and extend the analysis.