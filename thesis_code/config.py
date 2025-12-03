"""Configuration module for the thesis project."""

from pathlib import Path

# Base directory for data files. When you run the code, ensure that
# the dataset CSVs reside in this directory. You can modify this
# path or provide absolute paths in the dataset_info entries below.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Directory where results (metrics tables) will be saved.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Directory where generated figures will be stored.
FIGURES_DIR = RESULTS_DIR / "figures"

# Metadata for each dataset used in the thesis. The keys represent dataset identifiers, and each value contains information needed
# the target column name, and the list of protected attributes to assess fairness on. Add or modify entries as needed.

DATASET_INFO = {
    # UCI Adult Income dataset. "income" column contains the target
    # classes "<=50K" and ">50K" and will be automatically mapped to 0/1.
    "adult": {
        
        "filename": "adult.csv",
        "target": "income",
        # Protected attributes on which fairness metrics will be computed.
        "protected_attrs": ["gender", "race"],
    },
    # Financial Loan Access dataset. "Loan_Approved" column contains
    # values like "Approved" and "Denied", which will be mapped to 0/1.
    "loan": {
        "filename": "Financial_Loan_Access_Dataset.csv",
        "target": "Loan_Approved",
        "protected_attrs": ["Gender", "Race"],
    },

    
    "german": {
        "filename": "german_credit_data.csv",
        "target": "Creditworthiness",  # replace with actual column name
        "protected_attrs": ["Sex"],
    },
}

# Ensure result directories exist when the module is imported.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def get_dataset_path(name: str) -> Path:
    """Return the full path to the CSV file for a given dataset name.

    Parameters
    ----------
    name: str
        The key of the dataset in DATASET_INFO.

    Returns
    -------
    pathlib.Path
        Path to the CSV file.
    """
    info = DATASET_INFO.get(name)
    if info is None:
        raise KeyError(f"Unknown dataset: {name}. Available datasets: {list(DATASET_INFO.keys())}")
    return DATA_DIR / info["filename"]
