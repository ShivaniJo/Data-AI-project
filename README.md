# Fairness-Aware Income Prediction using Machine Learning

This repository contains the implementation of a **fairness-aware machine learning pipeline** developed as part of a **Master’s thesis project**.  
The project focuses on predicting income levels using the Adult Income Dataset while **systematically analyzing algorithmic fairness across demographic groups**.

## 📌 Project Objectives

- Build machine learning models to predict whether an individual earns more than \$50K/year.
- Evaluate standard performance metrics (accuracy, confusion matrix).
- Perform **group-specific fairness analysis** across sensitive attributes (e.g., sex, race).
- Analyze how fairness metrics evolve across different models and datasets.
- Provide reproducible, research-quality results suitable for academic evaluation.

## 📊 Dataset

The project uses the **Adult Income Dataset**, which contains demographic and employment-related features such as:
- Age, education, occupation
- Gender, race
- Capital gain/loss
- Income label (`<=50K`, `>50K`)

Two versions of the dataset are used:
- **Raw dataset** (`adult.csv`)
- **Preprocessed dataset** (`Adult pre-processed dataset.csv`)

## 🧠 Models Implemented

- Logistic Regression  
- Random Forest  

Each model is evaluated on:
- Predictive performance
- Feature importance
- Group-level fairness metrics

## ⚖️ Fairness Evaluation

To assess fairness, the project computes **group-specific metrics** for protected attributes such as **sex** and **race**.

### Group-Specific Metrics:
- Confusion Matrix (TN, FP, FN, TP)
- Base Rate
- Positive Prediction Rate
- True Positive Rate (TPR)
- False Positive Rate (FPR)
- Precision

These metrics help identify potential biases and evaluate fairness notions such as:
- **Equal Opportunity**
- **Equalized Odds**

Fairness results are saved as CSV files for transparency and further analysis.

## 📁 Project Structure

thesis_code/
│
├── main.py # Entry point for running the full pipeline
├── config.py # Configuration for datasets and output paths
├── data_loader.py # Data loading and preprocessing
├── models.py # Model training and evaluation functions
├── evaluation.py # Performance and fairness evaluation logic
├── fairness.py # Group-specific fairness metrics
├── plot_utils.py # Confusion matrix and feature importance plots
├── visualizations.py # Fairness metric visualizations
│
├── results/ # CSV and JSON outputs
└── figures/ # Generated plots

## ▶️ How to Run the Project

### 1. Install dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn

📈 Outputs
After execution, the following outputs are generated:
📂 results/
performance_metrics.csv
fairness_metrics.csv
Group-specific fairness reports per dataset and model

📂 figures/
Confusion matrix plots
Feature importance plots
All outputs are automatically saved for reproducibility.

🎓 Academic Context
This project was developed as part of a Master’s thesis focusing on:
Ethical AI
Bias and fairness in machine learning
Responsible data-driven decision-making
The implementation follows best practices for modularity, reproducibility, and interpretability.

📜 License

This project is intended for academic and research use.
Please cite appropriately if used in publications or derivative works.
