import matplotlib.pyplot as plt

def plot_group_fairness(fairness_df):
    metrics = ["Base Rate", "Positive Rate", "TPR", "FPR", "Precision"]

    for metric in metrics:
        fairness_df[metric].plot(kind="bar")
        plt.title(f"{metric} by Group")
        plt.xlabel("Group")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()