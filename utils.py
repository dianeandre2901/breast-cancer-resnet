"""
utils.py
--------
Utility functions for evaluation metrics, plotting confusion matrices, and generating ROC curves
for binary classification of breast cancer histopathology images.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def plot_confusion_matrix(y_true, y_pred, class_labels, filename="confusion_matrix.pdf"):
    """
    Plots and saves a confusion matrix heatmap.

    Args:
        y_true (list or np.ndarray): True labels
        y_pred (list or np.ndarray): Predicted labels
        class_labels (list): Label names (e.g., ["Benign", "Malignant"])
        filename (str): Output filename for the saved figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return cm


def plot_roc_curve(y_true, y_probs, filename="roc_curve.pdf"):
    """
    Plots and saves the ROC curve with AUC score.

    Args:
        y_true (list or np.ndarray): True binary labels
        y_probs (list or np.ndarray): Predicted probabilities for the positive class
        filename (str): Output filename for the saved figure
    """
    auc_score = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return auc_score


def compute_classification_metrics(cm, y_true, y_probs, filename="final_test_metrics.csv"):
    """
    Computes classification metrics including accuracy, precision, recall, F1 score, and AUC.
    Saves results to a CSV file.

    Args:
        cm (np.ndarray): Confusion matrix
        y_true (list or np.ndarray): True labels
        y_probs (list or np.ndarray): Predicted probabilities for AUC
        filename (str): Output CSV filename
    """
    TP, TN = cm[1,1], cm[0,0]
    FP, FN = cm[0,1], cm[1,0]
    total = np.sum(cm)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = (TP + TN) / total
    auc_score = roc_auc_score(y_true, y_probs)

    metrics_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'AUC': [auc_score]
    })
    metrics_df.to_csv(filename, index=False)
    return metrics_df
