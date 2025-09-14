#!/usr/bin/env python

import json
import os
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationLoader:
    def __init__(self):
        self.results = {}

    def load_results(self, filepath):
            with open(filepath, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded evaluation results from {filepath}")
            
    def get_base_metrics(self):
        print(
        f"accuracy      - {self.results['accuracy']}\n"
        f"precision     - {self.results['precision']}\n"
        f"f1_score      - {self.results['f1_score']}\n"
        f"roc_auc_score - {self.results['roc_auc']}"
        )
        
    def plot_confusion_matrix(self, normalize=True, cmap="Blues"):
        cm = self.results["confusion_matrix"]
        if normalize:
            cm = [[val / sum(row) for val in row] for row in cm]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
        
    def plot_roc_curve(self):
        y_true = self.results["y_true"]
        y_prob = self.results["y_prob"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
        
    def plot_precision_recall_curve(self):
        y_true = self.results["y_true"]
        y_prob = self.results["y_prob"]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()
