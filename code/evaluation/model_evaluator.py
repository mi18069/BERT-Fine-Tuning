#!/usr/bin/env python


import torch
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.results = {}

    def evaluate(self, dataloader, num_labels=2, label_names=None):
        """
        Evaluates a PyTorch model on a given DataLoader.
        """
        self.model.eval()
        y_true, y_pred, all_scores = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                all_scores.extend(probs.cpu().tolist())

        self._compute_metrics(y_true, y_pred, all_scores, num_labels=num_labels, label_names=label_names)

    def _compute_metrics(self, y_true, y_pred, all_scores, num_labels=2, label_names=None):
        """
        Compute classification metrics.
        """
        accuracy_val = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=list(range(num_labels)))
        
        roc_auc = roc_auc_score(y_true, all_scores)
        pr_auc = average_precision_score(y_true, all_scores)

        self.results = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": all_scores,
            "accuracy": accuracy_val,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "report": classification_report(y_true, y_pred, target_names=label_names, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }
        
    def save_results(self, filepath="evaluation_metrics/results.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Saved evaluation results to {filepath}")
        
