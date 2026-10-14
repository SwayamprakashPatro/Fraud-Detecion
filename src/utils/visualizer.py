import os
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_loss_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "GraphSAGE Training & Validation Loss"
):
    """Plots training and validation loss curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(epochs, history["train_loss"], label="Train Loss (BCE)", color="#2563EB", lw=2.2)
    plt.plot(epochs, history["val_loss"], label="Val Loss (BCE)", color="#DC2626", lw=2.2, linestyle="--")
    
    # Highlight best epoch
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = history["val_loss"][best_epoch - 1]
    plt.scatter([best_epoch], [best_val_loss], color="#059669", s=90, zorder=5, label=f"Best Val Epoch ({best_epoch})")
    
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Validation Performance Curves (ROC-AUC & F1-Score)"
):
    """Plots validation ROC-AUC and F1 scores across epochs."""
    epochs = range(1, len(history["val_roc_auc"]) + 1)
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(epochs, history["val_roc_auc"], label="Val ROC-AUC", color="#7C3AED", lw=2.2)
    plt.plot(epochs, history["val_f1"], label="Val F1-Score", color="#059669", lw=2.2)
    plt.plot(epochs, history["val_recall"], label="Val Recall", color="#F59E0B", lw=1.8, linestyle=":")
    plt.plot(epochs, history["val_precision"], label="Val Precision", color="#0284C7", lw=1.8, linestyle="-.")
    
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=10, loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str
):
    """Plots both ROC and Precision-Recall curves side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # ROC Curve
    axes[0].plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"GraphSAGE (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.5, label="Random Chance")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")
    
    # PR Curve
    axes[1].plot(recall, precision, color="#059669", lw=2.5, label=f"GraphSAGE (PR-AUC = {pr_auc:.4f})")
    no_skill = np.sum(y_true) / len(y_true)
    axes[1].plot([0, 1], [no_skill, no_skill], color="#94A3B8", linestyle="--", lw=1.5, label=f"Baseline ({no_skill:.2f})")
    axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: List[str] = ["Legitimate", "Fraud"]
):
    """Plots normalized & raw count Confusion Matrix heatmap."""
    plt.figure(figsize=(6.5, 5.5), dpi=300)
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    total = np.sum(cm)
    for i in range(nrows):
        for j in range(ncols):
            count = cm[i, j]
            pct = count / total * 100
            annot[i, j] = f"{count}\n({pct:.1f}%)"
            
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=1.0,
        linecolor="#E2E8F0"
    )
    plt.title("Confusion Matrix (Holdout Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
