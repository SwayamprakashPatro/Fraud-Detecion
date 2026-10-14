import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import config
from src.models.graphsage import FraudGraphSAGE
from src.utils.metrics import (
    compute_classification_metrics,
    get_confusion_matrix_data,
    find_optimal_threshold,
)
from src.utils.visualizer import (
    plot_loss_curves,
    plot_metrics_curves,
    plot_roc_pr_curves,
    plot_confusion_matrix,
)

class FraudTrainer:
    """
    Orchestrates GraphSAGE training, live progress logging, early stopping,
    checkpointing, validation, test evaluation, and curve generation.
    """
    def __init__(
        self,
        model: FraudGraphSAGE,
        data: Any,
        metadata: Dict[str, Any],
        device: Optional[torch.device] = None,
        lr: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        epochs: int = config.EPOCHS,
        patience: int = config.PATIENCE
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data = data.to(self.device)
        self.metadata = metadata
        self.epochs = epochs
        self.patience = patience
        
        # Binary Cross Entropy with Logits Loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.6, patience=4
        )
        
        # History dictionary for tracking curves
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_roc_auc": [],
            "val_pr_auc": []
        }
        
    def train_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.tx_src,
            self.data.tx_dst,
            self.data.tx_edge_attr,
            mask=self.data.train_mask
        )
        
        train_targets = self.data.y[self.data.train_mask]
        loss = self.criterion(logits, train_targets)
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()
        
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, mask: torch.Tensor, threshold: float = 0.5) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluates model performance on the specified mask."""
        self.model.eval()
        logits = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.tx_src,
            self.data.tx_dst,
            self.data.tx_edge_attr,
            mask=mask
        )
        targets = self.data.y[mask]
        loss = float(self.criterion(logits, targets).item())
        
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = targets.cpu().numpy()
        
        metrics = compute_classification_metrics(y_true, probs, threshold=threshold)
        return loss, metrics, y_true, probs

    def fit(self) -> Dict[str, Any]:
        """
        Executes full training pipeline with live console reporting,
        tracking validation curves, early stopping, and checkpoint saving.
        """
        print("=" * 88)
        print("STARTING GRAPHSAGE FRAUD DETECTION PIPELINE TRAINING")
        print(f"- Device          : {self.device}")
        print(f"- Total Nodes     : {self.metadata['num_nodes']} (Accounts, Devices, Merchants, UPI)")
        print(f"- Total Edges     : {self.metadata['num_edges']} (Interactions)")
        print(f"- Transactions    : {self.metadata['num_transactions']} (Train: {self.metadata['train_count']}, Val: {self.metadata['val_count']}, Test: {self.metadata['test_count']})")
        print(f"- Max Epochs      : {self.epochs} | Early Stopping Patience: {self.patience}")
        print("=" * 88)
        print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Val Loss':^10} | {'Val ROC-AUC':^12} | {'Val F1':^9} | {'Val Recall':^11} | {'Status':^12}")
        print("-" * 88)
        
        best_val_auc = -1.0
        best_threshold = 0.5
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics, y_val, y_val_probs = self.evaluate(self.data.val_mask, threshold=0.5)
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_roc_auc"].append(val_metrics["roc_auc"])
            self.history["val_pr_auc"].append(val_metrics["pr_auc"])
            
            self.scheduler.step(val_metrics["roc_auc"])
            
            # Checkpoint best model based on ROC-AUC
            status = ""
            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_threshold = find_optimal_threshold(y_val, y_val_probs)
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "best_val_auc": best_val_auc,
                    "best_threshold": best_threshold
                }, config.BEST_MODEL_PATH)
                status = "[BEST MODEL]"
            else:
                patience_counter += 1
                status = f"Wait {patience_counter}/{self.patience}"
                
            # Live epoch output
            print(f"{epoch:^7} | {train_loss:^11.4f} | {val_loss:^10.4f} | {val_metrics['roc_auc']:^12.4f} | {val_metrics['f1']:^9.4f} | {val_metrics['recall']:^11.4f} | {status:<12}", flush=True)
            
            # Early stopping check
            if patience_counter >= self.patience:
                print("-" * 88)
                print(f"[STOP] Early stopping triggered at epoch {epoch}. Best Val ROC-AUC: {best_val_auc:.4f}")
                break
                
        total_time = time.time() - start_time
        print("=" * 88)
        print(f"[DONE] Training completed in {total_time:.2f} seconds.")
        
        # Load best checkpoint for final evaluation
        if os.path.exists(config.BEST_MODEL_PATH):
            checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            best_threshold = checkpoint.get("best_threshold", 0.5)
            print(f"Loaded optimal checkpoint from Epoch {checkpoint['epoch']} (Val AUC: {checkpoint['best_val_auc']:.4f}, Optimal Threshold: {best_threshold:.2f})")
            
        # Evaluate on Test Set with optimal threshold
        test_loss, test_metrics, y_test, y_test_probs = self.evaluate(self.data.test_mask, threshold=best_threshold)
        cm, cm_breakdown = get_confusion_matrix_data(y_test, y_test_probs, threshold=best_threshold)
        
        print("\n" + "=" * 55)
        print("  FINAL HOLDOUT TEST SET EVALUATION")
        print("=" * 55)
        print(f"- Decision Threshold: {best_threshold:.2f}")
        print(f"- Test Loss       : {test_loss:.4f}")
        print(f"- Test ROC-AUC    : {test_metrics['roc_auc']:.4f}")
        print(f"- Test PR-AUC     : {test_metrics['pr_auc']:.4f}")
        print(f"- Test F1-Score   : {test_metrics['f1']:.4f}")
        print(f"- Test Precision  : {test_metrics['precision']:.4f}")
        print(f"- Test Recall     : {test_metrics['recall']:.4f}")
        print(f"- Test Accuracy   : {test_metrics['accuracy']:.4f}")
        print(f"- Confusion Matrix: TP={cm_breakdown['TP']}, FP={cm_breakdown['FP']}, TN={cm_breakdown['TN']}, FN={cm_breakdown['FN']}")
        print("=" * 55)
        
        # Save training history and evaluation reports
        with open(config.HISTORY_PATH, "w") as f:
            json.dump(self.history, f, indent=4)
            
        report = {
            "test_metrics": test_metrics,
            "test_confusion_matrix": cm_breakdown,
            "best_validation_roc_auc": best_val_auc,
            "optimal_threshold": best_threshold,
            "total_epochs_trained": len(self.history["train_loss"]),
            "training_time_seconds": total_time
        }
        with open(config.EVAL_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
            
        # Generate & save publication-quality plots
        print("\n[*] Generating and saving performance curves...")
        plot_loss_curves(self.history, config.LOSS_PLOT_PATH)
        plot_metrics_curves(self.history, config.METRICS_PLOT_PATH)
        plot_roc_pr_curves(y_test, y_test_probs, config.ROC_PLOT_PATH)
        plot_confusion_matrix(cm, config.CONFUSION_MATRIX_PATH)
        
        print(f"[SAVED] Loss Curves          : {config.LOSS_PLOT_PATH}")
        print(f"[SAVED] Metrics Curves       : {config.METRICS_PLOT_PATH}")
        print(f"[SAVED] ROC & PR Curves      : {config.ROC_PLOT_PATH}")
        print(f"[SAVED] Confusion Matrix     : {config.CONFUSION_MATRIX_PATH}")
        print(f"[SAVED] Training Log         : {config.HISTORY_PATH}")
        print(f"[SAVED] Evaluation Report    : {config.EVAL_REPORT_PATH}")
        
        return report
