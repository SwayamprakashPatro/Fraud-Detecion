import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.config import config
from src.data_loader import FraudGraphDataset
from src.models.graphsage import FraudGraphSAGE
from src.utils.metrics import compute_classification_metrics, get_confusion_matrix_data

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained GraphSAGE Fraud Detector")
    parser.add_argument("--checkpoint", type=str, default=config.BEST_MODEL_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}. Please train the model first with train.py.")
        return

    print("Loading graph dataset and transaction features...")
    dataset = FraudGraphDataset()
    data, metadata = dataset.load_data()

    print(f"Loading optimal checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Auto-detect hidden dimension from checkpoint tensor shapes
    if "conv1.lin_l.bias" in state_dict:
        hidden_dim = state_dict["conv1.lin_l.bias"].shape[0]
    else:
        hidden_dim = config.HIDDEN_CHANNELS

    threshold = checkpoint.get("best_threshold", 0.5)

    model = FraudGraphSAGE(
        node_in_channels=metadata["node_feature_dim"],
        edge_in_channels=metadata["num_features"],
        hidden_channels=hidden_dim,
        out_channels=hidden_dim,
        num_layers=config.NUM_LAYERS,
        dropout=0.0
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        test_probs = model.predict_proba(
            data.x,
            data.edge_index,
            data.tx_src,
            data.tx_dst,
            data.tx_edge_attr,
            mask=data.test_mask
        ).numpy()
        test_y = data.y[data.test_mask].numpy()

    metrics = compute_classification_metrics(test_y, test_probs, threshold=threshold)
    cm, counts = get_confusion_matrix_data(test_y, test_probs, threshold=threshold)

    print("\n" + "=" * 60)
    print(f"  HOLDOUT TEST SET EVALUATION (Epoch {checkpoint.get('epoch', 'N/A')})")
    print("=" * 60)
    print(f"- Calibrated Threshold: {threshold:.2f}")
    for k, v in metrics.items():
        if k != "threshold":
            print(f"- {k.upper():<16}: {v:.4f}")
    print("-" * 60)
    print(f"Confusion Breakdown : TP={counts['TP']}, FP={counts['FP']}, TN={counts['TN']}, FN={counts['FN']}")
    print(f"Accuracy Rate       : {(counts['TP'] + counts['TN']) / len(test_y) * 100:.2f}%")
    print(f"Fraud Recall Rate   : {counts['TP'] / (counts['TP'] + counts['FN']) * 100:.2f}%")
    print(f"False Alarm Rate    : {counts['FP'] / (counts['FP'] + counts['TN']) * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
