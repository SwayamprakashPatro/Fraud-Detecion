import sys
import os
# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import random
import numpy as np
import torch
import warnings

# Suppress minor future warnings for clean terminal logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import config
from src.data_loader import FraudGraphDataset
from src.models.graphsage import FraudGraphSAGE
from src.trainer import FraudTrainer

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE Financial Fraud Detection Pipeline")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=config.HIDDEN_CHANNELS, help="Hidden dimension size")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=config.DROPOUT, help="Dropout probability")
    parser.add_argument("--aggregator", type=str, default=config.AGGREGATOR, choices=["mean", "max"], help="GraphSAGE neighbor aggregator")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    print("\n" + "=" * 88)
    print("  FINANCIAL FRAUD DETECTION - GRAPHSAGE MACHINE LEARNING PIPELINE")
    print("=" * 88)
    print(f"Loading graph topology from: {config.GRAPH_PATH}")
    print(f"Loading transaction features from: {config.CSV_DATA_PATH}\n")

    # 1. Load and process dataset
    dataset = FraudGraphDataset(seed=args.seed)
    data, metadata = dataset.load_data()
    print("[OK] Graph and transaction features prepared successfully.")

    # 2. Build GraphSAGE model
    model = FraudGraphSAGE(
        node_in_channels=metadata["node_feature_dim"],
        edge_in_channels=metadata["num_features"],
        hidden_channels=args.hidden,
        out_channels=args.hidden,
        num_layers=config.NUM_LAYERS,
        dropout=args.dropout,
        aggregator=args.aggregator
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model Architecture: GraphSAGE ({config.NUM_LAYERS} layers, aggr='{args.aggregator}', {total_params:,} trainable parameters)\n")

    # 3. Train model with live output
    trainer = FraudTrainer(
        model=model,
        data=data,
        metadata=metadata,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience
    )

    report = trainer.fit()
    print("\n[COMPLETE] Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
