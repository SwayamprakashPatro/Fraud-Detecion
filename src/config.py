import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ProjectConfig:
    # Base paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    GRAPH_PATH: str = os.path.join(BASE_DIR, "graph.gpickle")
    CSV_DATA_PATH: str = os.path.join(BASE_DIR, "Cleaned_Database.csv")
    
    # Output directories
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
    MODEL_DIR: str = os.path.join(OUTPUT_DIR, "models")
    PLOT_DIR: str = os.path.join(OUTPUT_DIR, "plots")
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")
    
    # File targets
    BEST_MODEL_PATH: str = os.path.join(MODEL_DIR, "best_graphsage.pt")
    HISTORY_PATH: str = os.path.join(LOG_DIR, "training_history.json")
    EVAL_REPORT_PATH: str = os.path.join(LOG_DIR, "evaluation_report.json")
    LOSS_PLOT_PATH: str = os.path.join(PLOT_DIR, "training_validation_loss.png")
    METRICS_PLOT_PATH: str = os.path.join(PLOT_DIR, "validation_metrics_curve.png")
    ROC_PLOT_PATH: str = os.path.join(PLOT_DIR, "roc_pr_curves.png")
    CONFUSION_MATRIX_PATH: str = os.path.join(PLOT_DIR, "confusion_matrix.png")
    
    # Model Architecture
    HIDDEN_CHANNELS: int = 48
    OUT_CHANNELS: int = 48
    NUM_LAYERS: int = 2
    AGGREGATOR: str = "mean"  # GraphSAGE aggregator: 'mean', 'max'
    DROPOUT: float = 0.20
    
    # Training Parameters
    EPOCHS: int = 60
    LEARNING_RATE: float = 0.003
    WEIGHT_DECAY: float = 1e-5
    PATIENCE: int = 15
    RANDOM_SEED: int = 42
    
    # Dataset Splits
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # Database config (for live DB re-sync)
    DB_HOST: str = "localhost"
    DB_USER: str = "root"
    DB_PASS: str = "1234"
    DB_NAME: str = "fraud_detection_gnn"
    
    def __post_init__(self):
        for directory in [self.OUTPUT_DIR, self.MODEL_DIR, self.PLOT_DIR, self.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)

config = ProjectConfig()
