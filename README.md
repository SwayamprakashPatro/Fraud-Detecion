# Financial Fraud Detection Pipeline using GraphSAGE (Graph Machine Learning)

An end-to-end Graph Machine Learning pipeline for financial transaction fraud detection leveraging **GraphSAGE** (Graph Sample and Aggregate) and deep multi-modal feature fusion.

---

## 📁 Project Architecture

```text
GML project/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Central configurations & hyperparameters
│   ├── data_loader.py            # Heterogeneous network & transaction dataset loader
│   ├── models/
│   │   ├── __init__.py
│   │   └── graphsage.py          # Inductive GraphSAGE with Edge Fraud Classifier
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py            # ROC-AUC, PR-AUC, F1, and threshold calibration
│   │   └── visualizer.py         # Loss curves, metrics curves, ROC/PR & confusion matrix plots
│   └── trainer.py                # Training engine with live epoch tracking & early stopping
├── outputs/
│   ├── models/
│   │   └── best_graphsage.pt     # Optimal checkpoint state dict
│   ├── logs/
│   │   ├── training_history.json # Epoch-by-epoch loss & metric curves
│   │   └── evaluation_report.json# Holdout test set metrics & breakdown
│   └── plots/
│       ├── training_validation_loss.png
│       ├── validation_metrics_curve.png
│       ├── roc_pr_curves.png
│       └── confusion_matrix.png
├── build_graph.py                # Graph builder querying MySQL database into NetworkX graph
├── train.py                      # Main entrypoint script to launch live training
├── evaluate.py                   # Standalone evaluation script for test inference
├── Cleaned_Database.csv          # Preprocessed financial transactions & behavioral features
├── graph.gpickle                 # Serialized interaction network (1,000 nodes, 25,394 edges)
└── requirements.txt              # Pinned environment dependencies
```

---

## ⚡ Quick Start

### 1. Run Live Training
To train the GraphSAGE model with real-time console feedback, loss tracking, and automatic curve generation:

```bash
python train.py --epochs 60 --hidden 48 --lr 0.003
```

### 2. Standalone Model Evaluation
To test the saved checkpoint on holdout transactions:

```bash
python evaluate.py
```

---

## 📊 Live Tracking & Metrics

During training, each epoch displays:
- **Train Loss (BCE)**
- **Validation Loss (BCE)**
- **Validation ROC-AUC**
- **Validation F1-Score**
- **Validation Recall**
- **Early Stopping & Checkpoint Status**

All results and curves are automatically generated and saved into `outputs/plots/`.

---

## 🖥️ Web UI and API

The model has an HTTP API (`api/`) and a web front end (`frontend/`). The API runs this repo's own code (`src/`) and the saved checkpoint, and the front end draws what the API returns. Nothing on the page is typed in.

```bash
pip install -r requirements.txt
python run_app.py --open        # then visit http://127.0.0.1:8000
```

| Part | What it does |
|---|---|
| `api/service.py` | Loads `FraudGraphDataset` and `FraudGraphSAGE`, scores every payment, computes facts and stress tests, checks the result against `evaluation_report.json` |
| `api/main.py` | FastAPI app: routes below, also serves `frontend/` |
| `frontend/` | `index.html` (markup), `css/styles.css` (style), `js/app.js` (logic), `js/data.js` (offline snapshot), `js/vendor/d3.min.js` |
| `tools/export_snapshot.py` | Rewrites `frontend/js/data.js` from the API payload, for opening the page without a server |

| Route | Purpose |
|---|---|
| `GET /api/health` | Model loaded, checkpoint epoch, whether it reproduces the saved report |
| `GET /api/data` | Scores for all payments, training history, report, facts, stress tests |
| `GET /api/payments/{id}` | One payment: score, verdict, split, most unusual inputs |
| `POST /api/score` | Score a payment between two existing nodes with feature values you supply |
| `GET /api/metrics?threshold=0.4` | Held-out test metrics at any threshold |
| `POST /api/reload` | Re-read the checkpoint and logs after retraining (the page has a button for it) |
| `GET /docs` | Interactive API documentation |

After retraining (`python train.py`), click **Reload from model** on the page, or restart the server.
