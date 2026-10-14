import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

from src.config import config

class FraudGraphDataset:
    """
    Constructs and prepares graph data for GraphSAGE fraud detection.
    
    Loads:
      - Heterogeneous interaction network (Accounts, Devices, Merchants, UPI)
      - Enriched transaction feature matrix (Cleaned_Database.csv)
      
    Outputs:
      - PyG Data object containing:
          - x: Node feature matrix [num_nodes, node_feat_dim]
          - edge_index: Interaction graph topology [2, num_edges_bidirectional]
          - tx_src: Source account indices for 9,000 transactions [num_tx]
          - tx_dst: Destination account indices for 9,000 transactions [num_tx]
          - tx_edge_attr: 138 transaction feature vectors [num_tx, 138]
          - y: Binary fraud labels [num_tx]
          - train_mask, val_mask, test_mask: Stratified split masks [num_tx]
    """
    def __init__(
        self,
        graph_path: str = config.GRAPH_PATH,
        csv_path: str = config.CSV_DATA_PATH,
        seed: int = config.RANDOM_SEED
    ):
        self.graph_path = graph_path
        self.csv_path = csv_path
        self.seed = seed
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        self.feature_columns = []
        
    def load_data(self) -> Tuple[Data, Dict[str, Any]]:
        # 1. Load NetworkX graph
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Graph file not found at {self.graph_path}. Run build_graph.py first.")
            
        with open(self.graph_path, "rb") as f:
            G = pickle.load(f)
            
        # 2. Load Cleaned Transaction Dataset
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Cleaned database not found at {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        # Sort by Transaction_ID to ensure strict indexing
        df = df.sort_values("Transaction_ID").reset_index(drop=True)
        
        # 3. Create canonical node index mapping
        nodes = sorted(list(G.nodes()))
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        num_nodes = len(nodes)
        
        # 4. Construct Node Features
        # Node Type: account, device, merchant, upi (4 dims)
        # Degrees: in-degree, out-degree, total degree (3 dims)
        # Aggregate amounts: mean_amount, sum_amount (2 dims)
        node_features = np.zeros((num_nodes, 10), dtype=np.float32)
        type_mapping = {"account": 0, "device": 1, "merchant": 2, "upi": 3}
        
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        # Compute financial volume aggregates per node
        node_amounts = {node: [] for node in nodes}
        for u, v, k, data in G.edges(keys=True, data=True):
            amt = data.get("amount", 0.0)
            if amt is not None and isinstance(amt, (int, float)):
                if u in node_amounts:
                    node_amounts[u].append(amt)
                if v in node_amounts:
                    node_amounts[v].append(amt)
                    
        for i, node in enumerate(nodes):
            node_type = G.nodes[node].get("type", "account")
            type_idx = type_mapping.get(node_type, 0)
            node_features[i, type_idx] = 1.0  # One-hot node type
            
            # Topological degrees (normalized with log1p)
            node_features[i, 4] = np.log1p(in_degrees.get(node, 0))
            node_features[i, 5] = np.log1p(out_degrees.get(node, 0))
            node_features[i, 6] = np.log1p(in_degrees.get(node, 0) + out_degrees.get(node, 0))
            
            # Volume stats
            amts = node_amounts.get(node, [])
            if len(amts) > 0:
                node_features[i, 7] = np.log1p(len(amts))
                node_features[i, 8] = np.log1p(max(0.0, float(np.mean(amts))))
                node_features[i, 9] = np.log1p(max(0.0, float(np.sum(amts))))
                
        # 5. Extract message-passing interaction graph (bidirectional)
        edges_src = []
        edges_dst = []
        for u, v in G.edges():
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            edges_src.extend([u_idx, v_idx])
            edges_dst.extend([v_idx, u_idx])
            
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        
        # 6. Extract Transaction Edge Instances & Rich Features
        tx_dict = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            if data.get("type") == "transaction":
                tx_id = int(k)
                tx_dict[tx_id] = (self.node_to_idx[u], self.node_to_idx[v])
                
        # Align with dataframe
        tx_src_list = []
        tx_dst_list = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            tx_id = int(row["Transaction_ID"])
            if tx_id in tx_dict:
                src, dst = tx_dict[tx_id]
                tx_src_list.append(src)
                tx_dst_list.append(dst)
                valid_indices.append(idx)
                
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        
        # Extract edge features
        self.feature_columns = [c for c in df_valid.columns if c not in ["Transaction_ID", "Fraud_Type", "Label"]]
        edge_attr_np = df_valid[self.feature_columns].values.astype(np.float32)
        labels_np = df_valid["Label"].values.astype(np.float32)
        
        # 7. Stratified Train / Validation / Test Split
        indices = np.arange(len(labels_np))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(config.VAL_RATIO + config.TEST_RATIO),
            random_state=self.seed,
            stratify=labels_np
        )
        
        val_share_of_temp = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_share_of_temp),
            random_state=self.seed,
            stratify=labels_np[temp_idx]
        )
        
        train_mask = torch.zeros(len(labels_np), dtype=torch.bool)
        val_mask = torch.zeros(len(labels_np), dtype=torch.bool)
        test_mask = torch.zeros(len(labels_np), dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # Pack into PyG Data
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            tx_src=torch.tensor(tx_src_list, dtype=torch.long),
            tx_dst=torch.tensor(tx_dst_list, dtype=torch.long),
            tx_edge_attr=torch.tensor(edge_attr_np, dtype=torch.float32),
            y=torch.tensor(labels_np, dtype=torch.float32),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        metadata = {
            "num_nodes": num_nodes,
            "num_edges": edge_index.size(1),
            "num_transactions": len(labels_np),
            "num_features": len(self.feature_columns),
            "node_feature_dim": node_features.shape[1],
            "train_count": int(train_mask.sum()),
            "val_count": int(val_mask.sum()),
            "test_count": int(test_mask.sum()),
            "fraud_ratio_train": float(labels_np[train_mask].mean()),
            "fraud_ratio_val": float(labels_np[val_mask].mean()),
            "fraud_ratio_test": float(labels_np[test_mask].mean()),
            "feature_names": self.feature_columns
        }
        
        return data, metadata
