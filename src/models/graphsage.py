import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional

class FraudGraphSAGE(nn.Module):
    """
    Enhanced Inductive GraphSAGE architecture for Financial Fraud Detection.
    
    Components:
    1. GraphSAGE Message Passing: Aggregates multi-hop structural context across
       accounts, shared devices, merchants, and UPI accounts.
    2. Batch-Normalized Edge Feature Encoder: Projects 138-dimensional transactional &
       behavioral attributes with proper feature scaling.
    3. Interaction Fusion: Combines sender node representation, receiver node representation,
       Hadamard interaction term, and transactional attributes.
    4. Deep Fraud Scoring Head: Outputs calibrated logits for fraud prediction.
    """
    def __init__(
        self,
        node_in_channels: int = 10,
        edge_in_channels: int = 138,
        hidden_channels: int = 48,
        out_channels: int = 48,
        num_layers: int = 2,
        dropout: float = 0.20,
        aggregator: str = "mean"
    ):
        super(FraudGraphSAGE, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 1. GraphSAGE Convolutional Layers
        self.conv1 = SAGEConv(node_in_channels, hidden_channels, aggr=aggregator)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggregator)
        
        # 2. Transaction Edge Feature Encoder with BatchNorm
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_channels, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU()
        )
        
        # 3. Fraud Classifier Head
        # Concatenates: [src_emb, dst_emb, hadamard_inter, edge_emb]
        # Dimensions: out_channels * 3 + hidden_channels
        fusion_dim = (out_channels * 3) + hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Propagates and samples neighborhood representations across graph."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tx_src: torch.Tensor,
        tx_dst: torch.Tensor,
        tx_edge_attr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass computing fraud risk logits.
        """
        # Node representations via GraphSAGE message passing
        node_emb = self.encode_nodes(x, edge_index)
        
        # Apply selection mask if provided
        if mask is not None:
            curr_src = tx_src[mask]
            curr_dst = tx_dst[mask]
            curr_edge_attr = tx_edge_attr[mask]
        else:
            curr_src = tx_src
            curr_dst = tx_dst
            curr_edge_attr = tx_edge_attr
            
        # Extract sender and receiver embeddings
        src_emb = node_emb[curr_src]
        dst_emb = node_emb[curr_dst]
        hadamard_inter = src_emb * dst_emb
        
        # Encode edge features
        edge_emb = self.edge_encoder(curr_edge_attr)
        
        # Multi-modal fusion
        fused = torch.cat([src_emb, dst_emb, hadamard_inter, edge_emb], dim=-1)
        
        # Output raw logits
        logits = self.classifier(fused).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tx_src: torch.Tensor,
        tx_dst: torch.Tensor,
        tx_edge_attr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes predicted fraud probabilities in [0, 1]."""
        logits = self.forward(x, edge_index, tx_src, tx_dst, tx_edge_attr, mask=mask)
        return torch.sigmoid(logits)
