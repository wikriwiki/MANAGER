# models/gcn.py
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class RGCNEncoder(nn.Module):
    """
    Relational GCN  ──>  node-level 768-d representation
    ─────────────────────────────────────────────────────
    Args
    ----
    in_dim       : 입력/출력 차원 (논문 768)
    num_rel      : edge_type 종류 (t_t, v_v, a_a, t_v, t_a = 5)
    num_layers   : GCN 레이어 깊이 (기본 2)
    dropout      : 드롭아웃 확률
    """
    def __init__(
        self,
        in_dim: int = 768,
        num_rel: int = 5,
        num_layers: int = 5,
        dropout: float = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            RGCNConv(
                in_channels=in_dim,
                out_channels=in_dim,
                num_relations=num_rel,
            )
            for _ in range(num_layers)
        )
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """
        x          : [N, 768] node embeddings
        edge_index : [2, E]
        edge_type  : [E]
        returns    : [N, 768] updated node embeddings
        """
        h = x
        for conv in self.layers:
            h = conv(h, edge_index, edge_type)
            h = self.act(h)
            h = self.drop(h)
        return h
