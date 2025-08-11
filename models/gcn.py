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
    num_rel      : edge_type 종류 (t_t, v_v, a_a, t_v, t_a, utt = 6). self-loop(-1)는 코드가 자동으로 처리합니다.
    num_layers   : GCN 레이어 깊이 (기본 2)
    dropout      : 드롭아웃 확률
    """
    def __init__(
        self,
        in_dim: int = 768,
        # 프로젝트에서는 6개의 관계(0~5)와 1개의 self-loop(-1)를 사용합니다.
        num_rel: int = 6, 
        num_layers: int = 5,
        dropout: float = 0,
    ):
        super().__init__()
        
        # [BUG FIX] self-loop의 edge_type이 -1이므로, 
        # 이를 처리하기 위해 관계의 수를 하나 늘려줍니다 (num_rel + 1).
        # 이렇게 하면 RGCNConv가 인덱스 [0, 6] (총 7개)을 처리할 수 있게 됩니다.
        true_num_relations = num_rel + 1
        
        self.layers = nn.ModuleList(
            RGCNConv(
                in_channels=in_dim,
                out_channels=in_dim,
                num_relations=true_num_relations,
            )
            for _ in range(num_layers)
        )
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """
        x          : [N, 768] node embeddings
        edge_index : [2, E]
        edge_type  : [E], 값의 범위는 [-1, 0, 1, 2, 3, 4, 5]
        returns    : [N, 768] updated node embeddings
        """
        h = x
        
        # [BUG FIX] edge_type에 1을 더하여 인덱스를 non-negative([0, 6])로 만듭니다.
        # 이 연산은 GPU 메모리 상에서 실시간으로 일어나며, 원본 캐시 파일은 변경되지 않습니다.
        safe_edge_type = edge_type + 1
        
        for conv in self.layers:
            # 수정된 safe_edge_type을 conv 레이어에 전달합니다.
            h = conv(h, edge_index, safe_edge_type)
            h = self.act(h)
            h = self.drop(h)
        return h