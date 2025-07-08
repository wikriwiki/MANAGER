# graph_utils.py
import torch
from torch_geometric.data import Data
from typing import Tuple
from models.graph_builder import EDGE_TYPE   # {'t_t', 'v_v', 'a_a', 't_v', 't_a'}

# ──────────────────────────────────────────────────────────
def _shift_graph(g: Data, offset: int) -> Data:
    """edge_index 를 +offset 한 shallow copy"""
    return Data(
        x          = g.x,
        edge_index = g.edge_index + offset,
        edge_type  = g.edge_type,
        node_meta  = dict(g.node_meta),      # copy 해 두는 편이 안전
    )

def _indices(g: Data, offset: int = 0) -> Tuple[int, int, int, int]:
    """
    (first_text, last_text, video_idx, audio_idx) 를 반환.
    노드 번호는 전체 그래프 기준(offset 적용).
    """
    n_text = g.node_meta["text_tokens"]
    n_know = g.node_meta["knowledge_tokens"]

    first_t = offset
    last_t  = offset + n_text - 1

    has_video = g.node_meta.get("video_nodes", 0)
    has_audio = g.node_meta.get("audio_nodes", 0)

    video_idx = offset + n_text + n_know if has_video else -1
    audio_idx = offset + n_text + n_know + has_video if has_audio else -1

    return first_t, last_t, video_idx, audio_idx

# ──────────────────────────────────────────────────────────
def merge_graph(prev: Data, new: Data) -> Data:
    """두 발화 그래프를 동일 모달(time) 엣지로만 연결한다."""
    offset = prev.num_nodes
    new_s  = _shift_graph(new, offset)

    # 1) 노드 concat
    x = torch.cat([prev.x, new_s.x], dim=0)

    # 2) 엣지 concat
    ei  = torch.cat([prev.edge_index, new_s.edge_index], dim=1)
    ety = torch.cat([prev.edge_type,  new_s.edge_type],  dim=0)

    # 3) 동일 모달 시간 엣지
    _, last_t_prev, v_prev, a_prev = _indices(prev, 0)
    first_t_new, _, v_new, a_new   = _indices(new_s, offset)

    # Text → Text
    ei  = torch.cat([ei,  torch.tensor([[last_t_prev], [first_t_new]], dtype=torch.long)], dim=1)
    ety = torch.cat([ety, torch.tensor([EDGE_TYPE["t_t"]], dtype=torch.long)])

    # Video → Video (존재할 때만)
    if v_prev != -1 and v_new != -1:
        ei  = torch.cat([ei, torch.tensor([[v_prev], [v_new]], dtype=torch.long)], dim=1)
        ety = torch.cat([ety, torch.tensor([EDGE_TYPE["v_v"]], dtype=torch.long)])

    # Audio → Audio (존재할 때만)
    if a_prev != -1 and a_new != -1:
        ei  = torch.cat([ei, torch.tensor([[a_prev], [a_new]], dtype=torch.long)], dim=1)
        ety = torch.cat([ety, torch.tensor([EDGE_TYPE["a_a"]], dtype=torch.long)])

    # 4) node_meta 합산
    meta = {
        k: prev.node_meta.get(k, 0) + new.node_meta.get(k, 0)
        for k in set(prev.node_meta) | set(new.node_meta)
    }

    return Data(x=x, edge_index=ei, edge_type=ety, node_meta=meta)
