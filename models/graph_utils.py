from __future__ import annotations
"""
models/graph_utils.py – bidirectional merge utilities
─────────────────────────────────────────────────────
* _shift_graph() : 서브그래프 노드 인덱스 offset 적용 (shallow copy)
* merge_graph(prev, new) : 두 발화 그래프를 **양방향** 시간 엣지로 연결
    · Text ↔ Text, Video ↔ Video, Audio ↔ Audio 모두 src↔dst 쌍 추가
"""

from typing import Tuple, List
import torch
from torch_geometric.data import Data

from models.graph_builder import EDGE_TYPE  # {'t_t','v_v','a_a','t_v','t_a'}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _shift_graph(g: Data, offset: int) -> Data:
    """edge_index 및 모든 노드 번호를 +offset 하는 shallow copy"""
    return Data(
        x=g.x,
        edge_index=g.edge_index + offset,
        edge_type=g.edge_type,
        node_meta=dict(g.node_meta),
    )


def _indices(g: Data, offset: int = 0) -> Tuple[int, int, int, int]:
    """(first_text, last_text, video_idx, audio_idx) 반환 (글로벌 노드 번호)"""
    n_text = g.node_meta["text_tokens"]
    n_know = g.node_meta["knowledge_tokens"]

    first_t = offset
    last_t = offset + n_text - 1

    has_video = g.node_meta.get("video_nodes", 0)
    has_audio = g.node_meta.get("audio_nodes", 0)

    video_idx = offset + n_text + n_know if has_video else -1
    audio_idx = offset + n_text + n_know + has_video if has_audio else -1

    return first_t, last_t, video_idx, audio_idx


def _add_bidir(ei: List[List[int]], ety: List[int], src: int, dst: int, etype: int):
    """src↔dst 쌍을 edge_index/edge_type 리스트에 추가"""
    ei[0].extend([src, dst])
    ei[1].extend([dst, src])
    ety.extend([etype, etype])


# ---------------------------------------------------------------------------
# public merge util
# ---------------------------------------------------------------------------

def merge_graph(prev: Data, new: Data) -> Data:
    """두 발화 그래프를 **양방향** 모달(time) 엣지로 연결 후 하나로 합친다."""
    offset = prev.num_nodes
    new_s = _shift_graph(new, offset)

    # 1) 노드 concat
    x = torch.cat([prev.x, new_s.x], dim=0)

    # 2) 엣지 concat (리스트 → 나중에 tensor)
    ei0: List[int] = prev.edge_index[0].tolist() + new_s.edge_index[0].tolist()
    ei1: List[int] = prev.edge_index[1].tolist() + new_s.edge_index[1].tolist()
    ety: List[int] = prev.edge_type.tolist() + new_s.edge_type.tolist()

    # 3) 시간 엣지 (양방향)
    first_t_prev, last_t_prev, v_prev, a_prev = _indices(prev, 0)
    first_t_new, last_t_new, v_new, a_new = _indices(new_s, offset)

    # Text ↔ Text
    _add_bidir([ei0, ei1], ety, last_t_prev, first_t_new, EDGE_TYPE["t_t"])

    # Video ↔ Video (존재할 때만)
    if v_prev != -1 and v_new != -1:
        _add_bidir([ei0, ei1], ety, v_prev, v_new, EDGE_TYPE["v_v"])

    # Audio ↔ Audio (존재할 때만)
    if a_prev != -1 and a_new != -1:
        _add_bidir([ei0, ei1], ety, a_prev, a_new, EDGE_TYPE["a_a"])

    # 4) node_meta 합산 (단순 카운트 합)
    meta = {
        k: prev.node_meta.get(k, 0) + new.node_meta.get(k, 0)
        for k in set(prev.node_meta) | set(new.node_meta)
    }

    # tensor 변환
    edge_index = torch.tensor([ei0, ei1], dtype=torch.long)
    edge_type = torch.tensor(ety, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type, node_meta=meta)
