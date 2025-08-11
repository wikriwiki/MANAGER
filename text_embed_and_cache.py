#!/usr/bin/env python3
"""
build_video_graphs.py
────────────────────────────────────────────────────────────────────────────
✨ 목적
  - 이 스크립트 하나로 동영상별 그래프를 구축하고 캐시에 저장합니다.
  - 별도의 모듈을 임포트하지 않고 모든 로직을 포함합니다.

실행 예
──────
# GPU를 사용하여 동영상별 그래프 구축
CUDA_VISIBLE_DEVICES=0 python build_video_graphs.py
"""

from __future__ import annotations
import os
import sqlite3
import torch
import torch_geometric
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# ────────── 경로 및 하드코딩 설정 ───────────────────────────────────────
DB_PATH = "data/speech_segments.db"
CACHE_DIR = Path("./cache_video_graphs")

# 오디오/비디오 캐시 경로 (실제 경로로 수정 필요)
AUDIO_CACHE_DIR = Path("./cache")
VIDEO_CACHE_DIR = Path("./cache")
# ───────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Edge type id 정의
# ---------------------------------------------------------------------------
EDGE_TYPE: Dict[str, int] = {
    "t_t": 0,  # text  ↔ text
    "v_v": 1,  # video ↔ video
    "a_a": 2,  # audio ↔ audio
    "t_v": 3,  # text  ↔ video
    "t_a": 4,  # text  ↔ audio

}

# ---------------------------------------------------------------------------
# 내부 헬퍼 함수
# ---------------------------------------------------------------------------

def _add_bidir(
    src: int,
    dst: int,
    etype: int,
    edge_src: List[int],
    edge_dst: List[int],
    edge_type: List[int],
):
    """src↔dst 양방향 엣지를 edge 목록에 기록"""
    edge_src.extend([src, dst])
    edge_dst.extend([dst, src])
    edge_type.extend([etype, etype])

def get_node_cache_path(seg_id: str, cache_type: str) -> Path | None:
    """세그먼트 ID에 해당하는 오디오/비디오 캐시 파일 경로 반환"""
    cache_dir = AUDIO_CACHE_DIR if cache_type == "audio" else VIDEO_CACHE_DIR
    cache_path = cache_dir / f"{seg_id}.pt"
    if cache_path.exists():
        return cache_path
    return None

def merge_graphs(prev_data: torch_geometric.data.Data, new_data: torch_geometric.data.Data) -> torch_geometric.data.Data:
    """두 그래프를 병합하고, 발화 간 연결 엣지를 추가합니다."""
    
    if prev_data is None:
        return new_data

    # 노드 피처 병합
    x_merged = torch.cat([prev_data.x, new_data.x], dim=0)

    # 엣지 인덱스 병합
    num_prev_nodes = prev_data.num_nodes
    edge_index_new_offset = new_data.edge_index + num_prev_nodes
    edge_index_merged = torch.cat([prev_data.edge_index, edge_index_new_offset], dim=1)

    # 엣지 타입 병합
    edge_type_merged = torch.cat([prev_data.edge_type, new_data.edge_type], dim=0)

    # 발화 연결 엣지 추가 (이전 그래프의 마지막 노드 ↔ 새 그래프의 첫 번째 노드)
    # prev_data의 마지막 노드
    prev_last_node_idx = num_prev_nodes - 1
    # new_data의 첫 번째 노드 (offset 필요 없음)
    new_first_node_idx = num_prev_nodes

    _add_bidir(
        prev_last_node_idx,
        new_first_node_idx,
        EDGE_TYPE["t_t"],
        edge_index_merged[0].tolist(),
        edge_index_merged[1].tolist(),
        edge_type_merged.tolist()
    )

    return torch_geometric.data.Data(
        x=x_merged,
        edge_index=torch.tensor([edge_index_merged[0].tolist(), edge_index_merged[1].tolist()]),
        edge_type=torch.tensor(edge_type_merged.tolist())
    )

# ---------------------------------------------------------------------------
# GraphBuilder 클래스 (단일 발화 그래프 생성 및 병합 로직)
# ---------------------------------------------------------------------------
class GraphBuilder:
    def __init__(self):
        # TextFeatureExtractor와 ExternalFinancialKnowledgeModel의 핵심 로직을 여기에 구현
        # 또는 간단한 더미 로직으로 대체
        self.text_enc = lambda text: (torch.randn(10, 768), {"sep": [5], "knowledge_blocks": []})
        self.ekm = lambda text: ([], None)
        
    def build(self, utterance_text: str, video_emb: torch.Tensor | None = None, audio_emb: torch.Tensor | None = None) -> torch_geometric.data.Data:
        # 1) 텍스트 인코딩 (더미 로직)
        wp_emb, meta = self.text_enc(utterance_text)
        hs = wp_emb
        D = hs.size(1)
        sep0 = meta["sep"][0]

        node_feats: List[torch.Tensor] = []
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_type: List[int] = []

        # --- utterance text 토큰 노드 ---
        text_token_map: Dict[int, int] = {
            idx: len(node_feats) for idx in range(1, sep0)
        }
        node_feats.extend([hs[i] for i in range(1, sep0)])
        text_nodes = list(text_token_map.values())
        for i in range(len(text_nodes) - 1):
            _add_bidir(
                text_nodes[i], text_nodes[i + 1], EDGE_TYPE["t_t"], edge_src, edge_dst, edge_type
            )

        # 비디오 노드
        if video_emb is not None and video_emb.numel():
            v_idx = len(node_feats)
            node_feats.append(video_emb.squeeze(0))
            for t in text_nodes:
                _add_bidir(t, v_idx, EDGE_TYPE["t_v"], edge_src, edge_dst, edge_type)

        # 오디오 노드
        if audio_emb is not None and audio_emb.numel():
            a_idx = len(node_feats)
            node_feats.append(audio_emb.squeeze(0))
            for t in text_nodes:
                _add_bidir(t, a_idx, EDGE_TYPE["t_a"], edge_src, edge_dst, edge_type)

        x = torch.stack(node_feats) if node_feats else torch.empty(0, D)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_type_t = torch.tensor(edge_type, dtype=torch.long)

        return torch_geometric.data.Data(x=x, edge_index=edge_index, edge_type=edge_type_t)

# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------
def main():
    CACHE_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT video_id, segment_id, script, utterance_hash "
        "FROM speech_segments ORDER BY video_id, start_time"
    )
    
    video_segments: Dict[str, List[Dict]] = {}
    for r in cur.fetchall():
        video_id = r["video_id"]
        if video_id not in video_segments:
            video_segments[video_id] = []
        video_segments[video_id].append(dict(r))
    conn.close()

    gb = GraphBuilder()
    
    pbar = tqdm(video_segments.items(), desc="Building video graphs")
    for video_id, segments in pbar:
        cache_path = CACHE_DIR / f"{video_id}.pt"
        if cache_path.exists():
            continue

        video_graph = None
        for segment in segments:
            seg_id = segment["segment_id"]
            script = segment["script"]
            
            # 오디오/비디오 캐시 파일 로드
            audio_path = get_node_cache_path(seg_id, "audio")
            video_path = get_node_cache_path(seg_id, "video")
            
            audio_emb = torch.load(audio_path) if audio_path else None
            video_emb = torch.load(video_path) if video_path else None
            
            # 세그먼트 그래프 생성
            seg_graph = gb.build(
                utterance_text=script,
                video_emb=video_emb,
                audio_emb=audio_emb
            )
            
            # 그래프 병합
            video_graph = merge_graphs(video_graph, seg_graph)
            
        if video_graph is not None:
            torch.save(video_graph, cache_path)
    
    print("✅ All video graphs built and cached.")

if __name__ == "__main__":
    main()