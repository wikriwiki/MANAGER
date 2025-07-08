"""
models/graph_builder.py – utterance‑level 그래프 + multi‑utterance merge 지원
────────────────────────────────────────────────────────────────────
* build() : 단일 발화(텍스트·지식·비디오·오디오) → Data (768 dim 노드)
* merge_graph(prev, new) : 이전 발화 그래프와 연결하여 대화 그래프 확장
    · 노드 feature cat
    · edge_index / edge_type cat (new 쪽 인덱스 offset)
    · 추가 엣지  type="utt" (prev 마지막 텍스트 노드 → new 첫 텍스트 노드)
"""
from __future__ import annotations

import os, re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data

from encoder import TextFeatureExtractor, ExternalFinancialKnowledgeModel

EDGE_TYPE = {
    "t_t": 0,
    "v_v": 1,
    "a_a": 2,
    "t_v": 3,
    "t_a": 4,
}
class GraphBuilder:
    """단일 발화 그래프 + 여러 발화 병합 유틸"""

    def __init__(self, time_iso: str, *, merge_anchor: bool = False):
        self.time_iso = time_iso
        self.merge_anchor = merge_anchor
        self.text_enc = TextFeatureExtractor()
        self.ekm = ExternalFinancialKnowledgeModel()

    # ---------------- PID → label ----------------------------
    def _triple_to_string(self, subj: str, pid: str, obj: str) -> str:
        if not hasattr(self, "_pid_cache"):
            self._pid_cache: Dict[str, str] = {}
        if pid not in self._pid_cache:
            row = self.ekm.wiki.execute(
                "SELECT label FROM property_labels WHERE pid=? LIMIT 1", (pid,)
            ).fetchone()
            self._pid_cache[pid] = row[0] if row else pid
        return f"{subj} [REL] {self._pid_cache[pid].replace(' ', '_')} [REL] {obj}"

    # ---------------- 단일 발화 그래프 ----------------------
    def build(
        self,
        utterance_text: str,
        video_emb: torch.Tensor | None = None,  # [1,768]
        audio_emb: torch.Tensor | None = None,  # [1,768]
    ) -> Data:
        """발화(텍스트+모달) → PyG Data"""
        # 1) triple
        ent_ids, ek_sub = self.ekm.acquire_related_external_knowledge(utterance_text, self.time_iso)
        triples = []
        if ek_sub.num_edges:
            for s, d, (prop,) in zip(*ek_sub.edge_index, ek_sub.edge_attr):
                triples.append((self.ekm.int2qid[s], f"P{prop}", self.ekm.int2qid[d]))

        # 2) 텍스트 encode
        wp_emb, meta = self.text_enc.encode(
            utterance_text,
            knowledge_triples=[self._triple_to_string(*t) for t in triples],
            anchor_entities=[self.ekm.int2qid[i] for i in ent_ids],
        )
        hs = wp_emb; D = hs.size(1); sep0 = meta["sep"][0]

        node_feats, edge_src, edge_dst, edge_type = [], [], [], []

        # --- utterance text 토큰 노드 (t_t 엣지) ---
        text_token_map = {idx: len(node_feats) for idx in range(1, sep0)}
        node_feats.extend([hs[i] for i in range(1, sep0)])
        text_nodes = list(text_token_map.values())
        for i in range(len(text_nodes) - 1):
            edge_src.append(text_nodes[i]); edge_dst.append(text_nodes[i+1]); edge_type.append(EDGE_TYPE["t_t"])

        # --- knowledge 토큰도 Text 카테고리 ---
        knowledge_token_map = {}
        for s, e in meta["knowledge_blocks"]:
            prev = None
            for idx in range(s, e):
                g = len(node_feats); knowledge_token_map[idx] = g; node_feats.append(hs[idx])
                if prev is not None:
                    edge_src.append(prev); edge_dst.append(g); edge_type.append(EDGE_TYPE["t_t"])
                prev = g
                
        # 비디오 노드
        if video_emb is not None and video_emb.numel():
            v_idx=len(node_feats); node_feats.append(video_emb.squeeze(0))
            for t in text_nodes:
                edge_src.append(t); edge_dst.append(v_idx); edge_type.append(EDGE_TYPE["t_v"])
        # 오디오 노드
        if audio_emb is not None and audio_emb.numel():
            a_idx=len(node_feats); node_feats.append(audio_emb.squeeze(0))
            for t in text_nodes:
                edge_src.append(t); edge_dst.append(a_idx); edge_type.append(EDGE_TYPE["t_a"])

        x=torch.stack(node_feats) if node_feats else torch.empty(0,D)
        edge_index=torch.tensor([edge_src,edge_dst],dtype=torch.long) if edge_src else torch.empty(2,0,dtype=torch.long)
        edge_type_t=torch.tensor(edge_type,dtype=torch.long) if edge_type else torch.empty(0,dtype=torch.long)

        data=Data(x=x,edge_index=edge_index,edge_type=edge_type_t)
        data.node_meta={
            "text_tokens":len(text_nodes),
            "knowledge_tokens":len(knowledge_token_map),
            "video_nodes":1 if video_emb is not None else 0,
            "audio_nodes":1 if audio_emb is not None else 0,
            "triples":len(triples),
        }
        return data

