from __future__ import annotations
"""
models/graph_builder.py – utterance-level 그래프 + multi-utterance merge 지원
────────────────────────────────────────────────────────────────────
* build() : 단일 발화(텍스트·지식·비디오·오디오) → Data (768-dim 노드)
* merge_graph(prev, new) : 이전 발화 그래프와 연결하여 대화 그래프 확장
    · 노드 feature cat
    · edge_index / edge_type cat (new 쪽 인덱스 offset)
    · 추가 엣지  type="utt" (prev 마지막 텍스트 노드 ↔ new 첫 텍스트 노드)  ← **양방향**

모든 새로 생성되는 엣지는 **양방향**(bidirectional)으로 추가된다.
"""

import os
import re
import gc                                 # === ADDED ===
import subprocess                         # === ADDED ===
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data

from models.encoder import TextFeatureExtractor, ExternalFinancialKnowledgeModel

# ---------------------------------------------------------------------------
# GPU VRAM 리포트 유틸  ────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# === ADDED : 어떤 모듈에서도 호출할 수 있게 간단 함수 구현 ===================
def _report_gpu_mem(tag: str = "GraphBuilder"):
    """GPU별 alloc / reserved / inactive / peak 요약 출력"""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    print(f"\n=== GPU VRAM report [{tag}] ===")
    for idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(idx)
        alloc    = torch.cuda.memory_allocated()    / 1024**2  # MB
        reserved = torch.cuda.memory_reserved()     / 1024**2
        inactive = reserved - alloc
        peak     = torch.cuda.max_memory_allocated() / 1024**2
        print(
            f"[GPU{idx}] alloc {alloc:7.1f} | reserved {reserved:7.1f} "
            f"| inactive {inactive:7.1f} | peak {peak:7.1f} MB"
        )
        torch.cuda.reset_peak_memory_stats(idx)

    # 드라이버 레벨 사용량도 함께 (nvidia-smi)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        print("[nvidia-smi]", " | ".join(
            f"GPU{n}:{row.strip()}" for n, row in enumerate(out.strip().splitlines())))
    except Exception:
        pass
# ============================================================================

# ---------------------------------------------------------------------------
# Edge type id 정의 (단방향/역방향 구분 없이 동일 id 사용)
# ---------------------------------------------------------------------------
EDGE_TYPE: Dict[str, int] = {
    "t_t": 0,  # text  ↔ text
    "v_v": 1,  # video ↔ video
    "a_a": 2,  # audio ↔ audio
    "t_v": 3,  # text  ↔ video
    "t_a": 4,  # text  ↔ audio
}

# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _add_bidir(
    src: int,
    dst: int,
    etype: int,
    edge_src: List[int],
    edge_dst: List[int],
    edge_type: List[int],
):
    """src↔dst 양방향을 edge 목록에 기록"""
    edge_src.extend([src, dst])
    edge_dst.extend([dst, src])
    edge_type.extend([etype, etype])


# ════════════════════════════════════════════════════════════════════════
class GraphBuilder:
    """단일 발화 그래프 + 여러 발화 병합 유틸 (***양방향 엣지 버전***)"""

    def __init__(self, time_iso: str, *, merge_anchor: bool = False):
        self.time_iso = time_iso
        self.merge_anchor = merge_anchor
        self.text_enc = TextFeatureExtractor()
        self.ekm = ExternalFinancialKnowledgeModel()

    def __del__(self):
        """GraphBuilder가 소멸될 때 TextFeatureExtractor 정리 & VRAM 리포트"""
        try:
            if hasattr(self, "text_enc"):
                self.text_enc.cleanup()
                del self.text_enc
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            print("GraphBuilder 메모리 해제 완료")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            _report_gpu_mem("after GraphBuilder cleanup") 

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
        """발화(텍스트+모달) → PyG Data (모든 엣지 양방향)"""
        if video_emb is not None and video_emb.is_cuda:
            video_emb = video_emb.cpu()
        if audio_emb is not None and audio_emb.is_cuda:
            audio_emb = audio_emb.cpu()

        # 1) 외부 지식 트리플 수집
        ent_ids, ek_sub = self.ekm.acquire_related_external_knowledge(
            utterance_text, self.time_iso
        )
        triples: List[Tuple[str, str, str]] = []
        if ek_sub.num_edges:
            for s, d, (prop,) in zip(*ek_sub.edge_index, ek_sub.edge_attr):
                triples.append(
                    (
                        self.ekm.int2qid[s],
                        f"P{prop}",
                        self.ekm.int2qid[d],
                    )
                )

        # 2) 텍스트 인코딩
        wp_emb, meta = self.text_enc.encode(
            utterance_text,
            knowledge_triples=[self._triple_to_string(*t) for t in triples],
            anchor_entities=[self.ekm.int2qid[i] for i in ent_ids],
        )
        if wp_emb.is_cuda:
            wp_emb = wp_emb.cpu()
        hs = wp_emb  # [N,768]
        D = hs.size(1)
        sep0 = meta["sep"][0]

        node_feats: List[torch.Tensor] = []
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_type: List[int] = []

        # --- utterance text 토큰 노드 (t_t 엣지, 양방향) ---
        text_token_map: Dict[int, int] = {
            idx: len(node_feats) for idx in range(1, sep0)
        }
        node_feats.extend([hs[i] for i in range(1, sep0)])
        text_nodes = list(text_token_map.values())
        for i in range(len(text_nodes) - 1):
            _add_bidir(
                text_nodes[i], text_nodes[i + 1], EDGE_TYPE["t_t"], edge_src, edge_dst, edge_type
            )

        # --- knowledge 토큰도 Text 카테고리 (t_t 양방향) ---
        knowledge_token_map: Dict[int, int] = {}
        for s, e in meta["knowledge_blocks"]:
            prev = None
            for idx in range(s, e):
                g = len(node_feats)
                knowledge_token_map[idx] = g
                node_feats.append(hs[idx])
                if prev is not None:
                    _add_bidir(prev, g, EDGE_TYPE["t_t"], edge_src, edge_dst, edge_type)
                prev = g

        # 비디오 노드 --------------------------------------------------
        if video_emb is not None and video_emb.numel():
            v_idx = len(node_feats)
            node_feats.append(video_emb.squeeze(0))
            for t in text_nodes:
                _add_bidir(t, v_idx, EDGE_TYPE["t_v"], edge_src, edge_dst, edge_type)
        else:
            v_idx = -1

        # 오디오 노드 --------------------------------------------------
        if audio_emb is not None and audio_emb.numel():
            a_idx = len(node_feats)
            node_feats.append(audio_emb.squeeze(0))
            for t in text_nodes:
                _add_bidir(t, a_idx, EDGE_TYPE["t_a"], edge_src, edge_dst, edge_type)
        else:
            a_idx = -1

        # -------------------- Tensor 변환 ------------------------------
        x = torch.stack(node_feats) if node_feats else torch.empty(0, D)
        edge_index = (
            torch.tensor([edge_src, edge_dst], dtype=torch.long)
            if edge_src
            else torch.empty(2, 0, dtype=torch.long)
        )
        edge_type_t = (
            torch.tensor(edge_type, dtype=torch.long)
            if edge_type
            else torch.empty(0, dtype=torch.long)
        )

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type_t)
        data.node_meta = {
            "text_tokens": len(text_nodes),
            "knowledge_tokens": len(knowledge_token_map),
            "video_nodes": 1 if v_idx != -1 else 0,
            "audio_nodes": 1 if a_idx != -1 else 0,
            "triples": len(triples),
        }
        return data
