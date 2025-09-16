"""
Graph builder with external-knowledge bridging (text<->relation<->tail)
- Keeps REAL PID strings via pid_vocab
- Concatenate utterance + knowledge text for a single-pass encoding
- 16-bit model load (no 8-bit quantization)
- Edge types include t_k (text↔knowledge) and k_k (knowledge↔knowledge)
- Utilities for visualization and summaries
"""
import os
import re
import sys
import json
import sqlite3
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import networkx as nx
import matplotlib.pyplot as plt

from transformers import (
    BertTokenizerFast,
    AutoTokenizer,
    AutoModelForCausalLM,
    # BitsAndBytesConfig,  # not used (we load at 16-bit now)
)

# ─────────────────────────────────────────────
# 환경
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Graph builder] Using device: {device}")

def _hash(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest() + ".pt"

def _to_device(data: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) for k, v in data.items()}

def log_vram(stage: str, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        dev_id = device.index if isinstance(device, torch.device) else torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev_id) / 1024**2
        reserved = torch.cuda.memory_reserved(dev_id) / 1024**2
        peak_a = torch.cuda.max_memory_allocated(dev_id) / 1024**2
        peak_r = torch.cuda.max_memory_reserved(dev_id) / 1024**2
        wasted = reserved - alloc
        print(f"[{stage:15s}] alloc: {alloc:6.1f} MB | reserved: {reserved:6.1f} MB | "
              f"peak_alloc: {peak_a:6.1f} MB | peak_reserved: {peak_r:6.1f} MB | wasted: {wasted:6.1f} MB")
        torch.cuda.reset_peak_memory_stats(dev_id)

# ─────────────────────────────────────────────
# 엣지 타입 (★ t_k, k_k 포함)
# ─────────────────────────────────────────────
EDGE_TYPE: Dict[str, int] = {
    "t_t": 0,  # text  ↔ text
    "v_v": 1,  # video ↔ video
    "a_a": 2,  # audio ↔ audio
    "t_v": 3,  # text  ↔ video
    "t_a": 4,  # text  ↔ audio
    "utt": 5,  # utterance ↔ utterance (for merging)
    "t_k": 6,  # text(언급 토큰) ↔ knowledge(관계)
    "k_k": 7,  # knowledge(관계) ↔ knowledge(테일 엔티티)
}

def _add_bidir(
    src: int,
    dst: int,
    etype: int,
    edge_src: List[int],
    edge_dst: List[int],
    edge_type: List[int],
):
    edge_src.extend([src, dst])
    edge_dst.extend([dst, src])
    edge_type.extend([etype, etype])

# ─────────────────────────────────────────────
# 외부 지식 모델 (라벨/특성 조회)
# ─────────────────────────────────────────────
class ExternalFinancialKnowledgeModel:
    def __init__(self,
                 wiki_db="data/wikidata_revisions.db",
                 speech_db="data/speech_segments.db"):
        self.wiki = sqlite3.connect(wiki_db); self.wiki.row_factory = sqlite3.Row
        self.speech = sqlite3.connect(speech_db); self.speech.row_factory = sqlite3.Row

        self.target_entities = self._load_target_entities()

        label_df = pd.read_sql(
            "SELECT DISTINCT qid, value AS label FROM labels WHERE lang='en';",
            self.wiki,
        )
        label_df = label_df[label_df["label"].str.lower().isin([n.lower() for n in self.target_entities])]
        label_df["id_int"] = pd.factorize(label_df["qid"])[0]

        self.qid2int = dict(zip(label_df["qid"], label_df["id_int"]))
        self.int2qid = {v: k for k, v in self.qid2int.items()}
        self.label2qid = {l.lower(): q for q, l in zip(label_df["qid"], label_df["label"])}

        def _name_to_pattern(name: str) -> str:
            name = name.strip().lower()
            if not name:
                return ""
            parts = re.split(r"\s+", name)
            parts = [re.escape(p) for p in parts if p]
            if not parts:
                return ""
            return r"\b" + r"\s+".join(parts) + r"\b"

        safe = [_name_to_pattern(n) for n in self.target_entities]
        safe = [p for p in safe if p]
        safe.sort(key=len, reverse=True)
        self._pat = re.compile("|".join(safe), flags=re.I)

    def _load_target_entities(self) -> List[str]:
        df = pd.read_sql("SELECT persons_found FROM video_metadata;", self.speech)
        names = set()
        for js in df["persons_found"]:
            if js:
                names.update(json.loads(js).keys())
        return list(names)

    def identify_entities(self, text: str) -> List[str]:
        t = re.sub(r"[^\w\s]", "", text.lower())
        return list({m.strip() for m in self._pat.findall(t)})

    def entities_to_id(self, ents: List[str]) -> List[int]:
        return [self.qid2int[self.label2qid[e.lower()]]
                for e in ents if e.lower() in self.label2qid]

    def qid_to_label(self, qid: str) -> str:
        row = self.wiki.execute(
            "SELECT value FROM labels WHERE qid=? AND lang='en' LIMIT 1",
            (qid,),
        ).fetchone()
        return row[0] if row else qid

    def pid_to_label(self, pid: str) -> str:
        row = self.wiki.execute(
            "SELECT label FROM property_labels WHERE pid=? LIMIT 1",
            (pid,),
        ).fetchone()
        return row[0] if row else pid

    @lru_cache(maxsize=32)
    def _graph_until(self, time_iso: str) -> Data:
        sql = (
            "SELECT c.qid subj, c.property pid, c.value_qid obj "
            "FROM claims c JOIN revisions r USING(qid,revision_id) "
            "WHERE r.timestamp<=?"
        )
        df = pd.read_sql(sql, self.wiki, params=(time_iso,))
        df = df[df["subj"].isin(self.qid2int) & df["obj"].isin(self.qid2int)]
        if df.empty:
            return Data()

        src = torch.tensor(df["subj"].map(self.qid2int).to_numpy(), dtype=torch.long)
        dst = torch.tensor(df["obj"].map(self.qid2int).to_numpy(), dtype=torch.long)

        # ★ factorize 하되, 고유 PID 목록을 vocab으로 보관
        codes, uniques = pd.factorize(df["pid"])
        rel = torch.tensor(codes, dtype=torch.long).view(-1, 1)

        G = Data(edge_index=torch.stack([src, dst]), edge_attr=rel)
        G.pid_vocab = [str(p) for p in uniques.tolist()]  # REAL PID strings, e.g., "P31"
        return G

    def acquire_related_external_knowledge(
        self, text: str, time_iso: str, add_reverse=True, add_self_loop=True
    ) -> Tuple[List[int], Data]:
        ids = self.entities_to_id(self.identify_entities(text))
        G = self._graph_until(time_iso)
        if not ids or G.edge_index.numel() == 0:
            return ids, Data()

        mask = (torch.isin(G.edge_index[0], torch.tensor(ids)) |
                torch.isin(G.edge_index[1], torch.tensor(ids)))
        ei, ea = G.edge_index[:, mask], G.edge_attr[mask]

        if add_reverse:
            ei = torch.cat([ei, ei.flip(0)], 1)
            ea = torch.cat([ea, ea], 0)
        if add_self_loop:
            loops = torch.tensor(ids, dtype=torch.long, device=ei.device)
            ei = torch.cat([ei, loops.unsqueeze(0).repeat(2, 1)], 1)
            ea = torch.cat([ea, torch.full((len(loops), 1), -1, dtype=torch.long, device=ei.device)], 0)

        sub = Data(edge_index=ei, edge_attr=ea)
        if hasattr(G, "pid_vocab"):
            sub.pid_vocab = G.pid_vocab  # ★ REAL pid vocab 전달
        return ids, sub

# ─────────────────────────────────────────────
# 텍스트 인코더 (BERT 오프셋 ↔ GLM 히든 매핑)
# ─────────────────────────────────────────────
def build_cross_map(wp_offsets: List[Tuple[int, int]],
                    glm_offsets: List[Tuple[int, int]]) -> List[List[int]]:
    mapping = [[] for _ in wp_offsets]
    p = 0
    for i, (ws, we) in enumerate(wp_offsets):
        while p < len(glm_offsets) and glm_offsets[p][1] <= ws:
            p += 1
        q = p
        while q < len(glm_offsets) and glm_offsets[q][0] < we:
            mapping[i].append(q)
            q += 1
    return mapping

class TextFeatureExtractor:
    def __init__(self):
        self.wp_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.glm_tok: AutoTokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm2-6b", trust_remote_code=True
        )
        # ★ 16-bit 로드 (no 8-bit quantization)
        self.glm = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm2-6b",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            output_hidden_states=True,
            device_map="auto",
        )
        self.proj_down = nn.Linear(4096, 768, bias=False).to(device).half()
        self.proj_up = nn.Linear(768, 4096, bias=False).to(device).half()
        self.norm = nn.LayerNorm(768).to(device).half()
        for m in (self.proj_down, self.proj_up):
            nn.init.xavier_uniform_(m.weight)

    def _manual_offsets(self, text: str, toks: List[str]):
        norm = text.lower()
        p, off = 0, []
        for t in toks:
            tc = t.lstrip(" ")
            tc = tc if tc else " "
            j = norm.find(tc, p)
            j = j if j != -1 else p
            off.append((j, j + len(tc)))
            p = j + len(tc)
        return off

    @torch.no_grad()
    def encode(
        self,
        utterance_text: str,
        knowledge_triples: Optional[List[Tuple[str, str, str]]] = None,  # not used (we override)
        anchor_entities: Optional[List[str]] = None,
        knowledge_text_override: Optional[str] = None,
    ):
        # knowledge_text 구성 (override 우선)
        if knowledge_text_override is not None:
            knowledge_text = knowledge_text_override
        else:
            know_parts = []
            if anchor_entities:
                know_parts.append(" [ENT] ".join(map(str, anchor_entities)))
            if knowledge_triples:
                tmp = []
                for h, r, t in knowledge_triples:
                    tmp.append(f"{h} [R] {r} [T] {t} [TRI] ")
                know_parts.append("".join(tmp).strip())
            knowledge_text = " ".join(know_parts) if know_parts else None

        # BERT: WP 토큰/오프셋/타입
        wp = self.wp_tok(
            utterance_text,
            text_pair=knowledge_text,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        wp = _to_device(wp, self.glm.device)
        wp_tokens = self.wp_tok.convert_ids_to_tokens(wp["input_ids"][0])
        wp_offsets = wp["offset_mapping"][0].tolist()
        token_types = wp["token_type_ids"][0].tolist()
        sep_idx = [i for i, tok in enumerate(wp_tokens) if tok == "[SEP]"]

        # knowledge 블록(WP 인덱스 범위들)
        knowledge_blocks: List[Tuple[int, int]] = []
        in_block, start = False, None
        for i, (tt, tok) in enumerate(zip(token_types, wp_tokens)):
            if tt == 1 and tok != "[SEP]":
                if not in_block:
                    start, in_block = i, True
            else:
                if in_block:
                    knowledge_blocks.append((start, i))
                    in_block = False
        if in_block:
            knowledge_blocks.append((start, len(wp_tokens)))

        # GLM: 합쳐진 문자열 기반 임베딩
        merged = utterance_text + (" [SEP] " + knowledge_text if knowledge_text else "")
        glm_ids = self.glm_tok.encode(merged, add_special_tokens=False)
        glm_enc = {
            "input_ids": torch.tensor([glm_ids], device=self.glm.device),
            "attention_mask": torch.ones(1, len(glm_ids), dtype=torch.long, device=self.glm.device),
        }
        glm_tokens = self.glm_tok.convert_ids_to_tokens(glm_ids)
        try:
            glm_offsets = self.glm_tok(
                merged, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
        except Exception:
            glm_offsets = self._manual_offsets(merged, glm_tokens)

        hidden4096 = self.glm(**glm_enc).hidden_states[-1][0]  # [L,4096]
        hid768 = self.norm(self.proj_down(hidden4096))         # [L,768]

        # WP ↔ GLM 토큰 정렬 매핑
        map_wp2glm = build_cross_map(wp_offsets, glm_offsets)
        max_idx = hid768.size(0)
        wp_emb = torch.stack([
            (
                hid768[torch.tensor(valid, dtype=torch.long, device=hid768.device)].mean(0)
                if (valid := [i for i in ids if i < max_idx])
                else torch.zeros(768, device=hid768.device)
            )
            for ids in map_wp2glm
        ])  # [N_wp,768]

        meta = {
            "wp_tokens": wp_tokens,
            "glm_tokens": glm_tokens,
            "map_wp2glm": map_wp2glm,
            "sep": sep_idx,
            "knowledge_blocks": knowledge_blocks,
            "wp_offsets": wp_offsets,
        }
        return wp_emb, meta

    def cleanup(self):
        print("TextFeatureExtractor 메모리 해제 중...")
        if hasattr(self, "glm"): del self.glm
        if hasattr(self, "proj_down"): del self.proj_down
        if hasattr(self, "proj_up"): del self.proj_up
        torch.cuda.empty_cache()
        print("TextFeatureExtractor 메모리 해제 완료")

# ─────────────────────────────────────────────
# 유틸: 언급 문자열이 덮는 WP 토큰 찾기(발화 구간)
# ─────────────────────────────────────────────
def _find_token_indices_for_phrase(
    text: str,
    phrase: str,
    wp_offsets: List[Tuple[int, int]],
    text_range: Tuple[int, int],  # (wp_start, wp_end) in WP index
) -> List[int]:
    t_norm = text.lower()
    p_norm = phrase.lower().strip()
    if not p_norm:
        return []
    spans = []
    start = 0
    while True:
        j = t_norm.find(p_norm, start)
        if j == -1:
            break
        spans.append((j, j + len(p_norm)))
        start = j + len(p_norm)

    wp_start, wp_end = text_range
    hits = []
    for (s, e) in spans:
        for i in range(wp_start, wp_end):
            ws, we = wp_offsets[i]
            if we > s and ws < e:  # overlap
                hits.append(i)
    return sorted(set(hits))

# ─────────────────────────────────────────────
# GraphBuilder (브리징 방식: text↔relation↔tail)
# ─────────────────────────────────────────────
class GraphBuilder:
    """발화 그래프 + 외부 지식 브리징(관계/테일을 라벨 임베딩으로)"""
    MAX_KG = 5  # 외부지식 최대 부착 개수

    def __init__(self, *, merge_anchor: bool = False):
        self.merge_anchor = merge_anchor
        self.text_enc = TextFeatureExtractor()
        self.ekm = ExternalFinancialKnowledgeModel()

    def __del__(self):
        if hasattr(self, "text_enc"):
            self.text_enc.cleanup()
            del self.text_enc
        print("GraphBuilder 메모리 해제 완료")
        log_vram("del", device)

    @staticmethod
    def _build_knowledge_text_and_spans(
        triples_lbl: List[Tuple[str, str, str]]
    ) -> Tuple[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        triples_lbl: [(head_label, rel_label, tail_label), ...]
        return:
          knowledge_text: "h [R] r [T] t [TRI] h [R] r [T] t [TRI] ..."
          spans: [ ((rel_start,rel_end),(tail_start,tail_end)), ... ]  (char offsets in knowledge_text)
        """
        parts: List[str] = []
        spans: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        pos = 0
        for (h, r, t) in triples_lbl:
            parts.append(h); pos += len(h)
            sepR = " [R] "; parts.append(sepR); pos += len(sepR)

            rel_start = pos
            parts.append(r); pos += len(r)
            rel_end = pos

            sepT = " [T] "; parts.append(sepT); pos += len(sepT)

            tail_start = pos
            parts.append(t); pos += len(t)
            tail_end = pos

            sepTri = " [TRI] "; parts.append(sepTri); pos += len(sepTri)

            spans.append(((rel_start, rel_end), (tail_start, tail_end)))

        return "".join(parts).strip(), spans

    @staticmethod
    def _charspan_to_wp_indices(
        span: Tuple[int, int],
        wp_offsets: List[Tuple[int, int]],
        wp_block: Tuple[int, int],  # knowledge block in WP index (start,end)
    ) -> List[int]:
        """knowledge_text의 char span → 해당하는 '전체 시퀀스'의 WP 인덱스 목록"""
        s, e = span
        b0, b1 = wp_block
        hit = []
        for i in range(b0, b1):
            ws, we = wp_offsets[i]
            if we > s and ws < e:  # overlap
                hit.append(i)
        return hit

    def build(
        self,
        utterance_text: str,
        time_iso: str,
        video_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
    ) -> Data:
        # (1) 외부지식 수집
        ent_ids, ek_sub = self.ekm.acquire_related_external_knowledge(utterance_text, time_iso)

        triples_qpt: List[Tuple[str, str, str]] = []  # (h_qid, pid, t_qid) with REAL pid
        if ek_sub.num_edges:
            pid_vocab = getattr(ek_sub, "pid_vocab", None)
            if pid_vocab is None:
                print("[WARN] ek_sub.pid_vocab 이 없습니다. PID 매핑이 불가합니다.")
            for s, d, (code,) in zip(*ek_sub.edge_index, ek_sub.edge_attr):
                pid = pid_vocab[code] if pid_vocab is not None else f"P{int(code)}"
                triples_qpt.append(
                    (self.ekm.int2qid[s.item()], pid, self.ekm.int2qid[d.item()])
                )

        # (2) QID/PID → 라벨
        triples_lbl: List[Tuple[str, str, str]] = []
        triples_keep: List[Tuple[str, str, str]] = []
        for (hq, pid, tq) in triples_qpt:
            h_label = self.ekm.qid_to_label(hq)
            r_label = self.ekm.pid_to_label(pid)
            t_label = self.ekm.qid_to_label(tq)
            triples_lbl.append((h_label, r_label, t_label))
            triples_keep.append((hq, pid, tq))

        if len(triples_lbl) > self.MAX_KG:
            triples_lbl = triples_lbl[: self.MAX_KG]
            triples_keep = triples_keep[: self.MAX_KG]

        # (3) knowledge_text + 스팬
        knowledge_text, spans_per_triple = self._build_knowledge_text_and_spans(triples_lbl)

        # (4) 인코딩(하나의 패스)
        wp_emb, meta = self.text_enc.encode(
            utterance_text=utterance_text,
            knowledge_triples=None,  # override 사용
            anchor_entities=[self.ekm.int2qid[i] for i in ent_ids],
            knowledge_text_override=knowledge_text if triples_lbl else None,
        )
        if wp_emb.is_cuda:
            wp_emb = wp_emb.cpu()

        hs = wp_emb; D = hs.size(1)
        sep0 = meta["sep"][0] if meta["sep"] else -1
        wp_offsets = meta.get("wp_offsets", [])
        kb = meta.get("knowledge_blocks", [])
        kb_range = kb[0] if kb else None

        node_feats: List[torch.Tensor] = []
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_type: List[int] = []
        node_types: List[int] = []  # 0=text, 1=knowledge, 2=video, 3=audio

        # (5) 텍스트 토큰
        text_nodes: List[int] = []
        if sep0 > 1:
            text_start = len(node_feats)
            node_feats.extend([hs[i] for i in range(1, sep0)])  # [CLS]=0 제외
            text_nodes = list(range(text_start, text_start + (sep0 - 1)))
            node_types.extend([0] * (sep0 - 1))
            for i in range(len(text_nodes) - 1):
                _add_bidir(text_nodes[i], text_nodes[i+1], EDGE_TYPE["t_t"], edge_src, edge_dst, edge_type)

        # (6) 비/오
        v_idx = -1; a_idx = -1
        if video_emb is not None and video_emb.numel():
            v_idx = len(node_feats)
            node_feats.append(video_emb.to(hs.dtype).squeeze(0))
            node_types.append(2)
            for t in text_nodes:
                _add_bidir(t, v_idx, EDGE_TYPE["t_v"], edge_src, edge_dst, edge_type)
        if audio_emb is not None and audio_emb.numel():
            a_idx = len(node_feats)
            node_feats.append(audio_emb.to(hs.dtype).squeeze(0))
            node_types.append(3)
            for t in text_nodes:
                _add_bidir(t, a_idx, EDGE_TYPE["t_a"], edge_src, edge_dst, edge_type)

        # (7) 브리징
        mentions = self.ekm.identify_entities(utterance_text)
        mention2qid = {m: self.ekm.label2qid[m.lower()] for m in mentions if m.lower() in self.ekm.label2qid}

        text_range = (1, sep0) if sep0 > 1 else (1, 1)
        mention_anchor_wp: Dict[str, List[int]] = {}
        for m in mention2qid.keys():
            wp_idxs = _find_token_indices_for_phrase(utterance_text, m, wp_offsets, text_range)
            if wp_idxs:
                mention_anchor_wp[m] = wp_idxs

        tail_qid_to_node: Dict[str, int] = {}
        knowledge_nodes_added = 0

        # 디버그 카운터
        dbg_rel_edges = 0
        dbg_tail_edges = 0
        dbg_trip_ok = 0
        dbg_trip_skip = 0

        for k, ((h_qid, pid, t_qid), (h_label, r_label, t_label)) in enumerate(zip(triples_keep, triples_lbl)):
            if kb_range is None:
                dbg_trip_skip += 1
                continue

            rel_span, tail_span = spans_per_triple[k]
            rel_wp = self._charspan_to_wp_indices(rel_span, wp_offsets, kb_range)
            tail_wp = self._charspan_to_wp_indices(tail_span, wp_offsets, kb_range)

            if not rel_wp or not tail_wp:
                dbg_trip_skip += 1
                continue

            rel_vec = hs[torch.tensor(rel_wp)].mean(dim=0)
            if t_qid in tail_qid_to_node:
                tail_node = tail_qid_to_node[t_qid]
            else:
                tail_vec = hs[torch.tensor(tail_wp)].mean(dim=0)
                tail_node = len(node_feats)
                node_feats.append(tail_vec)
                node_types.append(1)
                tail_qid_to_node[t_qid] = tail_node
                knowledge_nodes_added += 1

            rel_node = len(node_feats)
            node_feats.append(rel_vec)
            node_types.append(1)
            knowledge_nodes_added += 1

            # 헤드 앵커 탐색
            head_anchor_nodes: List[int] = []
            for m, mq in mention2qid.items():
                if mq == h_qid and m in mention_anchor_wp:
                    for ti in mention_anchor_wp[m]:
                        li = ti - 1
                        if 0 <= li < len(text_nodes):
                            head_anchor_nodes.append(text_nodes[li])
            if not head_anchor_nodes:
                # 라벨 fallback
                wp_idxs = _find_token_indices_for_phrase(utterance_text, h_label, wp_offsets, text_range)
                for ti in wp_idxs:
                    li = ti - 1
                    if 0 <= li < len(text_nodes):
                        head_anchor_nodes.append(text_nodes[li])

            # 연결
            for hnode in sorted(set(head_anchor_nodes)):
                _add_bidir(hnode, rel_node, EDGE_TYPE["t_k"], edge_src, edge_dst, edge_type)
                dbg_rel_edges += 2
            _add_bidir(rel_node, tail_node, EDGE_TYPE["k_k"], edge_src, edge_dst, edge_type)
            dbg_tail_edges += 2
            dbg_trip_ok += 1

        if dbg_trip_ok == 0 and len(triples_lbl) > 0:
            print(f"[KG] triples={len(triples_lbl)} 있었지만, 스팬 정렬 실패로 브리징 0개 (skip={dbg_trip_skip})")
        else:
            print(f"[KG] bridged_triples={dbg_trip_ok}  t_k_edges={dbg_rel_edges}  k_k_edges={dbg_tail_edges}")

        # (8) Data
        x = torch.stack(node_feats) if node_feats else torch.empty(0, D)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.empty(2, 0, dtype=torch.long)
        edge_type_t = torch.tensor(edge_type, dtype=torch.long) if edge_type else torch.empty(0, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type_t)
        data.node_meta = {
            "text_nodes": len(text_nodes),
            "knowledge_nodes": knowledge_nodes_added,
            "video_nodes": 1 if v_idx != -1 else 0,
            "audio_nodes": 1 if a_idx != -1 else 0,
            "triples": len(triples_lbl),
        }
        data.utt_meta = {
            "first_text_node": text_nodes[0] if text_nodes else -1,
            "last_text_node": text_nodes[-1] if text_nodes else -1,
        }
        data.node_type     = torch.tensor(node_types, dtype=torch.int8)
        data.idx_text      = torch.tensor(text_nodes, dtype=torch.long) if text_nodes else torch.empty(0, dtype=torch.long)
        data.idx_knowledge = torch.tensor([i for i, t in enumerate(node_types) if t == 1], dtype=torch.long)
        data.idx_video     = torch.tensor([v_idx], dtype=torch.long) if v_idx != -1 else torch.empty(0, dtype=torch.long)
        data.idx_audio     = torch.tensor([a_idx], dtype=torch.long) if a_idx != -1 else torch.empty(0, dtype=torch.long)

        # 디버그 정보
        data.debug_kg = {
            "triples_lbl": triples_lbl,
            "kb_range": kb_range,
            "mentions": mentions,
            "mention_anchor_wp": mention_anchor_wp,
            "bridged_ok": dbg_trip_ok,
            "bridged_skip": dbg_trip_skip,
        }
        return data

# ─────────────────────────────────────────────
# 그래프 병합 (비디오/오디오 크로스 연결 포함)
# ─────────────────────────────────────────────
def merge_graph(prev_graph: Optional[Data], current_graph: Data) -> Data:
    if prev_graph is None or prev_graph.x.numel() == 0:
        return current_graph

    # 노드 피처 병합
    x_merged = torch.cat([prev_graph.x, current_graph.x], dim=0)

    # 엣지 병합(오프셋)
    num_prev_nodes = prev_graph.x.size(0)
    edge_index_current_offset = current_graph.edge_index + num_prev_nodes
    edge_index_merged = torch.cat([prev_graph.edge_index, edge_index_current_offset], dim=1)
    edge_type_merged  = torch.cat([prev_graph.edge_type, current_graph.edge_type])

    # utterance 링크(양방향)
    if prev_graph.utt_meta["last_text_node"] != -1 and current_graph.utt_meta["first_text_node"] != -1:
        prev_last_node  = prev_graph.utt_meta["last_text_node"]
        curr_first_node = current_graph.utt_meta["first_text_node"] + num_prev_nodes
        utt_edge_index = torch.tensor([[prev_last_node, curr_first_node],
                                       [curr_first_node, prev_last_node]], dtype=torch.long)
        utt_edge_type  = torch.tensor([EDGE_TYPE["utt"], EDGE_TYPE["utt"]], dtype=torch.long)
        edge_index_merged = torch.cat([edge_index_merged, utt_edge_index], dim=1)
        edge_type_merged  = torch.cat([edge_type_merged,  utt_edge_type])

    # 비디오/오디오 크로스-utterance 연결
    def _idx_list(g: Data, attr: str, type_id: Optional[int], allow_meta_fallback: bool, offset: int) -> List[int]:
        t = getattr(g, attr, None)
        if t is not None and hasattr(t, "numel") and t.numel() > 0:
            return (t + offset).tolist()

        if type_id is not None and hasattr(g, "node_type") and g.node_type is not None and g.node_type.numel() > 0:
            idxs = torch.nonzero(g.node_type.to(torch.long) == int(type_id), as_tuple=True)[0]
            if idxs.numel() > 0:
                return (idxs + offset).tolist()

        if allow_meta_fallback and hasattr(g, "node_meta"):
            text_n = int(g.node_meta.get("text_nodes", 0))
            know_n = int(g.node_meta.get("knowledge_nodes", 0))
            has_v  = int(g.node_meta.get("video_nodes", 0)) == 1
            has_a  = int(g.node_meta.get("audio_nodes", 0)) == 1
            base = text_n + know_n
            if attr == "idx_video" and has_v:
                return [offset + base]
            if attr == "idx_audio" and has_a:
                return [offset + base + (1 if has_v else 0)]
        return []

    prev_v_list = _idx_list(prev_graph, "idx_video", 2, allow_meta_fallback=False, offset=0)
    prev_a_list = _idx_list(prev_graph, "idx_audio", 3, allow_meta_fallback=False, offset=0)

    curr_v_list = _idx_list(current_graph, "idx_video", 2, allow_meta_fallback=True,  offset=num_prev_nodes)
    curr_a_list = _idx_list(current_graph, "idx_audio", 3, allow_meta_fallback=True,  offset=num_prev_nodes)

    if prev_v_list and curr_v_list:
        pv = prev_v_list[-1]
        vv_ei = torch.tensor([[pv] * len(curr_v_list), curr_v_list], dtype=torch.long)
        vv_ei = torch.cat([vv_ei, vv_ei.flip(0)], dim=1)
        vv_et = torch.full((vv_ei.size(1),), EDGE_TYPE["v_v"], dtype=torch.long)
        edge_index_merged = torch.cat([edge_index_merged, vv_ei], dim=1)
        edge_type_merged  = torch.cat([edge_type_merged,  vv_et])

    if prev_a_list and curr_a_list:
        pa = prev_a_list[-1]
        aa_ei = torch.tensor([[pa] * len(curr_a_list), curr_a_list], dtype=torch.long)
        aa_ei = torch.cat([aa_ei, aa_ei.flip(0)], dim=1)
        aa_et = torch.full((aa_ei.size(1),), EDGE_TYPE["a_a"], dtype=torch.long)
        edge_index_merged = torch.cat([edge_index_merged, aa_ei], dim=1)
        edge_type_merged  = torch.cat([edge_type_merged,  aa_et])

    # 메타 병합
    node_meta_merged = {k: prev_graph.node_meta.get(k, 0) + current_graph.node_meta.get(k, 0)
                        for k in set(prev_graph.node_meta.keys()) | set(current_graph.node_meta.keys())}
    utt_meta_merged = {
        "first_text_node": prev_graph.utt_meta["first_text_node"],
        "last_text_node":  current_graph.utt_meta["last_text_node"] + num_prev_nodes,
    }

    merged_graph = Data(x=x_merged, edge_index=edge_index_merged, edge_type=edge_type_merged)
    merged_graph.node_meta = node_meta_merged
    merged_graph.utt_meta  = utt_meta_merged

    # node_type / idx_* 병합
    if hasattr(prev_graph, "node_type") or hasattr(current_graph, "node_type"):
        nt_prev = getattr(prev_graph, "node_type", None)
        nt_curr = getattr(current_graph, "node_type", None)
        if nt_prev is not None and nt_curr is not None:
            merged_graph.node_type = torch.cat([nt_prev, nt_curr], dim=0)
        elif nt_prev is not None:
            merged_graph.node_type = nt_prev
        elif nt_curr is not None:
            merged_graph.node_type = nt_curr

    def _cat_idx(prev_t: Optional[torch.Tensor], curr_t: Optional[torch.Tensor], offset: int):
        if prev_t is None and curr_t is None:
            return None
        if prev_t is None:
            return (curr_t + offset) if (curr_t is not None and curr_t.numel()) else curr_t
        if curr_t is None or not curr_t.numel():
            return prev_t
        return torch.cat([prev_t, curr_t + offset])

    merged_graph.idx_text      = _cat_idx(getattr(prev_graph, "idx_text", None),      getattr(current_graph, "idx_text", None),      num_prev_nodes)
    merged_graph.idx_knowledge = _cat_idx(getattr(prev_graph, "idx_knowledge", None), getattr(current_graph, "idx_knowledge", None), num_prev_nodes)
    merged_graph.idx_video     = _cat_idx(getattr(prev_graph, "idx_video", None),     getattr(current_graph, "idx_video", None),     num_prev_nodes)
    merged_graph.idx_audio     = _cat_idx(getattr(prev_graph, "idx_audio", None),     getattr(current_graph, "idx_audio", None),     num_prev_nodes)

    return merged_graph
def build_and_cache_graphs():
    """
    ready_videos.csv에 있는 video_id에 대해 그래프를 구축하고 캐시로 저장합니다.
    """
    # 파일 경로 설정
    ready_videos_path = "data/ready_videos.csv"
    speech_db_path = "data/speech_segments.db"
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    # 1. ready_videos.csv에서 video_id 목록 불러오기
    try:
        ready_videos_df = pd.read_csv(ready_videos_path)
        video_ids = ready_videos_df["video_id"].tolist()
        print(f"총 {len(video_ids)}개의 video_id를 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: {ready_videos_path} 파일을 찾을 수 없습니다.")
        return

    # 2. speech_segments.db 연결
    try:
        conn = sqlite3.connect(speech_db_path)
        conn.row_factory = sqlite3.Row
        print(f"{speech_db_path}에 연결되었습니다.")
    except sqlite3.Error as e:
        print(f"오류: {speech_db_path} 연결 실패 - {e}")
        return

    # [추가됨] GraphBuilder 인스턴스를 루프 밖에서 한 번만 생성
    print("GraphBuilder 인스턴스를 생성합니다. 모델 로딩으로 시간이 소요될 수 있습니다...")
    try:
        graph_builder = GraphBuilder()
    except Exception as e:
        print(f"오류: GraphBuilder 인스턴스 생성 실패 - {e}")
        conn.close()
        return

    # 3. 각 video_id에 대해 그래프 구축 및 캐시 저장
    for video_id in video_ids:
        # 🟢 추가된 로직: 최종 그래프 파일이 이미 존재하는지 확인
        final_graph_path = cache_dir / _hash(f"video_graph::{video_id}")
        if final_graph_path.exists():
            print(f"\n[Video ID: {video_id}] 최종 그래프가 이미 존재합니다. 건너뜁니다.")
            continue
            
        print(f"\n[Video ID: {video_id}] 그래프 구축 시작...")
        
        # 비디오 업로드 시간 조회
        try:
            video_meta = conn.execute(
                "SELECT published_date FROM video_metadata WHERE video_id = ?",
                (video_id,)
            ).fetchone()
            if not video_meta:
                print(f"경고: video_id {video_id}에 대한 메타데이터를 찾을 수 없습니다. 건너뜁니다.")
                continue
            upload_time = video_meta["published_date"]
        except sqlite3.Error as e:
            print(f"오류: video_id {video_id}의 업로드 시간 조회 실패 - {e}")
            continue

        # [수정됨] 루프 내에서 GraphBuilder를 생성하는 대신, 이미 생성된 객체 사용
        # graph_builder = GraphBuilder(time_iso=upload_time) # 이 라인 제거

        video_graph = None

        # 비디오에 대한 발화 불러오기
        speech_segments = conn.execute(
            "SELECT segment_id, script FROM speech_segments WHERE video_id = ? ORDER BY start_time",
            (video_id,)
        ).fetchall()
        
        if not speech_segments:
            print(f"경고: video_id {video_id}에 대한 발화가 없습니다. 그래프를 생성하지 않습니다.")
            continue
        
        for segment in speech_segments:
            seg_id = segment["segment_id"]
            utterance_text = segment["script"]
            
            # [추가됨] 빈 발화 텍스트 건너뛰기
            if not utterance_text or not utterance_text.strip():
                print(f" - 경고: 발화 '{seg_id}'의 텍스트가 비어있어 건너뜁니다.")
                continue
                
            print(f" - 발화 '{seg_id}' 처리 중...")

            # 캐시 파일 경로 설정
            video_key = f"vid::{seg_id}"
            audio_key = f"aud::{seg_id}"
            video_cache_path = cache_dir / _hash(video_key)
            audio_cache_path = cache_dir / _hash(audio_key)

            # 임베딩 텐서 로드
            video_emb, audio_emb = None, None
            try:
                if video_cache_path.exists():
                    video_emb = torch.load(video_cache_path)
                if audio_cache_path.exists():
                    audio_emb = torch.load(audio_cache_path)
            except Exception as e:
                print(f"오류: 발화 '{seg_id}'의 캐시 파일 로드 실패 - {e}")
                continue

            # 단일 발화 그래프 구축
            try:
                # [수정됨] build 메소드에 time_iso 인자 전달
                current_graph = graph_builder.build(
                    utterance_text=utterance_text,
                    time_iso=upload_time, 
                    video_emb=video_emb,
                    audio_emb=audio_emb
                )
            except Exception as e:
                print(f"오류: 발화 '{seg_id}'에 대한 그래프 구축 실패 - {e}")
                continue

            # 이전 그래프와 병합
            video_graph = merge_graph(video_graph, current_graph)

        # 전체 비디오 그래프 캐시 저장
        if video_graph:
            # 🟢 추가된 로직: 이전에 존재하지 않았던 경우에만 저장
            torch.save(video_graph, final_graph_path)
            print(f"\n성공: video_id {video_id}에 대한 최종 그래프가 {final_graph_path}에 저장되었습니다.")
        
        # [수정됨] 루프 내에서 del graph_builder 제거

    # [추가됨] 모든 작업이 끝난 후 GraphBuilder 인스턴스 명시적으로 삭제
    print("\n모든 비디오 처리가 완료되었습니다. GraphBuilder 리소스를 해제합니다.")
    del graph_builder
    torch.cuda.empty_cache()

    conn.close()
    print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    build_and_cache_graphs()