"""
MANAGER Prototype – Multimodal Encoders
"""

from __future__ import annotations
import os, re, json, sqlite3
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Dict

import torch, torch.nn as nn
import pandas as pd
import librosa
from PIL import Image
try:
    from torch_geometric.data import Data
except: pass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    BertTokenizerFast,
    AutoFeatureExtractor,
    HubertModel,
    BeitModel,
    BeitImageProcessor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_device(d: Dict[str, torch.Tensor], dev, dtype=None):
    if dtype is None:
        return {k: v.to(dev) for k, v in d.items()}
    return {k: v.to(dev).to(dtype) for k, v in d.items()}
# ════════════════════════════════════════════════════════════════════
# External Financial Knowledge Model
# ════════════════════════════════════════════════════════════════════
class ExternalFinancialKnowledgeModel:
    def __init__(self,
                 wiki_db="data/wikidata_revisions.db",
                 speech_db="data/speech_segments.db"):
        self.wiki = sqlite3.connect(wiki_db); self.wiki.row_factory = sqlite3.Row
        self.speech = sqlite3.connect(speech_db); self.speech.row_factory = sqlite3.Row

        self.target_entities = self._load_target_entities()
        label_df = pd.read_sql("SELECT DISTINCT qid,value AS label FROM labels "
                               "WHERE lang='en';", self.wiki)
        label_df = label_df[label_df["label"].str.lower()
                            .isin([n.lower() for n in self.target_entities])]
        label_df["id_int"] = pd.factorize(label_df["qid"])[0]

        self.qid2int = dict(zip(label_df["qid"], label_df["id_int"]))
        self.label2qid = {l.lower(): q for q, l in zip(label_df["qid"],
                                                       label_df["label"])}

        safe = [re.sub(r"\s+", r"\\s+", re.escape(n.lower()))
                for n in self.target_entities]
        self._pat = re.compile(r"(" + "|".join(safe) + r")", re.I)

    def _load_target_entities(self) -> List[str]:
        df = pd.read_sql("SELECT persons_found FROM video_metadata;", self.speech)
        names = set()
        for js in df["persons_found"]:
            if js: names.update(json.loads(js).keys())
        return list(names)

    # --------- entity id helpers ------------------------------------
    def identify_entities(self, text: str) -> List[str]:
        t = re.sub(r"[^\w\s]", "", text.lower())
        return list({m.strip() for m in self._pat.findall(t)})

    def entities_to_id(self, ents: List[str]) -> List[int]:
        return [self.qid2int[self.label2qid[e.lower()]]
                for e in ents if e.lower() in self.label2qid]

    # --------- graph snapshot ---------------------------------------
    @lru_cache(maxsize=32)
    def _graph_until(self, time_iso: str) -> Data:
        sql = ("SELECT c.qid subj,c.property pid,c.value_qid obj "
               "FROM claims c JOIN revisions r USING(qid,revision_id) "
               "WHERE r.timestamp<=?")
        df = pd.read_sql(sql, self.wiki, params=(time_iso,))
        df = df[df["subj"].isin(self.qid2int) & df["obj"].isin(self.qid2int)]
        if df.empty: return Data()

        src = torch.tensor(df["subj"].map(self.qid2int).to_numpy(), dtype=torch.long)
        dst = torch.tensor(df["obj"].map(self.qid2int).to_numpy(), dtype=torch.long)
        rel = torch.tensor(pd.factorize(df["pid"])[0], dtype=torch.long).view(-1, 1)
        return Data(edge_index=torch.stack([src, dst]), edge_attr=rel)

    # --------- public API with reverse & self-loop ------------------
    def acquire_related_external_knowledge(
        self, text: str, time_iso: str,
        add_reverse=True, add_self_loop=True
    ) -> Tuple[List[int], Data]:
        ids = self.entities_to_id(self.identify_entities(text))
        G = self._graph_until(time_iso)
        if not ids or G.edge_index.numel() == 0: return ids, Data()

        mask = (torch.isin(G.edge_index[0], torch.tensor(ids)) |
                torch.isin(G.edge_index[1], torch.tensor(ids)))
        ei, ea = G.edge_index[:, mask], G.edge_attr[mask]

        if add_reverse:
            ei = torch.cat([ei, ei.flip(0)], 1)
            ea = torch.cat([ea, ea], 0)
        if add_self_loop:
            loops = torch.tensor(ids, dtype=torch.long, device=ei.device)
            ei = torch.cat([ei, loops.unsqueeze(0).repeat(2, 1)], 1)
            ea = torch.cat([ea, torch.full((len(loops), 1), -1,
                                           dtype=torch.long, device=ei.device)], 0)
        return ids, Data(edge_index=ei, edge_attr=ea)

# ════════════════════════════════════════════════════════════════════
# Token offset mapping util
# ════════════════════════════════════════════════════════════════════
def build_cross_map(wp_offsets: List[Tuple[int, int]],
                    glm_offsets: List[Tuple[int, int]]) -> List[List[int]]:
    mapping = [[] for _ in wp_offsets]; p = 0
    for i,(ws,we) in enumerate(wp_offsets):
        while p < len(glm_offsets) and glm_offsets[p][1] <= ws: p += 1
        q = p
        while q < len(glm_offsets) and glm_offsets[q][0] < we:
            mapping[i].append(q); q += 1
    return mapping

# ════════════════════════════════════════════════════════════════════
#  Text Feature Extractor – 최종 수정본
# ════════════════════════════════════════════════════════════════════
class TextFeatureExtractor:
    def __init__(self):
        self.wp_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # ChatGLM Tokenizer 로드 (trust_remote_code=True)
        self.glm_tok: AutoTokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm2-6b", trust_remote_code=True
        )
        self.fast_offset = True # 빠른 토크나이저 사용 가능 여부 (버전 따라 다름)

        # ChatGLM Model 로드 (양자화 사용)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.glm = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm2-6b",
            trust_remote_code=True,
            quantization_config=quantization_config,
            output_hidden_states=True,
            device_map="auto"
        )

        self.proj_down = nn.Linear(4096, 768, bias=False).to(device).half()  # .half() 추가
        self.proj_up   = nn.Linear(768, 4096, bias=False).to(device).half()  # .half() 추가
        self.norm      = nn.LayerNorm(768).to(device).half()                # .half() 추가

        # Xavier 초기화는 half() 뒤에 하는 것이 더 안정적일 수 있습니다.
        for m in (self.proj_down, self.proj_up):
            nn.init.xavier_uniform_(m.weight)

    def _manual_offsets(self, text: str, toks: List[str]):
        # ... (이 함수는 수정사항 없음) ...
        norm = text.lower(); p, off = 0, []
        for t in toks:
            tc = t.lstrip(" "); tc = tc if tc else " "
            j = norm.find(tc, p)
            j = j if j != -1 else p
            off.append((j, j + len(tc))); p = j + len(tc)
        return off

    @torch.no_grad()
    def encode(self,
               utterance_text: str,
               knowledge_triples: list | None = None,
               anchor_entities : list | None = None):

        # 1) knowledge 문자열 합치기 --------------------------------------
        know_parts = []
        if anchor_entities:
            know_parts.append(" [ENT] ".join(map(str, anchor_entities)))
        if knowledge_triples:
            know_parts += [f"{h} [R] {r} [T] {t}" for h, r, t in knowledge_triples]
        knowledge_text = " ".join(know_parts) if know_parts else None

        # 2) WordPiece 토큰화 (text_pair 사용) ---------------------------
        wp = self.wp_tok(
            utterance_text,
            text_pair=knowledge_text,
            return_offsets_mapping=True,
            return_token_type_ids=True,     # ← 변경
            return_tensors="pt",
        )
        wp = _to_device(wp, self.glm.device)

        wp_tokens  = self.wp_tok.convert_ids_to_tokens(wp["input_ids"][0])
        wp_offsets = wp["offset_mapping"][0].tolist()
        token_types = wp["token_type_ids"][0].tolist()

        # SEP 위치
        sep_idx = [i for i, tok in enumerate(wp_tokens) if tok == "[SEP]"]

        # knowledge 블록 탐색 (token_type_id == 1 & tok != [SEP])
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

        # 3) ChatGLM 토큰화 ---------------------------------------------
        merged = utterance_text + (" [SEP] " + knowledge_text if knowledge_text else "")
        glm_ids = self.glm_tok.encode(merged, add_special_tokens=False)
        glm_enc = {
            "input_ids": torch.tensor([glm_ids], device=self.glm.device),
            "attention_mask": torch.ones(
                1, len(glm_ids), dtype=torch.long, device=self.glm.device)
        }
        glm_tokens  = self.glm_tok.convert_ids_to_tokens(glm_ids)
        glm_offsets = (self.glm_tok(
            merged, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
                       if hasattr(self.glm_tok, "get_offsets_mapping")
                       else self._manual_offsets(merged, glm_tokens))

        # 4) 히든 → 768 투영 & 매핑 -------------------------------------
        hidden4096 = self.glm(**glm_enc).hidden_states[-1][0]   # [T,4096]
        proj_device = self.proj_down.weight.device
        hidden4096 = hidden4096.to(proj_device)
        hid768     = self.norm(self.proj_down(hidden4096))      # [T,768]

        map_wp2glm = build_cross_map(wp_offsets, glm_offsets)
        max_idx    = hid768.size(0)
        wp_emb = torch.stack([
            (
                hid768[torch.tensor(valid, dtype=torch.long,
                                    device=hid768.device)].mean(0)
                if (valid := [i for i in ids if i < max_idx])
                else torch.zeros(768, device=hid768.device)
            )
            for ids in map_wp2glm
        ])                                                      # [N_wp,768]

        # 5) 메타 반환 ---------------------------------------------------
        meta = {
            "wp_tokens"       : wp_tokens,
            "glm_tokens"      : glm_tokens,
            "map_wp2glm"      : map_wp2glm,
            "sep"             : sep_idx,
            "knowledge_blocks": knowledge_blocks,               # ← 추가
        }
        return wp_emb, meta
    
    def cleanup(self):
        """GPU 메모리 명시적 해제"""
        print("TextFeatureExtractor 메모리 해제 중...")
        
        if hasattr(self, 'glm'):
            del self.glm
        if hasattr(self, 'proj_down'):
            del self.proj_down  
        if hasattr(self, 'proj_up'):
            del self.proj_up
            
        torch.cuda.empty_cache()
        
        print("TextFeatureExtractor 메모리 해제 완료")
# ════════════════════════════════════════════════════════════════════
#  Video Feature Extractor – Temporal Attention
# ════════════════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):       
        x = x.to(self.W.weight.dtype)   # x: [K,768]
        e = self.v(torch.tanh(self.W(x))).squeeze(-1)  # [K]
        a = torch.softmax(e, dim=0)                    # [K]
        return torch.sum(a.unsqueeze(-1) * x, dim=0)   # [768]

class VideoFeatureExtractor:
    def __init__(self, name="microsoft/beit-base-patch16-224", fps=1, max_frames=32):
        self.processor = BeitImageProcessor.from_pretrained(name)
        self.model = BeitModel.from_pretrained(name).to(device).eval().half()
        self.temporal_attn = TemporalAttention(768).to(device)
        self.fps, self.max_frames = fps, max_frames

    @torch.no_grad()
    def _encode_frame(self, img: Image.Image):
        proc = self.processor(images=img, return_tensors="pt")
        inputs = _to_device(proc, device, dtype=torch.float16)
        patches = self.model(**inputs).last_hidden_state[:, 1:, :]  # [1,L,768]
        return patches.mean(1).squeeze(0)                           # [768]

    def encode_clip(self, frames: List[Image.Image]) -> torch.Tensor:
        if not frames: return torch.zeros(768, device=device)

        # 프레임 수 제한
        if len(frames) > self.max_frames:
            idx = torch.linspace(0, len(frames)-1, steps=self.max_frames).long()
            frames = [frames[i] for i in idx]

        vecs = torch.stack([self._encode_frame(f) for f in frames])  # [K,768]
        return self.temporal_attn(vecs)                              # [768]

    def extract_from_video(self, frames_dir: str) -> torch.Tensor:
        fnames = sorted(f for f in os.listdir(frames_dir)
                        if f.lower().endswith((".jpg", ".png")))
        frames = [Image.open(Path(frames_dir)/f).convert("RGB") for f in fnames]
        return self.encode_clip(frames)

# ════════════════════════════════════════════════════════════════════
#  Audio Feature Extractor – 3s window, 50% overlap
# ════════════════════════════════════════════════════════════════════
class AudioFeatureExtractor:
    def __init__(self, name="facebook/hubert-base-ls960"):
        self.processor = AutoFeatureExtractor.from_pretrained(name)
        self.model = HubertModel.from_pretrained(name).to(device).eval().half()

    @torch.no_grad()
    def encode_clip(self, wav_path: str) -> torch.Tensor:
        wav, sr = librosa.load(wav_path, sr=16_000)
        win, hop, vecs = 3*sr, (3*sr)//2, []
        for s in range(0, len(wav), hop):
            seg = wav[s:s+win]
            if len(seg) < sr: continue
            proc = self.processor(torch.tensor(seg), sampling_rate=16_000,
                                  return_tensors="pt")
            inp = _to_device(proc, device, dtype=torch.float16)
            h = self.model(**inp).last_hidden_state.mean(1).squeeze(0)  # [768]
            vecs.append(h)
        return (torch.stack(vecs).mean(0) if vecs
                else torch.zeros(768, device=device))
