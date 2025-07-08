"""
MANAGER 프로토타입 – 텍스트·오디오·이미지 임베딩 (옵션 A: 4096→768 투사)
────────────────────────────────────────────────────────────────────────
• 텍스트   : WordPiece + ChatGLM2 (8-bit) 양쪽 토크나이즈 → 4096→768 투사
• 비디오   : BEiT-base (768d)
• 오디오   : HuBERT-base (768d)
• 그래프 GCN : 노드 피처 768d (뒤에서 768→4096 재투사 후 soft prefix)
"""

from __future__ import annotations

# ── Standard libs ───────────────────────────────────────────────────
import os, re, json, sqlite3
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple

# ── Third-party ────────────────────────────────────────────────────
import torch, torch.nn as nn
import pandas as pd
import librosa
from PIL import Image
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertTokenizerFast,
    AutoFeatureExtractor,
    HubertModel,
    BeitModel,
    BeitImageProcessor,
    BitsAndBytesConfig,
)

# ── Device ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ╔══════════════════════════════════════════════════════════════════╗
# 1. External Financial Knowledge Model
# ╚══════════════════════════════════════════════════════════════════╝
class ExternalFinancialKnowledgeModel:
    """
    ↳ 텍스트에 언급된 인물/엔티티(QID)를 앵커로 시점 제한 서브그래프를 추출
    """

    def __init__(
        self,
        wiki_db: str | Path = "data/wikidata_revisions.db",
        speech_db: str | Path = "data/speech_segments.db",
    ):
        self.wiki = sqlite3.connect(wiki_db)
        self.wiki.row_factory = sqlite3.Row
        self.speech = sqlite3.connect(speech_db)
        self.speech.row_factory = sqlite3.Row

        # ① persons_found 수집
        self.target_entities = self._load_target_entities()

        # ② label ↔ QID ↔ intID
        df = pd.read_sql(
            "SELECT DISTINCT qid, value AS label "
            "FROM labels WHERE lang='en';",
            self.wiki,
        )
        df = df[df["label"].str.lower().isin(
            [n.lower() for n in self.target_entities]
        )].reset_index(drop=True)
        df["id_int"] = pd.factorize(df["qid"])[0]

        self.qid2int = dict(zip(df["qid"], df["id_int"]))
        self.label2qid = {l.lower(): q for q, l in zip(df["qid"], df["label"])}

        safe = [re.sub(r"\s+", r"\\s+", re.escape(n.lower()))
                for n in self.target_entities]
        self._pat = re.compile(r"(" + "|".join(safe) + r")", re.I)

    def _load_target_entities(self) -> List[str]:
        df = pd.read_sql(
            "SELECT persons_found FROM video_metadata;", self.speech
        )
        names = set()
        for js in df["persons_found"]:
            if js:
                names.update(json.loads(js).keys())
        return list(names)

    # ------------- 텍스트 → entity id 들 ---------------------------- #
    def identify_entities(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", "", text.lower())
        return list({m.strip() for m in self._pat.findall(text)})

    def entities_to_id(self, ents: List[str]) -> List[int]:
        return [
            self.qid2int[self.label2qid[e.lower()]]
            for e in ents if e.lower() in self.label2qid
        ]

    # ------------- 그래프 스냅숏 ------------------------------------ #
    @lru_cache(maxsize=32)
    def _graph_until(self, time_iso: str) -> Data:
        sql = (
            "SELECT c.qid subj, c.property pid, c.value_qid obj "
            "FROM claims c JOIN revisions r USING(qid, revision_id) "
            "WHERE r.timestamp <= ?"
        )
        df = pd.read_sql(sql, self.wiki, params=(time_iso,))
        df = df[df["subj"].isin(self.qid2int) & df["obj"].isin(self.qid2int)]
        if df.empty:
            return Data()

        src = df["subj"].map(self.qid2int).to_numpy()
        dst = df["obj"].map(self.qid2int).to_numpy()
        rel = pd.factorize(df["pid"])[0]
        return Data(
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.tensor(rel, dtype=torch.long).view(-1, 1),
        )

    # ------------- 공개 API --------------------------------------- #
    def acquire_related_external_knowledge(
        self, text: str, time_iso: str
    ) -> Tuple[List[int], Data]:
        ids = self.entities_to_id(self.identify_entities(text))
        G = self._graph_until(time_iso)
        if G.num_edges == 0 or not ids:
            return ids, G

        mask = torch.isin(G.edge_index[0], torch.tensor(ids)) | \
               torch.isin(G.edge_index[1], torch.tensor(ids))
        return ids, Data(
            edge_index=G.edge_index[:, mask],
            edge_attr=G.edge_attr[mask],
        )


# ╔══════════════════════════════════════════════════════════════════╗
# 2. 유틸: WordPiece ↔ ChatGLM 토큰 오프셋 매핑
# ╚══════════════════════════════════════════════════════════════════╝
def build_cross_map(
    wp_offsets: List[Tuple[int, int]],
    glm_offsets: List[Tuple[int, int]],
) -> List[List[int]]:
    """
    wp_offsets[i]와 겹치는 ChatGLM 토큰 인덱스를 리스트로 리턴
    """
    mapping: List[List[int]] = [[] for _ in wp_offsets]
    p = 0
    for i, (ws, we) in enumerate(wp_offsets):
        while p < len(glm_offsets) and glm_offsets[p][1] <= ws:
            p += 1
        q = p
        while q < len(glm_offsets) and glm_offsets[q][0] < we:
            mapping[i].append(q)
            q += 1
    return mapping


# ╔══════════════════════════════════════════════════════════════════╗
# 3. Text Feature Extractor (WordPiece + ChatGLM2)
# ╚══════════════════════════════════════════════════════════════════╝
class TextFeatureExtractor:
    """
    • WordPiece (bert-base-uncased) : 앵커 엔티티 정밀 탐색
    • ChatGLM2-6B (8-bit)        : 언어 표현 4096d → 768d 투사
    """

    def __init__(self):
        # ① WordPiece 토크나이저
        self.wp_tok = BertTokenizerFast.from_pretrained(
            "bert-base-uncased",
            add_special_tokens=False,
        )

        # ② ChatGLM2 모델 (+ 토크나이저)
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        self.glm_tok = AutoTokenizer.from_pretrained(
            "THUDM/chatglm2-6b", trust_remote_code=True
        )
        self.glm = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm2-6b",
            trust_remote_code=True,
            quantization_config=quant_cfg,
            output_hidden_states=True,
            device_map="auto",
        )

        # ③ 차원 프로젝션 4096 → 768, 768 → 4096
        self.proj_down = nn.Linear(4096, 768, bias=False).to(device)
        self.proj_up   = nn.Linear(768, 4096, bias=False).to(device)

    # ──────────────────────────────────────────────────────────────
    def encode(
        self,
        utterance_text: str,
        knowledge_triples: List[str] | None = None,  
        anchor_entities: List[str] | None = None,
    ):
        """
        Returns
        -------
        wp_emb     : [N_wp, 768] WordPiece 노드 임베딩
        meta_dict  : 토큰 정보, 매핑 표 등
        """
        # (1) WordPiece 토크나이즈 → offset
        wp_out = self.wp_tok(
            utterance_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        wp_offsets = wp_out["offset_mapping"]

        # (2) ChatGLM 토크나이즈 → offset
        glm_out = self.glm_tok(
            utterance_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        glm_offsets = glm_out["offset_mapping"][0].tolist()  # List[(s,e)]

        # (3) ChatGLM hidden 4096d
        hidden_4096 = self.glm(**glm_out).hidden_states[-1][0]  # [L,4096]

        # (4) 4096 → 768
        hidden_768 = self.proj_down(hidden_4096)                # [L,768]

        # (5) 오프셋 매핑 테이블
        map_wp2glm = build_cross_map(wp_offsets, glm_offsets)

        # (6) WordPiece 노드 임베딩 = 평균
        wp_emb = torch.stack([
            hidden_768[idxs].mean(0) if idxs else torch.zeros(768, device=device)
            for idxs in map_wp2glm
        ])  # [N_wp,768]

        meta = {
            "wp_tokens": self.wp_tok.convert_ids_to_tokens(wp_out["input_ids"]),
            "glm_tokens": self.glm_tok.convert_ids_to_tokens(glm_out["input_ids"][0]),
            "map_wp2glm": map_wp2glm,
            "offsets_wp": wp_offsets,
            "offsets_glm": glm_offsets,
        }
        return wp_emb, meta


# ╔══════════════════════════════════════════════════════════════════╗
# 4. Video Feature Extractor ‒ BEiT-base → 768d (두 단계 평균)
# ╚══════════════════════════════════════════════════════════════════╝
class VideoFeatureExtractor:
    """
    • 프레임 v_kj : BEiT 패치 토큰 평균 → x_frame (768d)
    • 클립  v_j  : 프레임 x_frame 들을 평균 → x_clip (768d)
    """
    def __init__(self, model_name: str = "microsoft/beit-base-patch16-224"):
        self.processor = BeitImageProcessor.from_pretrained(model_name)
        self.model = BeitModel.from_pretrained(model_name).to(device).eval()
        self.model.half()                       # fp16 메모리 절약

    @torch.no_grad()
    def _encode_frame(self, img: Image.Image) -> torch.Tensor:
        """한 프레임 → 768 벡터 (패치 평균)"""
        inputs = self.processor(images=img, return_tensors="pt").to(device)
        out = self.model(**{k: v.half() for k, v in inputs.items()})
        # out.last_hidden_state : [1, L_patch+1, 768]  (앞 1개 CLS 제외)
        patch_feats = out.last_hidden_state[:, 1:, :]            # [1, L, 768]
        return patch_feats.mean(dim=1).squeeze(0)               # [768]

    def encode_clip(self, frames: List[Image.Image]) -> torch.Tensor:
        """발화 구간의 프레임 리스트 → 768 벡터"""
        if not frames:
            return torch.zeros(768, device=device)
        frame_vecs = torch.stack([self._encode_frame(f) for f in frames])  # [N,768]
        return frame_vecs.mean(dim=0)                                      # [768]

    def extract_from_video(self, frames_dir: str) -> torch.Tensor:
        """
        기존 폴더 전체 프레임 평균이 아니라,
        호출 측에서 "한 utterance 에 해당하는 프레임들"을 넘겨야 논문과 동일해집니다.
        예시용으로 폴더의 모든 jpg/png 를 평균.
        """
        fnames = sorted(
            f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))
        )
        frames = [Image.open(Path(frames_dir) / f).convert("RGB") for f in fnames]
        return self.encode_clip(frames)          # [768]



# ╔══════════════════════════════════════════════════════════════════╗
# 5. Audio Feature Extractor (HuBERT-base → 768d CLS)
# ╚══════════════════════════════════════════════════════════════════╝
# 5. Audio Feature Extractor – 단일 wav 입력 버전
class AudioFeatureExtractor:
    def __init__(self, model_name="facebook/hubert-base-ls960"):
        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = (
            HubertModel.from_pretrained(model_name)
            .to(self.device)
            .eval()
            .half()
        )

    @torch.no_grad()
    def encode_clip(self, wav_path: str) -> torch.Tensor:
        """
        하나의 wav 파일(= 한 utterance) → 768-차 벡터
        1) wav 로드
        2) HuBERT 통과 → [1, T, 768]
        3) 시간축 평균 → [768]
        """
        wav, _ = librosa.load(wav_path, sr=16000)
        inputs = self.processor(
            torch.tensor(wav), sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        hidden = self.model(**inputs.to(torch.float16)).last_hidden_state  # [1,T,768]
        return hidden.mean(dim=1).squeeze(0)  # [768]
