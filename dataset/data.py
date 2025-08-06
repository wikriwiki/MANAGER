# data/person_dataset.py
import os, json, hashlib, random, sqlite3, gc
from pathlib import Path
from typing import Dict, List, Tuple, Set

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from PIL import Image  # video 프레임 로딩용

from models.encoder import (
    TextFeatureExtractor,
    VideoFeatureExtractor,
    AudioFeatureExtractor,
)
from models.graph_builder import GraphBuilder
from models.graph_utils import merge_graph


class VideoPersonDataset(Dataset):
    """
    영상 인물 페어 → {graph, person, label} 을 반환하는 Dataset.

    Args
    ----
    db_path      : SQLite DB (video_metadata, speech_segments 포함)
    split        : "train" | "val" | "test"  (무작위 7:1:2 분할)
    cache_dir    : 임베딩 캐시(.pt) 저장 폴더
    video_root   : frames/<video_id>/<segment_id>/*.jpg
    audio_root   : wav/<video_id>/<segment_id>.wav
    merge_dialog : True → 같은 video_id 내 발화 그래프를 merge_graph 로 연결
    max_samples  : None 또는 사용할 최대 샘플 수
    filter_ids   : 사용할 video_id들의 집합 (None이면 전체 사용)
    seed         : random shuffle seed
    """

    def __init__(
        self,
        db_path: str,
        split: str,
        cache_dir: str,
        video_root: str,
        audio_root: str,
        merge_dialog: bool = True,
        max_samples: int | None = None,
        filter_ids: Set[str] | None = None,
        seed: int = 42,
    ):
        super().__init__()

        # ── DB 연결 ───────────────────────────────────────────
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # ── 경로 및 옵션 ──────────────────────────────────────
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.video_root = Path(video_root)
        self.audio_root = Path(audio_root)
        self.merge_dialog = merge_dialog

        # ── Feature Encoders ─────────────────────────────────
        self.text_enc = TextFeatureExtractor()
        self.video_enc = VideoFeatureExtractor()
        self.audio_enc = AudioFeatureExtractor()

        # ── 1) 무작위 split 7:1:2 ────────────────────────────
        videos = [
            (row["video_id"], row["published_date"])
            for row in self.conn.execute(
                "SELECT video_id, published_date FROM video_metadata"
            )
        ]
        # filter_ids로 지정된 영상만 선택
        if filter_ids is not None:
            videos = [(vid, pub) for vid, pub in videos if str(vid) in filter_ids]
        random.Random(seed).shuffle(videos)

        n = len(videos)
        n_train, n_val = int(0.7 * n), int(0.8 * n)
        if split == "train":
            sel_videos = videos[:n_train]
        elif split == "val":
            sel_videos = videos[n_train:n_val]
        else:
            sel_videos = videos[n_val:]

        if max_samples is not None:
            sel_videos = sel_videos[:max_samples]

        # ── 2) (video, person, label, pub_date) 인덱스 구축 ──
        self.index: List[Tuple[str, str, int, str]] = []
        for vid, pub in sel_videos:
            persons_json = self.conn.execute(
                "SELECT persons_found FROM video_metadata WHERE video_id=?",
                (vid,)
            ).fetchone()["persons_found"]
            for name, lab in json.loads(persons_json).items():
                self.index.append((vid, name, int(lab), pub))

        # ── 3) speech_segments 캐싱 ─────────────────────────
        self.seg_cache: Dict[str, List[sqlite3.Row]] = {
            vid: self.conn.execute(
                "SELECT * FROM speech_segments WHERE video_id=? ORDER BY start_time",
                (vid,)
            ).fetchall()
            for vid, _ in sel_videos
        }

    # ── helper: 캐시 경로 ────────────────────────────────────
    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.pt"

    def _text_embed(self, script: str | None) -> Tuple[torch.Tensor, Dict]:
        # --- 디버깅 코드 시작 ---
        print(f"DEBUG: Received script with type: {type(script)}, value: '{script}'")
        if not isinstance(script, str):
            print("DEBUG: Non-string type detected. Converting to empty string.")
            script = ""
        # --- 디버깅 코드 끝 ---
        try:
            p = self._cache_path("txt::" + script)
            if p.exists():
                emb, meta = torch.load(p, map_location="cpu")
                return emb, meta

            emb, meta = self.text_enc.encode(
                script, knowledge_triples=[], anchor_entities=[]
            )
            emb = emb.cpu()
            torch.save((emb, meta), p)
            return emb, meta
        except ValueError as e:
            print("\n" + "=" * 50)
            print("!!!!!! CRITICAL ERROR CATCH !!!!!!")
            print("ValueError가 아래 script 값으로 인해 발생했습니다:")
            print(f"  - Type: {type(script)}")
            print(f"  - Value: {script}")
            print(f"  - Original Error: {e}")
            print("=" * 50 + "\n")
            raise

    def _video_embed(self, seg: sqlite3.Row) -> torch.Tensor:
        p = self._cache_path("vid::" + seg["segment_id"])
        if p.exists():
            return torch.load(p, map_location="cpu")
        frames_dir = self.video_root / seg["video_id"] / seg["segment_id"]
        if not frames_dir.exists():
            vec = torch.zeros(1, 768)
        else:
            imgs = [Image.open(f) for f in sorted(frames_dir.glob("*.jpg"))]
            vec = self.video_enc.encode_clip(imgs).unsqueeze(0).cpu()
        torch.save(vec, p)
        return vec

    def _audio_embed(self, seg: sqlite3.Row) -> torch.Tensor:
        p = self._cache_path("aud::" + seg["segment_id"])
        if p.exists():
            return torch.load(p, map_location="cpu")
        wav = self.audio_root / seg["video_id"] / f"{seg['segment_id']}.wav"
        if not wav.exists():
            vec = torch.zeros(1, 768)
        else:
            vec = self.audio_enc.encode_clip(str(wav)).unsqueeze(0).cpu()
        torch.save(vec, p)
        return vec

    # ── Dataset 인터페이스 ──────────────────────────────────
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        video_id, person, label, pub_date = self.index[idx]
        seg_rows = self.seg_cache[video_id]

        # 발화별 그래프 생성
        graphs: List[Data] = []
        time_iso = f"{pub_date}"
        
        gb = None
        try:
            gb = GraphBuilder(time_iso=time_iso)
            for seg in seg_rows:
                t_emb, _ = self._text_embed(seg["script"])
                v_emb = self._video_embed(seg)
                a_emb = self._audio_embed(seg)
                graphs.append(gb.build(seg["script"], v_emb, a_emb))
                
        finally:
            # 반드시 메모리 해제
            if gb is not None:
                if hasattr(gb, 'text_enc'):
                    gb.text_enc.cleanup()
                del gb
                
            # 추가 정리
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Sample {idx} 처리 완료, GPU 메모리 해제함")

        # 발화 병합 여부
        graph_final = graphs[0]
        if self.merge_dialog and len(graphs) > 1:
            for g_next in graphs[1:]:
                graph_final = merge_graph(graph_final, g_next)

        return {
            "graph": graph_final,
            "person": person,
            "label": label,
            "video_id": video_id,
        }


# 사용예시
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_ds = VideoPersonDataset(
        db_path="data/speech_segments.db",
        split="train",
        cache_dir="cache/",
        video_root="frames/",
        audio_root="wav/",
        merge_dialog=True,
        max_samples=5000,
        filter_ids=None,
        seed=123,
    )
    loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    for sample in loader:
        g = sample["graph"]
        p = sample["person"]
        lbl = sample["label"]
        break