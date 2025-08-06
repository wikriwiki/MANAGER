# MANAGER/embed_utils_audio.py
"""
오디오(.wav) → 768-차원 벡터만 처리한다.
· Video/Text 모듈 import 안 하므로 torch_geometric 불필요
· 실패(wav 없음 등)는 cache_dir/fail_segments_audio.csv 로 기록
"""

from __future__ import annotations
import hashlib, csv, torch
from pathlib import Path
from models.encoder import AudioFeatureExtractor   # 기존 encoder 재사용

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUD = AudioFeatureExtractor()

def _h(key: str) -> str:                         # md5 + .pt
    return hashlib.md5(key.encode()).hexdigest() + ".pt"

def _log(cache: Path, seg_id: str, reason: str):
    f = cache / "fail_segments_audio.csv"
    header_needed = not f.exists()
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("a", newline="") as fp:
        w = csv.writer(fp)
        if header_needed:
            w.writerow(["segment_id", "reason"])
        w.writerow([seg_id, reason])

@torch.no_grad()
def embed_audio_only(wav_path: Path, cache: Path, seg_id: str):
    out = cache / _h("aud::" + seg_id)
    if out.exists():
        return                                  # 이미 있음
    if not wav_path.exists():
        _log(cache, seg_id, "wav_missing")
        return
    emb = AUD.encode_clip(str(wav_path)).unsqueeze(0)   # [1,768]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb.half().cpu(), out)          # fp16 & CPU 저장
