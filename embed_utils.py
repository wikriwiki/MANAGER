"""
embed_utils.py
────────────────────────────────────────────────────────────────────────────
텍스트 / 비디오 / 오디오 세그먼트를 768-차원 임베딩으로 변환하고
cache_dir/xxxxxxxx.pt 에 저장해 주는 유틸리티 모듈.

추가 기능
────────
· 프레임이 없거나 wav 가 없어서 **zero-vector** 로 떨어지면 저장하지 않고
  cache_dir/fail_segments.csv 에
      segment_id, modality, reason
  형식으로 한 줄씩 누적 기록한다.
"""

from __future__ import annotations
import hashlib, csv, torch, os
from pathlib import Path
from typing import List
from PIL import Image

# ───────── 모델 로더 ────────────────────────────────────────────
from models.encoder import (
    VideoFeatureExtractor,
    AudioFeatureExtractor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VID = VideoFeatureExtractor()
AUD = AudioFeatureExtractor()

# ───────── 내부 헬퍼 ────────────────────────────────────────────
def _hash(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()

def _save_tensor(t: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t.half().cpu(), path)        # fp16 + CPU 저장

def _log_fail(cache_dir: Path, seg_id: str, modality: str, reason: str):
    """
    실패 세그먼트를 cache_dir/fail_segments.csv 에 append.
    헤더가 없으면 자동으로 작성.
    """
    csv_path = cache_dir / "fail_segments.csv"
    need_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["segment_id", "modality", "reason"])
        w.writerow([seg_id, modality, reason])

# ───────── 공개 API ────────────────────────────────────────────
@torch.no_grad()
def embed_video(frames_dir: Path, cache_dir: Path, seg_id: str):
    """
    frames_dir 하위 *.jpg 로부터 1×768 벡터를 추출.
    • 프레임이 없으면: CSV 로그만 남기고 .pt 저장 안 함.
    """
    key  = "vid::" + seg_id
    path = cache_dir / f"{_hash(key)}.pt"
    if path.exists():
        return  # 이미 임베딩 완료

    imgs: List[Image.Image] = [
        Image.open(p).convert("RGB")
        for p in sorted(frames_dir.glob("*.jpg"))
    ]

    if not imgs:
        _log_fail(cache_dir, seg_id, "video", "no_frames")
        return

    emb = VID.encode_clip(imgs).unsqueeze(0)   # [1,768]
    _save_tensor(emb, path)

@torch.no_grad()
def embed_audio(wav_path: Path, cache_dir: Path, seg_id: str):
    """
    wav_path 로부터 1×768 벡터 추출.
    • wav 가 없으면: CSV 로그만 남기고 .pt 저장 안 함.
    """
    key  = "aud::" + seg_id
    path = cache_dir / f"{_hash(key)}.pt"
    if path.exists():
        return

    if not wav_path.exists():
        _log_fail(cache_dir, seg_id, "audio", "wav_missing")
        return

    emb = AUD.encode_clip(str(wav_path)).unsqueeze(0)  # [1,768]
    _save_tensor(emb, path)

# ───────── CLI 테스트용 ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", help="프레임 디렉터리")
    ap.add_argument("--wav",    help="wav 파일 경로")
    ap.add_argument("--cache",  required=True)
    ap.add_argument("--seg",    default="demo_seg")
    args = ap.parse_args()

    cdir = Path(args.cache)
    if args.frames:
        embed_video(Path(args.frames), cdir, args.seg)
    if args.wav:
        embed_audio(Path(args.wav), cdir, args.seg)
    print("done.")
