#!/usr/bin/env python3
"""
audio_embed_only_v3.py
────────────────────────────────────────────────────────────────────────────
📌 목적
  1) 세그먼트 wav 없으면 ffmpeg 로 추출
  2) HuBERT-base 로 768-d 오디오 임베딩 생성
  3) 물리 GPU × SLOTS_PER_GPU 개의 프로세스로 동시 처리
     (카드당 HuBERT 최대 3개만 올라가도록 메모리 제한)

실행 예
──────
# 0,1,2 번 GPU 세 장만 사용
CUDA_VISIBLE_DEVICES=0,1,2,3 python audio_embed_only.py \
    --cache ./cache --workers 16
"""

from __future__ import annotations
import os, sqlite3, subprocess, argparse, multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from tqdm import tqdm

# ────────── 경로 및 하드코딩 설정 ───────────────────────────────────────
DB_PATH   = "data/speech_segments.db"
RAW_VDIR  = Path("/mnt/shares/videos")
WAV_ROOT  = Path("/mnt/third_ssd/data/wav")
CACHE_DIR = Path("./cache")

SLOTS_PER_GPU   =  5           # 카드당 HuBERT 인스턴스 수
THREADS_PER_PROC = 4           # 한 프로세스당 CPU 작업 스레드 수
FPS              = 1           # 프레임 추출 fps (wav 생성만 할 때는 사용 안 함)
# ────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════
# Helper: ffmpeg 로 wav 추출
# ════════════════════════════════════════════════════════════════════════
def ensure_wav(video_id: str, seg_id: str, st: float, et: float) -> Path:
    vid_file = RAW_VDIR / f"{video_id}.mp4"
    wav_path = WAV_ROOT / video_id / f"{seg_id}.wav"
    if wav_path.exists():
        return wav_path

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", str(st), "-to", str(et), "-i", str(vid_file),
        "-ar", "16000", "-ac", "1", str(wav_path)
    ]
    subprocess.run(cmd, check=True)
    return wav_path


# ════════════════════════════════════════════════════════════════════════
# embed_utils_audio 전용 함수 – GPU 프로세스에서만 import
# ════════════════════════════════════════════════════════════════════════
def embed_audio_only(wav_path: Path, cache_dir: Path, seg_id: str):
    from embed_utils_audio import embed_audio_only  # <- 경량 모듈
    embed_audio_only(wav_path, cache_dir, seg_id)


# ════════════════════════════════════════════════════════════════════════
# 비디오별 오디오 임베딩 진행률 계산
# ════════════════════════════════════════════════════════════════════════
def audio_ready_ratio(seg_ids: List[str],
                      cache_dir: Path) -> float:
    total = len(seg_ids)
    have  = sum(
        (cache_dir / f"{seg_id}.pt").exists()
        for seg_id in seg_ids
    )
    return have / total if total else 0.0


# ════════════════════════════════════════════════════════════════════════
# GPU 슬롯별 워커
# ════════════════════════════════════════════════════════════════════════
def slot_worker(gpu_id: int,
                slot_idx: int,
                videos: List[str],
                seg_rows: Dict[str, List[Tuple]],
                cpu_threads: int,
                cache_dir: Path):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    desc = f"GPU{gpu_id}-S{slot_idx}"
    fails = 0

    with ProcessPoolExecutor(max_workers=cpu_threads) as ex:
        for vid in videos:
            rows = seg_rows[vid]
            futs = []

            for seg_id, st, et in rows:
                wav_path = ensure_wav(vid, seg_id, st, et)
                futs.append(
                    ex.submit(embed_audio_only, wav_path, cache_dir, seg_id)
                )

            pbar = tqdm(as_completed(futs),
                         total=len(futs),
                         ncols=90,
                         desc=f"{desc} {vid[:10]}")
            for fut in pbar:
                try:
                    fut.result()
                except Exception as e:
                    fails += 1
                    pbar.write(f"⚠ {e}")

    print(f"[{desc}] done. fails={fails}")


# ════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",   default=str(CACHE_DIR))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    cache_dir.mkdir(exist_ok=True)

    # ── DB 로드: video_id → [(seg_id, st, et)…] ─────────────────────────
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT video_id, segment_id, start_time, end_time "
        "FROM speech_segments"
    )
    video2rows: Dict[str, List[Tuple[str, float, float]]] = {}
    for r in cur.fetchall():
        video2rows.setdefault(r["video_id"], []).append(
            (r["segment_id"], r["start_time"], r["end_time"]))
    conn.close()

    # ── 진척률 높은 순으로 정렬 ─────────────────────────────────────────
    sorted_videos = sorted(video2rows.keys(),
                           key=lambda v: audio_ready_ratio(
                               [s[0] for s in video2rows[v]], cache_dir),
                           reverse=True)

    # ── 슬롯 분배 ───────────────────────────────────────────────────────
    total_slots = torch.cuda.device_count() * SLOTS_PER_GPU
    shards = [sorted_videos[i::total_slots] for i in range(total_slots)]

    print(f"Total videos: {len(sorted_videos)} ; GPUs: "
          f"{torch.cuda.device_count()} ; Slots: {total_slots}")

    cpu_per_proc = max(1, args.workers // total_slots)
    procs = []
    slot_id = 0
    for gpu in range(torch.cuda.device_count()):
        for local in range(SLOTS_PER_GPU):
            vids = shards[slot_id]
            p = mp.Process(target=slot_worker,
                           args=(gpu, local, vids,
                                 video2rows, cpu_per_proc, cache_dir))
            p.start(); procs.append(p)
            print(f"Spawned GPU{gpu}-S{local} : {len(vids)} videos "
                  f"(CPU {cpu_per_proc})")
            slot_id += 1

    for p in procs: p.join()
    print("✅ All slots finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
