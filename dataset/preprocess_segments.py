#!/usr/bin/env python3
"""Parallel segment extractor (wav + frames) with resume capability

Usage:
    python preprocess_segments.py

It reads `data/speech_segments.db`, extracts
  • wav clips  -> /mnt/third_ssd/data/wav/<video_id>/<segment_id>.wav
  • frames     -> /mnt/third_ssd/data/frames/<video_id>/<segment_id>/fXXXXXX.jpg
skipping any segment whose output files already exist.
"""
from __future__ import annotations

import os
import argparse
import sqlite3
import subprocess
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── 설정 (하드코딩 경로 유지) -------------------------------------------
DB_PATH        = "data/speech_segments.db"
RAW_VIDEO_DIR  = Path("/mnt/shares/videos")
WAV_OUT_ROOT   = Path("/mnt/third_ssd/data/wav")
FRAME_OUT_ROOT = Path("/mnt/third_ssd/data/frames")
FPS            = 1

# Ensure root dirs exist ------------------------------------------------
WAV_OUT_ROOT.mkdir(parents=True, exist_ok=True)
FRAME_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── ffmpeg helpers -----------------------------------------------------

def run_ffmpeg(cmd: list[str]):
    """Run ffmpeg, raising RuntimeError on failure (stderr snippet)."""
    r = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="ignore")[:400])


def extract_wav(video_file: Path, start: float, end: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", str(video_file),
        "-ar", "16000", "-ac", "1",
        str(out_path),
    ])


def extract_frames(video_file: Path, start: float, end: float, out_dir: Path, fps: int = FPS):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", str(video_file),
        "-vf", f"fps={fps}",
        str(out_dir / "f%06d.jpg"),
    ])

# ── worker func --------------------------------------------------------

def process_segment(task: Tuple[str, str, float, float]) -> Tuple[str, bool, str | None]:
    """
    Processes a single segment. It checks for existing files to allow for
    resuming, and returns the result.
    Return: (segment_id, success, error_msg).
    """
    video_id, segment_id, start_t, end_t = task
    video_file = RAW_VIDEO_DIR / f"{video_id}.mp4"
    if not video_file.exists():
        return segment_id, False, f"missing video {video_file}"

    wav_path   = WAV_OUT_ROOT / video_id / f"{segment_id}.wav"
    frame_dir  = FRAME_OUT_ROOT / video_id / segment_id

    try:
        # Resume Logic: Skip if wav exists and frame dir is not empty.
        # This check is now robustly handled per-worker.
        wav_exists = wav_path.exists()
        frames_exist = frame_dir.exists() and any(frame_dir.iterdir())

        if wav_exists and frames_exist:
            return segment_id, True, "skipped" # Return a specific message for already done tasks

        # Extract wav if it doesn't exist
        if not wav_exists:
            extract_wav(video_file, start_t, end_t, wav_path)
            
        # Extract frames if the directory is empty
        if not frames_exist:
            extract_frames(video_file, start_t, end_t, frame_dir)
            
        return segment_id, True, None
    except Exception as e:
        return segment_id, False, str(e)

# ── main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel segment extractor with resume capability.")
    # ## 1. 워커 수 자동 설정으로 변경 ##
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel ffmpeg workers. Defaults to the number of CPU cores.")
    args = parser.parse_args()

    # Load segment list once (read-only)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT segment_id, video_id, start_time, end_time FROM speech_segments"
    ).fetchall()
    conn.close()

    tasks = [(r["video_id"], r["segment_id"], r["start_time"], r["end_time"]) for r in rows]

    if not tasks:
        print("No segments found in the database.")
        return

    # ## 2. 재시작 기능 개선: 메인 스레드에서 느린 사전 필터링 제거 ##
    # 이제 각 워커가 개별적으로 처리 여부를 확인하므로 시작이 빠릅니다.
    print(f"Found {len(tasks)} segments. Starting parallel extraction with {args.workers} workers.")

    processed_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_segment, t): t[1] for t in tasks}
        pbar = tqdm(total=len(futs), ncols=100, desc="Extracting")
        for fut in as_completed(futs):
            seg_id = futs[fut]
            try:
                _, ok, err = fut.result()
                if err == "skipped":
                    processed_count += 1
                elif not ok:
                    pbar.write(f"[FAIL] {seg_id}: {err}")
            except Exception as exc:
                pbar.write(f"[ERROR] {seg_id}: {exc}")
            pbar.update(1)
        pbar.close()

    print(f"\nDONE. Skipped {processed_count} already processed segments.")


if __name__ == "__main__":
    main()