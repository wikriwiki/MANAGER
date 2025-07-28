#!/usr/bin/env python3
"""Parallel segment extractor (wav + frames) with resume capability

Usage:
    python preprocess_segments_parallel.py --workers 10

It reads `data/speech_segments.db`, extracts
  • wav clips  -> /mnt/third_ssd/data/wav/<video_id>/<segment_id>.wav
  • frames     -> /mnt/third_ssd/data/frames/<video_id>/<segment_id>/fXXXXXX.jpg
skipping any segment whose output files already exist.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── 설정 --------------------------------------------------------------
DB_PATH        = "data/speech_segments.db"
RAW_VIDEO_DIR  = Path("/mnt/shares/videos")                  # <id>.mp4
WAV_OUT_ROOT   = Path("/mnt/third_ssd/data/wav")            # wav/<id>/<seg>.wav
FRAME_OUT_ROOT = Path("/mnt/third_ssd/data/frames")         # frames/<id>/<seg>/fXXXXXX.jpg
FPS            = 1                                            # 1 fps

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
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-ss", str(start), "-to", str(end),
        "-i", str(video_file),
        "-ar", "16000", "-ac", "1",
        str(out_path),
    ])


def extract_frames(video_file: Path, start: float, end: float, out_dir: Path, fps: int = FPS):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-ss", str(start), "-to", str(end),
        "-i", str(video_file),
        "-vf", f"fps={fps}",
        str(out_dir / "f%06d.jpg"),
    ])

# ── worker func --------------------------------------------------------

def process_segment(task: Tuple[str, str, float, float]) -> Tuple[str, bool, str | None]:
    """Return (segment_id, success, error_msg)."""
    video_id, segment_id, start_t, end_t = task
    video_file = RAW_VIDEO_DIR / f"{video_id}.mp4"
    if not video_file.exists():
        return segment_id, False, f"missing video {video_file}"

    wav_path   = WAV_OUT_ROOT / video_id / f"{segment_id}.wav"
    frame_dir  = FRAME_OUT_ROOT / video_id / segment_id

    try:
        # wav
        if not wav_path.exists():
            extract_wav(video_file, start_t, end_t, wav_path)
        # frames (check at least 1 file exists)
        if not frame_dir.exists() or not any(frame_dir.iterdir()):
            extract_frames(video_file, start_t, end_t, frame_dir)
        return segment_id, True, None
    except Exception as e:
        return segment_id, False, str(e)

# ── main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10, help="number of parallel ffmpeg workers")
    args = parser.parse_args()

    # 1) Load segment list once (read-only)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT segment_id, video_id, start_time, end_time FROM speech_segments"
    ).fetchall()
    conn.close()

    tasks = [(r["video_id"], r["segment_id"], r["start_time"], r["end_time"]) for r in rows]

    # Allow resume: filter out those already done
    def _already_done(t):
        v, s, *_ = t
        return (WAV_OUT_ROOT / v / f"{s}.wav").exists() and \
               (FRAME_OUT_ROOT / v / s).exists() and any((FRAME_OUT_ROOT / v / s).iterdir())

    tasks = [t for t in tasks if not _already_done(t)]

    if not tasks:
        print("All segments already processed.")
        return

    # 2) Parallel extraction
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_segment, t): t[1] for t in tasks}
        pbar = tqdm(total=len(futs), ncols=88, desc="extracting")
        for fut in as_completed(futs):
            seg_id = futs[fut]
            ok, err = False, "?"
            try:
                _, ok, err = fut.result()
            except Exception as exc:
                err = str(exc)
            if not ok:
                pbar.write(f"[fail] {seg_id}: {err}")
            pbar.update(1)
        pbar.close()

    print("DONE.")


if __name__ == "__main__":
    main()
