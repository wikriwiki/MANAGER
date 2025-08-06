#!/usr/bin/env python3
"""
list_ready_videos.py
────────────────────────────────────────────────────────────────────────────
캐시(.pt) 존재 여부를 기준으로
  • video_id
  • segment_count   (영상에 포함된 세그먼트 수)
를 ready_videos.csv 로 저장.

Usage
-----
python list_ready_videos.py --cache ./cache   [--db data/speech_segments.db]
"""

from __future__ import annotations
import hashlib, sqlite3, csv, argparse
from pathlib import Path
from tqdm import tqdm

def pt_hash(key: str) -> str:
    """embed_utils 와 동일한 md5 해시 규칙."""
    return hashlib.md5(key.encode()).hexdigest() + ".pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--db",    default="data/speech_segments.db")
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    if not cache_dir.exists():
        raise SystemExit(f"❌ cache dir {cache_dir} not found")

    # ── DB 로드: video_id ↔ [segment_id…] 매핑 ──────────────────────
    conn = sqlite3.connect(args.db); conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT video_id, segment_id FROM speech_segments"
    ).fetchall(); conn.close()

    video2segs: dict[str, list[str]] = {}
    for r in rows:
        video2segs.setdefault(r["video_id"], []).append(r["segment_id"])

    ready_videos: list[tuple[str, int]] = []

    print(f"▶ Checking {len(video2segs):,} videos …")
    for vid, segs in tqdm(video2segs.items(), ncols=95):
        all_ok = True
        for sid in segs:
            vid_pt = cache_dir / pt_hash(f"vid::{sid}")
            aud_pt = cache_dir / pt_hash(f"aud::{sid}")
            if not (vid_pt.exists() and aud_pt.exists()):
                all_ok = False
                break
        if all_ok:
            ready_videos.append((vid, len(segs)))

    out_csv = Path("data") / "ready_videos.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "segment_count"])
        w.writerows(sorted(ready_videos))

    print(f"✅ {len(ready_videos):,} / {len(video2segs):,} videos ready.")
    print(f"→ saved to {out_csv}")

if __name__ == "__main__":
    main()
