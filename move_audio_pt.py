#!/usr/bin/env python3
"""
move_old_audio_pt.py
────────────────────────────────────────────────────────────────
speech_segments.db의 segment_id → aud::<segment_id> 해시를 계산해
<cache>  아래의  *.pt 중 **오디오 임베딩**만  <cache>/error 로 옮긴다.

Usage
-----
python move_old_audio_pt.py --cache ./cache --db data/speech_segments.db
"""

from __future__ import annotations
import hashlib, sqlite3, shutil, argparse
from pathlib import Path
from tqdm import tqdm

def hash_aud(seg_id: str) -> str:
    return hashlib.md5(f"aud::{seg_id}".encode()).hexdigest() + ".pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--db",    default="data/speech_segments.db")
    args = ap.parse_args()

    cache = Path(args.cache)
    error_dir = cache / "error"
    error_dir.mkdir(exist_ok=True)

    # ── DB에서 모든 segment_id 읽기 ─────────────────────────────
    conn = sqlite3.connect(args.db); conn.row_factory = sqlite3.Row
    seg_ids = [r["segment_id"] for r in
               conn.execute("SELECT segment_id FROM speech_segments")]
    conn.close()

    moved = 0
    for sid in tqdm(seg_ids, desc="Moving old audio pt"):
        pt = cache / hash_aud(sid)
        if pt.exists():
            shutil.move(pt, error_dir / pt.name)
            moved += 1

    print(f"✅  moved {moved:,} audio .pt files → {error_dir}")

if __name__ == "__main__":
    main()
