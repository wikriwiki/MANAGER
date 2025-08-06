#!/usr/bin/env python3
# match_pt_to_segments.py
"""
cache/*.pt ↔ (video_id, segment_id, modality) 매칭
결과:
  • matched.csv     : 파일이 어떤 seg·modality 규칙으로 맞았는지
  • unmapped_pt.csv : 어느 규칙으로도 못 맞춘 캐시 파일
규칙은 patterns 리스트에 자유롭게 추가/수정하세요.
"""

import hashlib, sqlite3, csv, argparse
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

# *** 필요하면 여기에 규칙을 계속 추가하세요 ***
def patterns(video_id: str, seg_id: str):
    yield "vid", md5(f"vid::{seg_id}")          # 현재 규칙 (video)
    yield "aud", md5(f"aud::{seg_id}")          # 현재 규칙 (audio)
    yield "old_vid", md5(f"vid::{video_id}::{seg_id}")   # 옛 규칙 예
    yield "old_aud", md5(f"aud::{video_id}::{seg_id}")   # 옛 규칙 예
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--db",    default="data/speech_segments.db")
    args = ap.parse_args()

    cache_paths = {p.name: p for p in Path(args.cache).glob("*.pt")}
    print(f"PT files in cache : {len(cache_paths):,}")

    # DB 로드
    conn = sqlite3.connect(args.db); conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT video_id, segment_id FROM speech_segments"
    ).fetchall(); conn.close()
    print(f"Segments in DB    : {len(rows):,}")

    matched, unmatched = [], set(cache_paths)   # 파일명(32+3) 집합
    for r in tqdm(rows, ncols=90):
        vid, sid = r["video_id"], r["segment_id"]
        for tag, h in patterns(vid, sid):
            fn = f"{h}.pt"
            if fn in cache_paths:
                matched.append((fn, vid, sid, tag))
                unmatched.discard(fn)

    # 결과 저장
    Path("data").mkdir(exist_ok=True)
    with open("data/matched.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "video_id", "segment_id", "rule"])
        w.writerows(matched)
    with open("data/unmapped_pt.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file"])
        w.writerows([[u] for u in sorted(unmatched)])

    print("\n● matched  :", len(matched))
    print("● unmapped :", len(unmatched))
    print("→ data/matched.csv , data/unmapped_pt.csv 작성 완료")

if __name__ == "__main__":
    main()
