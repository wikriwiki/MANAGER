#!/usr/bin/env python3
"""
list_embedding_status.py
────────────────────────────────────────────────────────────────────────────
캐시(.pt) 존재 여부를 기준으로 영상별로
  1) 비디오+오디오 임베딩 모두 완료
  2) 비디오 임베딩만 완료
  3) 오디오 임베딩만 완료
리스트를 CSV로 출력.

사용법
-----
python list_embedding_status.py \
    --cache ./cache \
    --db data/speech_segments.db \
    --out_dir data
"""

import hashlib, sqlite3, csv, argparse
from pathlib import Path

def pt_hash(key: str) -> str:
    """embed_utils.py 와 동일한 md5(key) + '.pt' 규칙"""
    return hashlib.md5(key.encode()).hexdigest() + ".pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",   default="./cache",
                    help="임베딩 캐시 디렉토리 경로")
    ap.add_argument("--db",      default="data/speech_segments.db",
                    help="SQLite DB 파일 경로")
    ap.add_argument("--out_dir", default="data",
                    help="출력 CSV 파일을 저장할 디렉토리")
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    if not cache_dir.exists():
        raise SystemExit(f"❌ 캐시 디렉토리를 찾을 수 없습니다: {cache_dir}")

    # DB에서 video_id ↔ segment_id 매핑 읽기
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT video_id, segment_id FROM speech_segments"
    ).fetchall()
    conn.close()

    video2segs = {}
    for vid, sid in rows:
        video2segs.setdefault(vid, []).append(sid)

    both_complete = []
    video_only    = []
    audio_only    = []

    for vid, segs in video2segs.items():
        total = len(segs)
        vid_cnt = sum((cache_dir/pt_hash(f"vid::{sid}")).exists()
                      for sid in segs)
        aud_cnt = sum((cache_dir/pt_hash(f"aud::{sid}")).exists()
                      for sid in segs)

        if vid_cnt == total and aud_cnt == total:
            both_complete.append(vid)
        elif vid_cnt == total and aud_cnt < total:
            video_only.append(vid)
        elif aud_cnt == total and vid_cnt < total:
            audio_only.append(vid)
        # 그 외(절반 이하 임베딩)는 무시

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(filename: str, vids: list[str]):
        with (out_dir/filename).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id"])
            for v in sorted(vids):
                w.writerow([v])

    write_csv("both_complete_videos.csv", both_complete)
    write_csv("video_only_videos.csv",    video_only)
    write_csv("audio_only_videos.csv",    audio_only)

    print(f"✅ 완료: both={len(both_complete)}, "
          f"video_only={len(video_only)}, audio_only={len(audio_only)}")

if __name__ == "__main__":
    main()
