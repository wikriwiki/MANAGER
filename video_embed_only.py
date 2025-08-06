#!/usr/bin/env python3
"""
video_embed_only.py
────────────────────────────────────────────────────────────────────────────
  • 세그먼트 프레임 없으면 ffmpeg 로 추출
  • BEiT-base + TemporalAttention 으로 768-d 비디오 임베딩
  • GPU 갯수만큼 프로세스 생성, 영상(video_id) 단위로 순차 처리
사용 예)
CUDA_VISIBLE_DEVICES=0,1,2,3 python video_embed_only.py \
    --cache ./cache --workers 64
"""
from __future__ import annotations
import os, sqlite3, subprocess, argparse, multiprocessing as mp, logging, warnings
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch                                         # 메인 프로세스

# ───── 경로/파라미터 ────────────────────────────────────────────────
DB_PATH        = "data/speech_segments.db"
RAW_VIDEO_DIR  = Path("/mnt/shares/videos")
FRAME_OUT_ROOT = Path("/mnt/third_ssd/data/frames")
FPS            = 1

# ───── 로그/워닝 최소화 ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"]            = "error"
for n in ["transformers","urllib3","accelerate","huggingface_hub",
          "numba","requests","PIL"]:
    logging.getLogger(n).setLevel(logging.ERROR)

# ---------------- ffmpeg ------------------------------------------
def run_ffmpeg(cmd: List[str]):
    subprocess.run(cmd, check=True,
                   stderr=subprocess.PIPE,
                   stdout=subprocess.DEVNULL)

# --------------- 세그먼트 작업 (CPU) -------------------------------
def process_segment(seg: Tuple[str,str,float,float],
                    cache_dir: Path) -> str|None:
    """
    seg = (segment_id, video_id, start, end)
    성공 ➜ None, 실패 ➜ str(메시지)
    """
    seg_id, vid, st, et = seg
    video_file = RAW_VIDEO_DIR / f"{vid}.mp4"
    if not video_file.exists():
        return f"{seg_id}: video_missing"

    frame_dir = FRAME_OUT_ROOT / vid / seg_id
    if not (frame_dir.exists() and any(frame_dir.iterdir())):
        frame_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_ffmpeg([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-ss", str(st), "-to", str(et), "-i", str(video_file),
                "-vf", f"fps={FPS}", str(frame_dir/"f%06d.jpg")
            ])
        except subprocess.CalledProcessError as e:
            return f"{seg_id}: ffmpeg_fail {e}"

    # 임베딩 (GPU 모델은 상위 프로세스에 이미 올라가 있음)
    from embed_utils import embed_video
    try:
        embed_video(frame_dir, cache_dir, seg_id)
    except Exception as e:
        return f"{seg_id}: embed_fail {e}"
    return None

# --------------- GPU 프로세스 -------------------------------------
def gpu_worker(gid:int,
               rows_slice:List[Tuple[str,str,float,float]],
               cache_dir:Path,
               cpu_threads:int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    from embed_utils import embed_video   # BEiT 로드 (1회)

    # video_id → [세그…] 그룹핑
    vids:Dict[str, List[Tuple]] = {}
    for seg in rows_slice:
        vids.setdefault(seg[1], []).append(seg)

    print(f"[GPU{gid}] videos={len(vids)}  segs={len(rows_slice)}")

    for vid_idx,(vid,segs) in enumerate(vids.items(),1):
        desc = f"GPU{gid} {vid} ({vid_idx}/{len(vids)})"
        fails = 0
        with ProcessPoolExecutor(max_workers=cpu_threads) as ex:
            futs = [ex.submit(process_segment,s,cache_dir) for s in segs]
            for fut in tqdm(as_completed(futs),
                            total=len(futs),desc=desc,ncols=95,leave=False):
                err = fut.result()
                if err:
                    fails += 1
                    tqdm.write("⚠ "+err)
        ok = len(segs)-fails
        print(f"[GPU{gid}] {vid}  {ok}/{len(segs)} done  fails={fails}")

    print(f"[GPU{gid}] finished all videos")

# ----------------------------- 메인 -------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--cache", default="./cache")
    pa.add_argument("--workers",type=int,default=os.cpu_count())
    args = pa.parse_args()
    cache_dir = Path(args.cache).resolve(); cache_dir.mkdir(exist_ok=True)

    # DB 읽기
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = [(r["segment_id"], r["video_id"],
             r["start_time"], r["end_time"])
            for r in conn.execute(
            "SELECT segment_id,video_id,start_time,end_time FROM speech_segments")]
    conn.close()
    if not rows: raise SystemExit("❌ DB가 비어 있습니다")

    gpus = torch.cuda.device_count()
    rows_per_gpu = (len(rows)+gpus-1)//gpus
    cpu_per_gpu  = max(1, args.workers//gpus)

    procs=[]
    for gid in range(gpus):
        subset = rows[gid*rows_per_gpu:(gid+1)*rows_per_gpu]
        p = mp.Process(target=gpu_worker,
                       args=(gid,subset,cache_dir,cpu_per_gpu))
        p.start(); procs.append(p)
        print(f"Spawn GPU{gid}: segs={len(subset)} cpu={cpu_per_gpu}")

    for p in procs: p.join()
    print("🎉  video embedding completed")

if __name__ == "__main__":
    mp.set_start_method("spawn",force=True)
    main()
