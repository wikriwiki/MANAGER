# #!/usr/bin/env python3
# """
# (1) 세그먼트 추출(wav + frames)  +  (2) 임베딩 캐시 생성
# GPU 4개를 전용 프로세스 4개로 분산해서 실행.

# Usage
# -----
# python extract_and_embed.py \
#        --cache ./cache \
#        --workers 64          # CPU 워커 총합 (GPU당 자동 ¼)
# """

# from __future__ import annotations
# import os, sqlite3, subprocess, argparse, multiprocessing as mp
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import List, Tuple
# from tqdm import tqdm
# import torch     # 메인 프로세스에서만 import

# # ────── 공통 경로/설정 ────────────────────────────────────────────────
# DB_PATH        = "data/speech_segments.db"
# RAW_VIDEO_DIR  = Path("/mnt/shares/videos")
# WAV_OUT_ROOT   = Path("/mnt/third_ssd/data/wav")
# FRAME_OUT_ROOT = Path("/mnt/third_ssd/data/frames")
# FPS            = 1
# # ────────────────────────────────────────────────────────────────────

# # ------------------ ffmpeg Helper ----------------------------------
# def run_ffmpeg(cmd: list[str]):
#     subprocess.run(cmd, check=True, stderr=subprocess.PIPE,
#                    stdout=subprocess.DEVNULL)

# # ------------- 개별 세그먼트 CPU 워커 함수 -------------------------
# def process_segment(task: Tuple, cache_dir: Path) -> str | None:
#     """
#     task: (seg_id, video_id, start, end, script)
#     성공 -> None, 실패/스킵 -> str(메시지)
#     """
#     seg_id, vid, st, et, script = task
#     video_file = RAW_VIDEO_DIR / f"{vid}.mp4"
#     if not video_file.exists():
#         return f"{seg_id}: missing video"

#     wav_path  = WAV_OUT_ROOT   / vid / f"{seg_id}.wav"
#     frame_dir = FRAME_OUT_ROOT / vid / seg_id

#     try:
#         # ① wav
#         if not wav_path.exists():
#             wav_path.parent.mkdir(parents=True, exist_ok=True)
#             run_ffmpeg([
#                 "ffmpeg","-nostdin","-loglevel","error","-y",
#                 "-ss",str(st),"-to",str(et),"-i",str(video_file),
#                 "-ar","16000","-ac","1",str(wav_path)
#             ])

#         # ② frames
#         if not (frame_dir.exists() and any(frame_dir.iterdir())):
#             frame_dir.mkdir(parents=True, exist_ok=True)
#             run_ffmpeg([
#                 "ffmpeg","-nostdin","-loglevel","error","-y",
#                 "-ss",str(st),"-to",str(et),"-i",str(video_file),
#                 "-vf",f"fps={FPS}",str(frame_dir/"f%06d.jpg")
#             ])

#         # ③ 임베딩 (embed_utils 는 GPU 프로세스에서 import 됨)
#         from embed_utils import embed_video, embed_audio
        
#         embed_video(frame_dir, cache_dir, seg_id)
#         embed_audio(wav_path, cache_dir, seg_id)
#         return None
#     except Exception as e:
#         return f"{seg_id}: {e}"

# # ------------------ GPU 전용 프로세스 함수 --------------------------
# def gpu_worker(gpu_id: int,
#                rows_slice: List[Tuple],
#                cache_dir: Path,
#                cpu_workers: int):

#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # GPU 고정
#     from embed_utils import embed_video, embed_audio  # GPU 올라갈 시점

#     desc  = f"GPU{gpu_id}"
#     fails = 0
#     with ProcessPoolExecutor(max_workers=cpu_workers) as ex:
#         futs = [ex.submit(process_segment, r, cache_dir) for r in rows_slice]
#         pbar = tqdm(as_completed(futs), total=len(futs), ncols=95, desc=desc)
#         for fut in pbar:
#             err = fut.result()
#             if err:
#                 fails += 1
#                 pbar.write("⚠ " + err)
#     print(f"[GPU{gpu_id}] done. fails={fails}/{len(rows_slice)}")

# # --------------------------- 메인 -----------------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--cache",  default="./cache")
#     ap.add_argument("--workers", type=int, default=os.cpu_count())
#     args = ap.parse_args()

#     cache_dir = Path(args.cache); cache_dir.mkdir(exist_ok=True)

#     # DB load
#     conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
#     rows = conn.execute(
#         "SELECT segment_id, video_id, start_time, end_time, script "
#         "FROM speech_segments"
#     ).fetchall(); conn.close()
#     rows: List[Tuple[str, str, float, float, str]] = [
#         (r["segment_id"], r["video_id"], r["start_time"], r["end_time"], r["script"])
#         for r in rows
#         ]
#     if not rows:
#         print("❌ No segments in DB."); return

#     if torch.cuda.device_count() < 4:
#         print(f"❌ Need 4 GPUs, found {torch.cuda.device_count()}"); return

#     # 4 등분
#     shard = (len(rows) + 3) // 4
#     cpu_per_gpu = max(1, args.workers // 4)

#     procs = []
#     for gpu in range(4):
#         subset = rows[gpu*shard : (gpu+1)*shard]
#         p = mp.Process(target=gpu_worker,
#                        args=(gpu, subset, cache_dir, cpu_per_gpu))
#         p.start(); procs.append(p)
#         print(f"Spawned GPU{gpu}  segs={len(subset):,}  cpu={cpu_per_gpu}")

#     for p in procs: p.join()
#     print(" All GPUs finished.")

# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)   # CUDA 안전
#     main()


#!/usr/bin/env python3
"""
(1) 세그먼트 추출(wav + frames)  +  (2) 임베딩 캐시 생성
GPU 4개를 전용 프로세스 4개로 분산. GPU 프로세스는
'영상(video_id) 단위' 로 순차 처리하여 같은 영상의
세그먼트를 모두 끝낸 뒤 다음 영상으로 진행한다.

Usage
-----
python extract_and_embed.py \
       --cache   ./cache \
       --workers 64           # CPU 워커 총합 (GPU당 자동 ¼)
"""

from __future__ import annotations
import os, sqlite3, subprocess, argparse, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch   # 메인 프로세스에서만 import

# ──────────────────────────────────────────────────────────────
#  📣  LOG / WARNING 필터링 ― tqdm 빼고 모두 잠재우기
# ──────────────────────────────────────────────────────────────
import os, warnings, logging

# 1) Python warnings 전체 off  (필요하면 category 조건 주기)
warnings.filterwarnings("ignore")

# 2) HuggingFace 노이즈 끄기
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"]            = "error"   # fatal|error|warning|info|debug

# 3) 불필요한 로거 레벨 낮추기
for name in [
    "transformers", "urllib3", "accelerate", "huggingface_hub",
    "numba",        "requests", "PIL"
]:
    logging.getLogger(name).setLevel(logging.ERROR)

# 4) (선택) ffmpeg Python-binding 경고 차단
logging.getLogger("ffmpeg").setLevel(logging.ERROR)
# ──────────────────────────────────────────────────────────────

# ─────── 공통 경로/설정 ────────────────────────────────────────────
DB_PATH        = "data/speech_segments.db"
RAW_VIDEO_DIR  = Path("/mnt/shares/videos")
WAV_OUT_ROOT   = Path("/mnt/third_ssd/data/wav")
FRAME_OUT_ROOT = Path("/mnt/third_ssd/data/frames")
FPS            = 1
# ─────────────────────────────────────────────────────────────────

# ------------------ ffmpeg Helper --------------------------------
def run_ffmpeg(cmd: List[str]):
    """ffmpeg 한 번 실행, 오류 시 RuntimeError."""
    subprocess.run(cmd, check=True,
                   stderr=subprocess.PIPE,
                   stdout=subprocess.DEVNULL)

# ------------- 개별 세그먼트 CPU 워커 함수 -----------------------
def process_segment(task: Tuple, cache_dir: Path) -> str | None:
    """
    task = (segment_id, video_id, start, end, script)
      성공 → None
      실패 → "msg" 문자열
    """
    seg_id, vid, st, et, script = task
    video_file = RAW_VIDEO_DIR / f"{vid}.mp4"
    if not video_file.exists():
        return f"{seg_id}: missing video"

    wav_path  = WAV_OUT_ROOT   / vid / f"{seg_id}.wav"
    frame_dir = FRAME_OUT_ROOT / vid / seg_id

    try:
        # ① wav 추출
        if not wav_path.exists():
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            run_ffmpeg([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-ss", str(st), "-to", str(et), "-i", str(video_file),
                "-ar", "16000", "-ac", "1", str(wav_path)
            ])

        # ② frame 추출
        if not (frame_dir.exists() and any(frame_dir.iterdir())):
            frame_dir.mkdir(parents=True, exist_ok=True)
            run_ffmpeg([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-ss", str(st), "-to", str(et), "-i", str(video_file),
                "-vf", f"fps={FPS}", str(frame_dir / "f%06d.jpg")
            ])

        # ③ 임베딩 (GPU 프로세스에 올라온 모델 이용)
        from embed_utils import embed_video, embed_audio
        embed_video(frame_dir, cache_dir, seg_id)
        embed_audio(wav_path, cache_dir, seg_id)
        return None

    except Exception as e:
        return f"{seg_id}: {e}"

# ------------------ GPU 전용 프로세스 -----------------------------
def gpu_worker(
    gpu_id: int,
    rows_slice: List[Tuple[str, str, float, float, str]],
    cache_dir: Path,
    cpu_workers: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # 고정 할당
    from embed_utils import embed_video, embed_audio    # GPU 초기화

    # ❶ rows → video_id 로 그룹
    vids: Dict[str, List[Tuple]] = {}
    for row in rows_slice:
        vids.setdefault(row[1], []).append(row)   # row[1] == video_id

    total_videos = len(vids)
    print(f"[GPU{gpu_id}] handling {total_videos} videos")

    # ❷ 영상 단위 순차 처리
    for idx, (vid, seg_rows) in enumerate(vids.items(), 1):
        desc  = f"GPU{gpu_id}  {vid} ({idx}/{total_videos})"
        fails = 0

        with ProcessPoolExecutor(max_workers=cpu_workers) as ex:
            futs = [ex.submit(process_segment, r, cache_dir) for r in seg_rows]
            for fut in tqdm(as_completed(futs),
                            total=len(futs),
                            desc=desc,
                            ncols=95,
                            leave=False):
                err = fut.result()
                if err:
                    fails += 1
                    tqdm.write("⚠ " + err)

        done = len(seg_rows) - fails
        print(f"[GPU{gpu_id}] {vid}  done {done}/{len(seg_rows)}  fails={fails}")

    print(f"[GPU{gpu_id}] ALL videos finished.")

# --------------------------- 메인 ---------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--cache",   default="./cache",
                    help="임베딩 .pt 저장 폴더")
    pa.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="CPU worker 총합 (GPU당 자동 ¼)")
    args = pa.parse_args()

    cache_dir = Path(args.cache).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # DB 로드
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT segment_id, video_id, start_time, end_time, script "
        "FROM speech_segments"
    ).fetchall()
    conn.close()
    rows = [(r["segment_id"], r["video_id"], r["start_time"], r["end_time"], r["script"])
            for r in rows]
    if not rows:
        raise SystemExit("❌ No segments in DB.")

    num_gpu = torch.cuda.device_count()
    if num_gpu < 4:
        raise SystemExit(f"❌ Need 4 GPUs, found {num_gpu}")

    # ➊ 세그먼트를 GPU 수(4)로 균등 분할
    rows_per_gpu = (len(rows) + num_gpu - 1) // num_gpu
    cpu_per_gpu  = max(1, args.workers // num_gpu)

    procs = []
    for gid in range(num_gpu):
        subset = rows[gid * rows_per_gpu : (gid + 1) * rows_per_gpu]
        p = mp.Process(target=gpu_worker,
                       args=(gid, subset, cache_dir, cpu_per_gpu),
                       daemon=False)
        p.start()
        procs.append(p)
        print(f"Spawned GPU{gid}  segs={len(subset):,}  cpu={cpu_per_gpu}")

    for p in procs:
        p.join()

    print("🎉  All GPU workers finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # CUDA 다중 프로세스 안전
    main()
