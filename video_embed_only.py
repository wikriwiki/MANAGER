#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
video_embed_only.py

- AV1 안정 디코딩(dav1d 우선), JPEG→PNG 폴백
- CSV 기준 video_graph 캐시(.pt) 없는 video_id만 처리 옵션
- GPU당 여러 모델 프로세스
- 세그먼트/비디오 캐시 프리체크, forward-fill

Usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python video_embed_only.py \
  --cache ./cache --workers 64 \
  --models-per-gpu 2 --model-vram-est-mb 1600 --vram-reserve-mb 1500 \
  --only-missing-from-csv MMSF_dataset.csv --csv-cache-dir cache
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import multiprocessing as mp
import os
import sqlite3
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

DB_PATH        = "data/speech_segments.db"
RAW_VIDEO_DIR  = Path("/mnt/shares/videos")
FRAME_OUT_ROOT = Path("/mnt/third_ssd/data/frames")
FPS            = 1
MIN_DUR_SEC    = 1.0 / FPS

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"]            = "error"
for _n in ["transformers","urllib3","accelerate","huggingface_hub","numba","requests","PIL"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

EMBED_EXTS = (".npy",".npz",".pt",".pth",".pkl",".pickle",".bin",".json",".arrow",".feather")
SKIP_SUFFIXES = (".lock",".tmp",".part",".incomplete")


def _check_runtime_deps() -> None:
    try:
        import PIL  # noqa: F401
    except Exception:
        raise SystemExit(
            "❌ Pillow(PIL)가 없습니다.\n"
            "   conda install -c conda-forge 'pillow>=10.3,<11' libjpeg-turbo"
        )


def safe_exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:
        return False


def to_float_or_none(x) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _has_valid_file(p: Path) -> bool:
    if not p.is_file():
        return False
    if any(str(p.name).endswith(sfx) for sfx in SKIP_SUFFIXES):
        return False
    try:
        return p.stat().st_size > 0
    except Exception:
        return False


def is_cached_embedding(cache_dir: Path, seg_id: str, vid: Optional[str] = None) -> bool:
    for ext in EMBED_EXTS:
        p1 = cache_dir / f"{seg_id}{ext}"
        if safe_exists(p1) and _has_valid_file(p1):
            return True
        if vid:
            p2 = cache_dir / vid / f"{seg_id}{ext}"
            if safe_exists(p2) and _has_valid_file(p2):
                return True
    for d in [cache_dir / seg_id, (cache_dir / vid / seg_id) if vid else None]:
        if d and d.is_dir():
            try:
                for f in d.iterdir():
                    if f.suffix in EMBED_EXTS and _has_valid_file(f):
                        return True
            except Exception:
                pass
    return False


def _hash_video_graph_key(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest() + ".pt"


def compute_missing_video_ids(csv_path: str, cache_dir: Path, col: str = "video_id") -> set:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise SystemExit(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise SystemExit(f"❌ CSV 읽기 실패: {e}")
    if col not in df.columns:
        raise SystemExit(f"❌ CSV에 '{col}' 컬럼이 없습니다")

    vids = df[col].dropna().astype(str).unique().tolist()
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing, found = [], 0
    for vid in vids:
        fname = _hash_video_graph_key(f"video_graph::{vid}")
        p = cache_dir / fname
        if p.is_file() and p.stat().st_size > 0:
            found += 1
        else:
            missing.append(vid)

    print("✅ CSV 캐시 스캔 요약")
    print(f"  - 총 video_id: {len(vids)}")
    print(f"  - 캐시 존재: {found}")
    print(f"  - 캐시 없음(임베딩 대상): {len(missing)}")
    return set(missing)


def _run(cmd: List[str]) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return r.returncode, r.stdout or "", r.stderr or ""
    except Exception as e:
        return -1, "", f"{type(e).__name__}: {e}"


def detect_ffmpeg_features() -> Dict[str, bool]:
    feats = {"libdav1d": False, "libaom-av1": False, "av1": False,
             "av1_cuvid": False, "cuda_hwaccel": False}
    _, out1, err1 = _run(["ffmpeg", "-hide_banner", "-decoders"])
    txt1 = (out1 + err1).lower()
    for k in ["libdav1d", "libaom-av1", "av1_cuvid", "av1"]:
        feats[k] = (k in txt1)
    _, out2, err2 = _run(["ffmpeg", "-hide_banner", "-hwaccels"])
    feats["cuda_hwaccel"] = ("cuda" in (out2 + err2).lower())
    return feats


FFMPEG_FEATS = detect_ffmpeg_features()


def get_video_codec(video_file: Path) -> Optional[str]:
    code, out, _ = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nw=1:nk=1",
        str(video_file)
    ])
    return out.strip() if code == 0 and out.strip() else None


def _run_ffmpeg(cmd: List[str]) -> tuple[bool, str]:
    try:
        res = subprocess.run(cmd, check=False, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
        return (res.returncode == 0), (res.stderr or b"").decode("utf-8", "ignore").strip()
    except FileNotFoundError:
        return False, "ffmpeg_not_found"
    except Exception as e:
        return False, f"exception:{type(e).__name__}:{e}"


def _build_seek_cmd(seek_mode: str, start: float, duration: float,
                    input_args: List[str], vf: str, out_pattern: str,
                    pix_fmt: str = "yuvj420p", strict_unofficial: bool = True) -> List[str]:
    base = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if seek_mode == "fast":
        base += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"] + input_args
    else:
        base += input_args + ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"]
    out_args = ["-vf", vf, "-pix_fmt", pix_fmt]
    if strict_unofficial:
        out_args += ["-strict", "-2"]
    return base + out_args + [out_pattern]


def _input_args_with_decoder(video_file: Path, decoder: str) -> tuple[List[str], str]:
    if decoder == "av1_cuvid":
        return (["-hwaccel", "cuda", "-c:v", "av1_cuvid", "-i", str(video_file)], ",hwdownload")
    elif decoder in ("libdav1d", "libaom-av1", "av1"):
        return (["-c:v", decoder, "-i", str(video_file)], "")
    else:
        return (["-i", str(video_file)], "")


def _decoder_variants_for(codec_hint: Optional[str]) -> List[str]:
    variants: List[str] = []
    if codec_hint == "av1":
        if FFMPEG_FEATS.get("libdav1d"):   variants.append("libdav1d")
        if FFMPEG_FEATS.get("libaom-av1"): variants.append("libaom-av1")
        if FFMPEG_FEATS.get("av1_cuvid") and FFMPEG_FEATS.get("cuda_hwaccel"):
            variants.append("av1_cuvid")
    variants.append("")  # 기본
    return variants


def extract_frames(video_file: Path, out_dir: Path, start: float, end: float, codec_hint: Optional[str]) -> tuple[bool, str]:
    duration = max(MIN_DUR_SEC, float(end) - float(start))
    jpg_pattern = str(out_dir / "f%06d.jpg")
    png_pattern = str(out_dir / "f%06d.png")

    tried: List[str] = []
    for dec in _decoder_variants_for(codec_hint):
        input_args, vf_fix = _input_args_with_decoder(video_file, dec)
        vf = f"fps={FPS},format=yuv420p" if not vf_fix else f"fps={FPS},hwdownload,format=yuv420p"

        for seek_mode in ("fast", "slow"):
            # JPEG
            cmd_jpg = _build_seek_cmd(seek_mode, start, duration, input_args, vf,
                                      jpg_pattern, pix_fmt="yuvj420p", strict_unofficial=True)
            ok, msg = _run_ffmpeg(cmd_jpg)
            if ok:
                try:
                    if any(out_dir.iterdir()):
                        return True, ""
                except Exception:
                    pass
                msg = "no_frames_generated(jpg)"

            # PNG fallback
            cmd_png = _build_seek_cmd(seek_mode, start, duration, input_args, vf,
                                      png_pattern, pix_fmt="rgb24", strict_unofficial=False)
            ok2, msg2 = _run_ffmpeg(cmd_png)
            if ok2:
                try:
                    if any(out_dir.iterdir()):
                        return True, ""
                except Exception:
                    pass
                msg2 = "no_frames_generated(png)"

            tried.append(f"{dec or 'default'}:{seek_mode}:jpg({msg})|png({msg2 if not ok2 else 'ok'})")

    return False, "; ".join(tried)


def cpu_extract_task(seg: Tuple[str, str, float, float], codec_hint: Optional[str], cache_dir: Path) -> tuple[str, Optional[str]]:
    seg_id, vid, st, et = seg
    if is_cached_embedding(cache_dir, seg_id, vid):
        return seg_id, None

    video_file = RAW_VIDEO_DIR / f"{vid}.mp4"
    if not safe_exists(video_file):
        return seg_id, f"video_missing_or_io: {video_file}"

    frame_dir = FRAME_OUT_ROOT / vid / seg_id
    try:
        already = frame_dir.exists() and any(frame_dir.iterdir())
    except Exception:
        already = False

    if not already:
        frame_dir.mkdir(parents=True, exist_ok=True)
        ok, msg = extract_frames(video_file, frame_dir, st, et, codec_hint)
        if not ok:
            return seg_id, f"ffmpeg_fail: {msg}"
    return seg_id, None


def gpu_worker(gid: int, rows_slice: List[Tuple[str, str, float, float]], cache_dir: Path, cpu_threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    from embed_utils import embed_video  # GPU 워커에서만 로드

    vids: Dict[str, List[Tuple[str, str, float, float]]] = {}
    for seg in rows_slice:
        vids.setdefault(seg[1], []).append(seg)

    print(f"[GPU{gid}] videos={len(vids)}  segs={len(rows_slice)}  cpu_threads={cpu_threads}")

    mp_ctx = mp.get_context("spawn")
    for idx, (vid, segs_all) in enumerate(vids.items(), 1):
        segs = [s for s in segs_all if not is_cached_embedding(cache_dir, s[0], vid)]
        if not segs:
            print(f"[GPU{gid}] {vid}  all segments cached → skip")
            continue

        desc = f"GPU{gid} {vid} ({idx}/{len(vids)})"
        video_file = RAW_VIDEO_DIR / f"{vid}.mp4"
        codec_hint = (get_video_codec(video_file) or "").lower() or None

        # 1) 프레임 추출
        fails_extract = 0
        with ProcessPoolExecutor(max_workers=cpu_threads, mp_context=mp_ctx) as ex:
            futs = [ex.submit(cpu_extract_task, s, codec_hint, cache_dir) for s in segs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc+" [extract]", ncols=95, leave=False):
                sid, err = fut.result()
                if err:
                    fails_extract += 1
                    tqdm.write(f"⚠ {sid}: {err}")

        # 2) 임베딩
        fails_embed = 0
        last_good_frame_dir: Optional[Path] = None

        for seg_id, _, _, _ in tqdm(segs, desc=desc+" [embed]", ncols=95, leave=False):
            if is_cached_embedding(cache_dir, seg_id, vid):
                tqdm.write(f"⏭ {seg_id}: cached")
                continue

            cur_dir = FRAME_OUT_ROOT / vid / seg_id
            try:
                has_frames = cur_dir.exists() and any(cur_dir.iterdir())
            except Exception:
                has_frames = False

            try:
                if has_frames:
                    embed_video(cur_dir, cache_dir, seg_id)
                    last_good_frame_dir = cur_dir
                elif last_good_frame_dir is not None:
                    tqdm.write(f"ℹ {seg_id}: embed_from_prev_frames -> {last_good_frame_dir}")
                    embed_video(last_good_frame_dir, cache_dir, seg_id)
                else:
                    fails_embed += 1
                    tqdm.write(f"⚠ {seg_id}: no_frames_and_no_prev")
            except Exception as e:
                fails_embed += 1
                tqdm.write(f"⚠ {seg_id}: embed_fail {type(e).__name__}: {e}")

        ok_est = max(0, len(segs) - max(fails_extract, 0) - max(fails_embed, 0))
        print(f"[GPU{gid}] {vid}  ok≈{ok_est}/{len(segs)}  extract_fails={fails_extract}  embed_fails={fails_embed}")


def gpu_worker_wrapped(gid: int, idx: int, rows_slice, cache_dir, cpu_threads, per_proc_frac: float) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    try:
        try:
            torch.cuda.set_per_process_memory_fraction(per_proc_frac, device=0)
        except Exception:
            pass
        gpu_worker(gid, rows_slice, cache_dir, cpu_threads)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[GPU{gid}.{idx}] ⚠ CUDA OOM. per_proc_frac={per_proc_frac}. 인스턴스 수/배치를 줄이세요.")
        else:
            print(f"[GPU{gid}.{idx}] RuntimeError: {e}")
        raise


def load_and_ffill_rows(bad_log_path: Path) -> List[Tuple[str, str, float, float]]:
    if not Path(DB_PATH).exists():
        raise SystemExit(f"❌ DB가 없습니다: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT segment_id, video_id, start_time, end_time
        FROM speech_segments
        WHERE segment_id IS NOT NULL AND video_id IS NOT NULL
        ORDER BY video_id, start_time NULLS LAST, segment_id
    """)
    rows_raw = [dict(r) for r in cur]
    conn.close()

    vids: Dict[str, List[dict]] = {}
    for r in rows_raw:
        vids.setdefault(r["video_id"], []).append(r)

    rows: List[Tuple[str, str, float, float]] = []
    bad_lines: List[str] = []

    for vid, items in vids.items():
        prev_st: Optional[float] = None
        prev_et: Optional[float] = None

        for r in items:
            seg_id = r["segment_id"]
            st = to_float_or_none(r["start_time"])
            et = to_float_or_none(r["end_time"])
            reason = None

            if st is None or et is None:
                if prev_st is not None and prev_et is not None:
                    st, et = prev_st, prev_et
                    reason = f"ffill_null st={r['start_time']} et={r['end_time']}"
                else:
                    bad_lines.append(f"{seg_id}\t{vid}\tnull_no_prev_skip\tst={r['start_time']}\tet={r['end_time']}")
                    continue

            if st < 0:
                if prev_st is not None and prev_et is not None:
                    reason = f"ffill_neg_start st={st}"
                    st, et = prev_st, prev_et
                else:
                    bad_lines.append(f"{seg_id}\t{vid}\tneg_start_clamped\tst={st}->0.0\tet={et}")
                    st = 0.0

            if et <= st:
                if prev_st is not None and prev_et is not None:
                    reason = f"ffill_nonpos_dur st={st} et={et}"
                    st, et = prev_st, prev_et
                else:
                    bad_lines.append(f"{seg_id}\t{vid}\tnonpos_dur_no_prev_skip\tst={st}\tet={et}")
                    continue

            if (et - st) < MIN_DUR_SEC:
                et = st + MIN_DUR_SEC
                if reason is None:
                    reason = "min_dur_clamped"

            rows.append((seg_id, vid, float(st), float(et)))
            prev_st, prev_et = float(st), float(et)
            if reason:
                bad_lines.append(f"{seg_id}\t{vid}\t{reason}\tst={st}\tet={et}")

    if bad_lines:
        try:
            with open(bad_log_path, "w", encoding="utf-8") as f:
                f.write("# segment_id\tvideo_id\treason\tst\tet\n")
                for line in bad_lines:
                    f.write(line + "\n")
            print(f"ℹ 전진 채움/보정/스킵 로그 {len(bad_lines)}행 → {bad_log_path}")
        except Exception as e:
            print(f"⚠ bad_segments.log 기록 실패: {e}")

    if not rows:
        raise SystemExit("❌ 유효/대체 가능한 세그먼트가 없습니다")

    return rows


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--cache", default="./cache")
    pa.add_argument("--workers", type=int, default=32, help="총 CPU 워커 수 (GPU당 workers/#GPUs)")
    pa.add_argument("--models-per-gpu", type=int, default=12, help="한 GPU에 띄울 모델/프로세스 개수")
    pa.add_argument("--vram-reserve-mb", type=int, default=1500, help="VRAM 안전 여유(MB)")
    pa.add_argument("--model-vram-est-mb", type=int, default=1400, help="모델 1개+런타임 추정 VRAM(MB)")
    pa.add_argument("--only-missing-from-csv", type=str, default="MMSF_dataset.csv", help="CSV의 video_id 중 캐시(.pt) 없는 것만 임베딩")
    pa.add_argument("--csv-cache-dir", type=str, default="cache", help="CSV 캐시(.pt) 탐색 디렉터리")
    pa.add_argument("--csv-videoid-col", type=str, default="video_id", help="CSV 내 video_id 컬럼명")
    args = pa.parse_args()

    _check_runtime_deps()

    cache_dir = Path(args.cache).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = load_and_ffill_rows(cache_dir / "bad_segments.log")

    # CSV 기반 비디오 필터
    if args.only_missing_from_csv is not None:
        missing_vids = compute_missing_video_ids(args.only_missing_from_csv,
                                                 Path(args.csv_cache_dir),
                                                 col=args.csv_videoid_col)
        before = len(rows)
        rows = [r for r in rows if r[1] in missing_vids]  # (seg_id, video_id, st, et)
        print(f"ℹ CSV 필터 적용: rows {before} → {len(rows)}")
        if not rows:
            raise SystemExit("✅ CSV 기준으로 임베딩 대상이 없습니다(모두 캐시 존재).")

    # 세그먼트 캐시 프리체크
    before = len(rows)
    rows = [r for r in rows if not is_cached_embedding(cache_dir, r[0], r[1])]
    skipped = before - len(rows)
    if skipped > 0:
        print(f"ℹ cache precheck: skip_cached={skipped}/{before}  to_process={len(rows)}")

    gpus = torch.cuda.device_count()
    if gpus <= 0:
        raise SystemExit("❌ 사용 가능한 CUDA GPU가 없습니다")

    rows_per_gpu = (len(rows) + gpus - 1) // gpus

    # per-GPU 인스턴스 수 결정 (NVML 가능 시 여유 고려)
    def _free_mb(dev: int) -> int:
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(dev)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            return mem.free // (1024 * 1024)
        except Exception:
            return 8000  # 정보 불가 시 보수적 추정

    req_k = max(1, args.models_per_gpu)
    real_k_per_gpu: List[int] = []
    for gid in range(gpus):
        free_mb = _free_mb(gid)
        cap = max(1, (free_mb - args.vram_reserve_mb) // max(1, args.model_vram_est_mb))
        real_k_per_gpu.append(max(1, min(req_k, cap)))

    procs: List[mp.Process] = []
    for gid in range(gpus):
        k = real_k_per_gpu[gid]
        subset_all = rows[gid * rows_per_gpu:(gid + 1) * rows_per_gpu]

        if k > 1:
            step = max(1, (len(subset_all) + k - 1) // k)
            slices = [subset_all[i * step:(i + 1) * step] for i in range(k)]
        else:
            slices = [subset_all]

        cpu_per_proc = max(1, (args.workers // gpus) // k)
        per_proc_frac = min(0.9, 0.9 / max(1, k))

        for j, subset in enumerate(slices):
            p = mp.Process(target=gpu_worker_wrapped,
                           args=(gid, j, subset, cache_dir, cpu_per_proc, per_proc_frac))
            p.start()
            procs.append(p)
            print(f"Spawn GPU{gid}.{j}: segs={len(subset)} cpu={cpu_per_proc} (k={k})")

    for p in procs:
        p.join()

    print("🎉  video embedding completed")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
