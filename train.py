# # train.py
# import argparse, math, os, random, time
# from pathlib import Path

# import torch
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler
# from tqdm.auto import tqdm
# from torch_geometric.loader import DataLoader as GeoDataLoader
# from dataset.data import VideoPersonDataset
# from models.manager_graphtokens import GraphTokenManager   # 방금 만든 모델

# # ──────────────────────────
# def seed_all(seed: int):
#     random.seed(seed);  torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)

# # ──────────────────────────
# def get_dataloaders(db, cache, frames, wav, batch=1, max_samples=None, seed=42, filter_video_ids=None):
#     train_ds = VideoPersonDataset(
#         db_path=db, split="train",
#         cache_dir=cache, video_root=frames, audio_root=wav,
#         merge_dialog=True, max_samples=max_samples, filter_ids=filter_video_ids, seed=seed
#     )
#     val_ds   = VideoPersonDataset(
#         db_path=db, split="val",
#         cache_dir=cache, video_root=frames, audio_root=wav,
#         merge_dialog=True, max_samples=max_samples, filter_ids=filter_video_ids, seed=seed
#     )
#     train_loader = GeoDataLoader(train_ds, batch_size=batch, shuffle=True)
#     val_loader   = GeoDataLoader(val_ds, batch_size=1, shuffle=False)
#     return train_loader, val_loader

# # ──────────────────────────
# def evaluate(model, loader, device):
#     model.eval();  tp=fp=fn=0
#     with torch.no_grad():
#         for sample in loader:
#             g   = sample["graph"].to(device)
#             logit,_ = model(g, sample["person"][0])
#             pred = (torch.sigmoid(logit) > 0.5).item()
#             label= sample["label"].item()
#             if pred==1 and label==1: tp+=1
#             if pred==1 and label==0: fp+=1
#             if pred==0 and label==1: fn+=1
#     precision = tp / (tp+fp+1e-8)
#     recall    = tp / (tp+fn+1e-8)
#     f1        = 2*precision*recall/(precision+recall+1e-8)
#     return f1

# # ──────────────────────────
# def main(args):
#     seed_all(args.seed)
#     device = torch.device("cuda")

#     # Data
#     train_loader, val_loader = get_dataloaders(
#         db=args.db, cache=args.cache, frames=args.frames, wav=args.wav,
#         batch=1, max_samples=args.max_samples, seed=args.seed
#     )

#     # Model
#     model = GraphTokenManager().half().to(device)
#     model.train()

#     # Optimizer (8-bit AdamW)
#     optim = torch.optim.AdamW(
#         model.parameters(), lr=args.lr, betas=(0.9,0.95), eps=1e-8
#     )
#     scaler = GradScaler()

#     best_f1 = 0.0
#     ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(exist_ok=True)

#     for epoch in range(1, args.epochs+1):
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
#         for sample in pbar:
#             g   = sample["graph"].to(device)
#             lbl = sample["label"].float().unsqueeze(0).to(device)

#             optim.zero_grad()

#             # ── (A) Forward
#             with autocast():
#                 logit, loss = model(g, sample["person"][0], lbl)

#             # ── (B) 역전파 직전 메모리 디버그 출력
#             torch.cuda.synchronize()
#             dev_id      = torch.cuda.current_device()
#             props       = torch.cuda.get_device_properties(dev_id)
#             total_mem   = props.total_memory   / 1024**3
#             alloc_mem   = torch.cuda.memory_allocated(dev_id)    / 1024**3
#             reserved    = torch.cuda.memory_reserved(dev_id)     / 1024**3
#             free_cache  = reserved - alloc_mem
#             print(f"[GPU{dev_id}] 총 용량 : {total_mem:.2f} GB")
#             print(f"[GPU{dev_id}] 할당됨 : {alloc_mem:.2f} GB")
#             print(f"[GPU{dev_id}] 예약됨 : {reserved:.2f} GB")
#             print(f"[GPU{dev_id}] 캐시 여유: {free_cache:.2f} GB")
#             print(torch.cuda.memory_summary(device=dev_id, abbreviated=True))

#             # ── (C) Backward
#             scaler.scale(loss).backward()
#             scaler.unscale_(optim)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optim); scaler.update()

#             pbar.set_postfix(loss=loss.item())

#         # ── validation ──
#         f1 = evaluate(model, val_loader, device)
#         print(f"epoch {epoch}  val F1={f1:.4f}")

#         if f1 > best_f1:
#             best_f1 = f1
#             torch.save({
#                 "gcn" : model.gcn.state_dict(),
#                 "proj": model.proj_up.state_dict(),
#                 "lora": model.glm.state_dict(),      # LoRA adapter
#                 "optim": optim.state_dict(),
#             }, ckpt_dir / "best.pt")
#             print(f"  ✔ saved new best checkpoint (F1 {best_f1:.4f})")

# # ──────────────────────────
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--db",          default="data/speech_segments.db")
#     ap.add_argument("--cache",       default="cache/")
#     ap.add_argument("--frames",      default="data/frames/")
#     ap.add_argument("--wav",         default="data/wav/")
#     ap.add_argument("--ckpt_dir",    default="checkpoints/")
#     ap.add_argument("--epochs", type=int, default=5)
#     ap.add_argument("--lr",     type=float, default=2e-4)
#     ap.add_argument("--seed",   type=int,   default=42)
#     ap.add_argument("--max_samples", type=int, default=None,
#                     help="for quick debugging; use None for full data")
#     args = ap.parse_args()
#     main(args)
# train.py
import argparse
import json
import os
import random
import subprocess
import sqlite3
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm.auto import tqdm

from dataset.data import VideoPersonDataset
from models.manager_graphtokens import GraphTokenManager
# ─── GPU 메모리 리포트 함수 추가 ───────────────────────────────
import subprocess, textwrap, re, gc, torch

def report_gpu_mem(tag=""):
    """GPU별 alloc / reserved / 캐시 / peak 을 한눈에 표시"""
    torch.cuda.synchronize()
    print(f"\n=== GPU 메모리 리포트 {tag} ===")
    for idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(idx)
        alloc    = torch.cuda.memory_allocated()   / 1024**2  # MB
        reserved = torch.cuda.memory_reserved()    / 1024**2
        inactive = reserved - alloc
        peak     = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[GPU{idx}] alloc {alloc:7.1f} | reserved {reserved:7.1f} "
              f"| inactive {inactive:7.1f} | peak {peak:7.1f} MB")

        # 가장 큰 블록이 어디서 생겼는지 한 줄만 추려 보기
        for ln in torch.cuda.memory_summary().splitlines():
            if re.search(r"size:", ln):
                print("        ", ln.strip())
                break

        torch.cuda.reset_peak_memory_stats()

    # nvidia-smi 숫자도 덧붙이기 (optional)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True)
        rows = out.strip().splitlines()
        print("[nvidia-smi] used/total (MB):",
              " | ".join(f"GPU{n}:{r}" for n, r in enumerate(rows)))
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────

# ──────────────────────────
#  VRAM 로그 함수
# ──────────────────────────
def log_vram(stage: str, device: torch.device):
    torch.cuda.synchronize()
    dev_id   = device.index if isinstance(device, torch.device) else torch.cuda.current_device()
    alloc    = torch.cuda.memory_allocated(dev_id)    / 1024**2
    reserved = torch.cuda.memory_reserved(dev_id)     / 1024**2
    peak_a   = torch.cuda.max_memory_allocated(dev_id) / 1024**2
    peak_r   = torch.cuda.max_memory_reserved(dev_id)  / 1024**2
    wasted   = reserved - alloc
    print(f"[{stage:15s}] alloc: {alloc:6.1f} MB | reserved: {reserved:6.1f} MB | "
          f"peak_alloc: {peak_a:6.1f} MB | peak_reserved: {peak_r:6.1f} MB | wasted: {wasted:6.1f} MB")
    torch.cuda.reset_peak_memory_stats(dev_id)

# ──────────────────────────
#  시드 설정
# ──────────────────────────
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ──────────────────────────
#  ffprobe 로 비디오 길이(초) 가져오기
# ──────────────────────────
def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(proc.stdout)
    return float(info["format"]["duration"])

# ──────────────────────────
#  DataLoader 생성
# ──────────────────────────
def get_dataloaders(db, cache, frames, wav, batch=1, max_samples=None,
                    seed=42, filter_video_ids=None):
    train_ds = VideoPersonDataset(
        db_path=db, split="train",
        cache_dir=cache, video_root=frames, audio_root=wav,
        merge_dialog=True, max_samples=max_samples,
        filter_ids=filter_video_ids, seed=seed,
    )
    val_ds = VideoPersonDataset(
        db_path=db, split="val",
        cache_dir=cache, video_root=frames, audio_root=wav,
        merge_dialog=True, max_samples=max_samples,
        filter_ids=filter_video_ids, seed=seed,
    )
    train_loader = GeoDataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = GeoDataLoader(val_ds, batch_size=1, shuffle=False)
    return train_loader, val_loader

# ──────────────────────────
#  검증 함수
# ──────────────────────────
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for sample in loader:
            g     = sample["graph"].to(device)
            logit, _ = model(g, sample["person"][0])
            pred  = (torch.sigmoid(logit) > 0.5).item()
            label = sample["label"].item()
            if pred == 1 and label == 1: tp += 1
            if pred == 1 and label == 0: fp += 1
            if pred == 0 and label == 1: fn += 1
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

# ──────────────────────────
#  메인
# ──────────────────────────
def main(args):
    seed_all(args.seed)
    device = torch.device("cuda")

    # 1) DB 에서 전체 video_id 수집
    conn = sqlite3.connect(args.db)
    cur  = conn.execute("SELECT DISTINCT video_id FROM speech_segments")
    all_video_ids = [row[0] for row in cur.fetchall()]
    conn.close()

    # 2) 길이 < 600초(10분) 영상만 필터링
    RAW_VIDEO_DIR = Path(args.frames).parent / "videos"
    short_videos = []
    for vid in all_video_ids:
        path = RAW_VIDEO_DIR / f"{vid}.mp4"
        if path.exists() and get_video_duration(path) < 600.0:
            short_videos.append(vid)
    print(f"총 영상 {len(all_video_ids):,}개 중 길이 < 10분: {len(short_videos):,}개")

    # 3) DataLoader 준비
    train_loader, val_loader = get_dataloaders(
        db=args.db, cache=args.cache, frames=args.frames, wav=args.wav,
        batch=args.batch_size, max_samples=args.max_samples,
        seed=args.seed, filter_video_ids=short_videos,
    )

    # 모델 초기화
    model  = GraphTokenManager().half().to(device)
    optim  = torch.optim.AdamW(model.parameters(),
                               lr=args.lr, betas=(0.9,0.95), eps=1e-8)
    scaler = GradScaler()
    best_f1 = 0.0

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # 학습 루프
    for epoch in range(1, args.epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        torch.cuda.empty_cache()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for sample in pbar:
            # Stage 0: step 시작 전
            log_vram("before_step", device)

            g   = sample["graph"].to(device)
            lbl = sample["label"].float().unsqueeze(0).to(device)
            log_vram("after_data_to_cuda", device)

            optim.zero_grad()

            # Stage 1: forward
            with autocast():
                logit, loss = model(g, sample["person"][0], lbl)
            log_vram("after_forward", device)

            # Stage 2: backward
            scaler.scale(loss).backward()
            report_gpu_mem(f"step {pbar.n} - after backward")
            log_vram("after_backward", device)

            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            log_vram("after_opt_step", device)
            del g, logit, loss, sample
            torch.cuda.empty_cache();  gc.collect()
            report_gpu_mem(f"step {pbar.n} - after cleanup")
            pbar.set_postfix(loss=loss.item())

        # 검증
        f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} Validation F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "gcn" : model.gcn.state_dict(),
                "proj": model.proj_up.state_dict(),
                "lora": model.glm.state_dict(),
                "optim": optim.state_dict(),
            }, ckpt_dir / "best.pt")
            print(f"✔ New best checkpoint saved (F1 {best_f1:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",          default="data/speech_segments.db")
    ap.add_argument("--cache",       default="cache/")
    ap.add_argument("--frames",      default="data/frames/")
    ap.add_argument("--wav",         default="data/wav/")
    ap.add_argument("--ckpt_dir",    default="checkpoints/")
    ap.add_argument("--epochs", type=int,   default=5)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--seed",   type=int,   default=42)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="for quick debugging; None 이면 전체 데이터")
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()
    main(args)

