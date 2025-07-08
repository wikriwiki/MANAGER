# train.py
import argparse, math, os, random, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from data.person_dataset import VideoPersonDataset
from models.manager_graphtokens import GraphTokenManager   # 방금 만든 모델

# ──────────────────────────
def seed_all(seed: int):
    random.seed(seed);  torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)

# ──────────────────────────
def get_dataloaders(db, cache, frames, wav, batch=1, max_samples=None, seed=42):
    train_ds = VideoPersonDataset(
        db_path=db, split="train",
        cache_dir=cache, video_root=frames, audio_root=wav,
        merge_dialog=True, max_samples=max_samples, seed=seed
    )
    val_ds   = VideoPersonDataset(
        db_path=db, split="val",
        cache_dir=cache, video_root=frames, audio_root=wav,
        merge_dialog=True, max_samples=max_samples, seed=seed
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_loader, val_loader

# ──────────────────────────
def evaluate(model, loader, device):
    model.eval();  tp=fp=fn=0
    with torch.no_grad():
        for sample in loader:
            g   = sample["graph"].to(device)
            logit,_ = model(g, sample["person"][0])
            pred = (torch.sigmoid(logit) > 0.5).item()
            label= sample["label"].item()
            if pred==1 and label==1: tp+=1
            if pred==1 and label==0: fp+=1
            if pred==0 and label==1: fn+=1
    precision = tp / (tp+fp+1e-8)
    recall    = tp / (tp+fn+1e-8)
    f1        = 2*precision*recall/(precision+recall+1e-8)
    return f1

# ──────────────────────────
def main(args):
    seed_all(args.seed)
    device = torch.device("cuda")

    # Data
    train_loader, val_loader = get_dataloaders(
        db=args.db, cache=args.cache, frames=args.frames, wav=args.wav,
        batch=1, max_samples=args.max_samples, seed=args.seed
    )

    # Model
    model = GraphTokenManager().half().to(device)
    model.train()

    # Optimizer (8-bit AdamW)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9,0.95), eps=1e-8
    )
    scaler = GradScaler()

    best_f1 = 0.0
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for sample in pbar:
            g   = sample["graph"].to(device)
            lbl = sample["label"].float().unsqueeze(0).to(device)

            optim.zero_grad()
            with autocast():
                logit, loss = model(g, sample["person"][0], lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()

            pbar.set_postfix(loss=loss.item())

        # ── validation ──
        f1 = evaluate(model, val_loader, device)
        print(f"epoch {epoch}  val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "gcn" : model.gcn.state_dict(),
                "proj": model.proj_up.state_dict(),
                "lora": model.glm.state_dict(),      # LoRA adapter
                "optim": optim.state_dict(),
            }, ckpt_dir / "best.pt")
            print(f"  ✔ saved new best checkpoint (F1 {best_f1:.4f})")

# ──────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",          default="data/monopoly.sqlite")
    ap.add_argument("--cache",       default="cache/")
    ap.add_argument("--frames",      default="data/frames/")
    ap.add_argument("--wav",         default="data/wav/")
    ap.add_argument("--ckpt_dir",    default="checkpoints/")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--seed",   type=int,   default=42)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="for quick debugging; use None for full data")
    args = ap.parse_args()
    main(args)
