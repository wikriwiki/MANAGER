# validate.py
import argparse, json, torch
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support

from data.person_dataset import VideoPersonDataset
from models.manager_graphtokens import GraphTokenManager

# ────────────────────────────────────────────────
@torch.no_grad()
def run_eval(loader, model, device):
    model.eval()
    y_true, y_pred = [], []

    for sample in tqdm(loader, desc="infer"):
        g = sample["graph"].to(device)
        logit, _ = model(g, sample["person"][0])     # label=None
        pred = (torch.sigmoid(logit) > 0.5).int().item()
        y_pred.append(pred)
        y_true.append(sample["label"].item())

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": p, "recall": r, "f1": f1}

# ────────────────────────────────────────────────
def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.gcn.load_state_dict(ckpt["gcn"])
    model.proj_up.load_state_dict(ckpt["proj"])
    model.glm.load_state_dict(ckpt["lora"], strict=False)   # LoRA adapter
    print(f"checkpoint loaded: {ckpt_path}")

# ────────────────────────────────────────────────
def get_loader(db, cache, frames, wav, split, max_samples=None, seed=42):
    ds = VideoPersonDataset(
        db_path=db, split=split,
        cache_dir=cache, video_root=frames, audio_root=wav,
        merge_dialog=True, max_samples=max_samples, seed=seed
    )
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# ────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda")

    # 데이터 로더
    loader = get_loader(
        db=args.db, cache=args.cache, frames=args.frames, wav=args.wav,
        split=args.split, max_samples=args.max_samples
    )
    print(f"{args.split} samples: {len(loader.dataset)}")

    # 모델 로드
    model = GraphTokenManager().half().to(device)
    load_ckpt(model, args.ckpt)

    # 평가
    metrics = run_eval(loader, model, device)
    print(json.dumps(metrics, indent=2))

# ────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",      default="data/monopoly.sqlite")
    ap.add_argument("--cache",   default="cache/")
    ap.add_argument("--frames",  default="data/frames/")
    ap.add_argument("--wav",     default="data/wav/")
    ap.add_argument("--ckpt",    default="checkpoints/best.pt")
    ap.add_argument("--split",   default="val", choices=["val", "test"])
    ap.add_argument("--max_samples", type=int, default=None,
                    help="디버깅용 샘플 제한")
    args = ap.parse_args()
    main(args)

"""
# 검증 세트 F1 확인
python validate.py \
  --db data/monopoly.sqlite \
  --frames data/frames \
  --wav data/wav \
  --ckpt checkpoints/best.pt \
  --split val

# 테스트 세트 전체 평가
python validate.py --split test
precision, recall, f1 가 JSON 형식으로 출력됩니다.

--max_samples 300 옵션으로 빠른 디버깅도 가능합니다.
"""