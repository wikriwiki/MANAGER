# eval.py
"""
테스트 세트 최종 평가 스크립트
  • checkpoints/best.pt 로드
  • test split에 대해 precision / recall / F1 / ROC-AUC 계산
  • 결과를 results_test.csv (video_id, person, label, pred, prob) 로 저장
"""
import argparse, csv, json, torch
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
)

from data.person_dataset import VideoPersonDataset
from models.manager_graphtokens import GraphTokenManager

# ────────────────────────────────────────────────
@torch.no_grad()
def run_test(loader, model, device, out_csv: Path):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "person", "label", "pred", "prob"])

        for sample in tqdm(loader, desc="test infer"):
            g = sample["graph"].to(device)
            logit,_ = model(g, sample["person"][0])
            prob = torch.sigmoid(logit).item()
            pred = int(prob > 0.5)
            label= int(sample["label"].item())

            writer.writerow([
                sample["graph"].node_meta.get("video_id","NA"),  # ← GraphBuilder에 video_id 넣었다면
                sample["person"][0],
                label, pred, prob
            ])

            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)

    p,r,f1,_ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_true, y_prob)
    return {"precision": p, "recall": r, "f1": f1, "roc_auc": roc}

# ────────────────────────────────────────────────
def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.gcn.load_state_dict(ckpt["gcn"])
    model.proj_up.load_state_dict(ckpt["proj"])
    model.glm.load_state_dict(ckpt["lora"], strict=False)
    print("checkpoint loaded")

# ────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda")

    # DataLoader (test split 고정)
    test_ds = VideoPersonDataset(
        db_path=args.db, split="test",
        cache_dir=args.cache, video_root=args.frames, audio_root=args.wav,
        merge_dialog=True, max_samples=args.max_samples, seed=args.seed
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"test samples: {len(test_ds)}")

    # Model
    model = GraphTokenManager().half().to(device)
    load_ckpt(model, args.ckpt)

    # Run evaluation
    out_csv = Path(args.output_csv)
    metrics = run_test(test_loader, model, device, out_csv)
    print(json.dumps(metrics, indent=2))

    # save metrics JSON
    Path(args.output_json).write_text(json.dumps(metrics, indent=2))
    print(f"results saved to {out_csv} & {args.output_json}")

# ────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",      default="data/monopoly.sqlite")
    ap.add_argument("--cache",   default="cache/")
    ap.add_argument("--frames",  default="data/frames/")
    ap.add_argument("--wav",     default="data/wav/")
    ap.add_argument("--ckpt",    default="checkpoints/best.pt")
    ap.add_argument("--output_csv",  default="results_test.csv")
    ap.add_argument("--output_json", default="metrics_test.json")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
