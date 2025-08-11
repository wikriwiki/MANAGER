import torch
import gc
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm.auto import tqdm

# [BUG FIX] PyTorch의 DataLoader 대신 torch_geometric의 DataLoader를 임포트합니다.
from torch_geometric.loader import DataLoader
from torch.cuda.amp import GradScaler, autocast

# ===============================================================
# 1. 초기 설정: 로깅 및 프로젝트 경로
# ===============================================================

def setup_logging():
    """실행 로그를 파일과 콘솔에 출력하도록 설정합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("experiment_simple.log"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from dataset.data import VideoPersonDataset
    from models.manager_graphtokens import GraphTokenManager
    from experiments import init_experiment_db, insert_new_experiment, update_experiment_metrics, insert_sample_predictions
    from train import seed_all
    from validate import run_eval
    from eval import run_test, load_ckpt
except ImportError as e:
    logging.exception(f"필수 모듈 임포트 실패: {e}")
    sys.exit(1)


# ===============================================================
# 2. 메인 실행 함수
# ===============================================================

def main(args):
    setup_logging()

    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    logging.info(f"단일 장치 모드. 사용 장치: {device}")

    seed_all(args.seed)
    logging.info(f"랜덤 시드 고정: {args.seed}")

    exp_conn = init_experiment_db(db_path=args.exp_db)
    experiment_id = insert_new_experiment(exp_conn, args)
    logging.info(f"실험 DB '{args.exp_db}' 초기화 완료. 실험 ID: {experiment_id}")

    dataset_args = {
        'db_path': args.db, 'cache_dir': args.cache,
        'max_samples': args.max_samples, 'seed': args.seed,
        'filter_ids': None
    }

    train_dataset = VideoPersonDataset(split='train', **dataset_args)
    val_dataset = VideoPersonDataset(split='val', **dataset_args)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    logging.info(f"데이터 로드 완료. 훈련 샘플: {len(train_dataset)}, 검증 샘플: {len(val_dataset)}")

    model = GraphTokenManager().to(device)
    logging.info("모델 로딩 및 장치 이동 완료.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    use_amp = (device.type == 'cuda')
    scaler = GradScaler() if use_amp else None

    best_f1 = 0.0
    best_ckpt_path = None
    Path(args.ckpt_dir).mkdir(exist_ok=True)

    logging.info("모델 훈련을 시작합니다...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (Train)")
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_data in pbar:
            if batch_data is None:
                continue

            try:
                g = batch_data.to(device)
                lbl = g.label
                person = g.person[0]
            except Exception as e:
                logging.warning(f"데이터 장치 이동 중 오류: {e}. 이 배치를 건너뜁니다.")
                continue

            with autocast(enabled=use_amp):
                logit, loss = model(g, person, lbl)

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Epoch {epoch}: 유효하지 않은 Loss 발생. 스텝을 건너뜁니다.")
                optimizer.zero_grad()
                continue

            loss = loss / args.accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * args.accumulation_steps

            if (pbar.n + 1) % args.accumulation_steps == 0 or (pbar.n + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(avg_loss=total_loss / (pbar.n + 1))

            del g, lbl, person, logit, loss, batch_data
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        logging.info(f"Epoch {epoch} 검증을 시작합니다...")
        val_metrics = run_eval(val_loader, model, device)
        logging.info(f"Epoch {epoch} Val Metrics: {json.dumps(val_metrics, indent=2)}")
        update_experiment_metrics(exp_conn, experiment_id, "val", epoch, val_metrics)

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_ckpt_path = Path(args.ckpt_dir) / f"exp{experiment_id}_best_model.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            logging.info(f"  ✔ 새로운 최적 체크포인트 저장됨 (Epoch {epoch}, F1: {best_f1:.4f}) -> '{best_ckpt_path}'")
            update_experiment_metrics(exp_conn, experiment_id, "best_val", epoch, val_metrics, str(best_ckpt_path))

    logging.info("모델 훈련 완료.")

    if best_ckpt_path and best_ckpt_path.exists():
        logging.info(f"\n테스트 세트 평가 시작... (체크포인트: {best_ckpt_path})")
        test_model = GraphTokenManager().to(device)
        test_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        logging.info("테스트용 모델에 최적 체크포인트 로드 완료.")

        test_dataset = VideoPersonDataset(split="test", **dataset_args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        logging.info(f"테스트 샘플: {len(test_dataset)}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        final_metrics, individual_predictions = run_test(test_loader, test_model, device, output_dir / f"exp{experiment_id}_predictions.csv")
        logging.info(f"\n최종 테스트 메트릭: {json.dumps(final_metrics, indent=2)}")

        update_experiment_metrics(exp_conn, experiment_id, "test", -1, final_metrics)
        insert_sample_predictions(exp_conn, experiment_id, individual_predictions, "test")
        logging.info("최종 테스트 메트릭 및 개별 예측 결과가 DB에 업데이트되었습니다.")

        metric_path = output_dir / f"exp{experiment_id}_metrics.json"
        metric_path.write_text(json.dumps(final_metrics, indent=4))
        logging.info(f"최종 결과가 {output_dir} 디렉토리에 저장되었습니다.")
    else:
        logging.warning("훈련 중 최적 모델이 저장되지 않았습니다. 테스트 평가를 건너뜁니다.")

    exp_conn.close()
    logging.info("모든 실험 과정 완료.")

# ===============================================================
# 3. 인자 파서 및 실행 로직
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MANAGER 모델 단일 GPU 훈련 및 평가 스크립트")
    parser.add_argument("--db", type=str, default="data/speech_segments.db")
    parser.add_argument("--cache", type=str, default="cache/")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_simple/")
    parser.add_argument("--output_dir", type=str, default="results_simple/")
    parser.add_argument("--exp_db", type=str, default="experiment_simple.db")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # [BUG FIX] 누락되었던 --gpu_id 인자를 다시 추가합니다.
    parser.add_argument("--gpu_id", type=int, default=0, help="사용할 GPU의 ID 번호 (0, 1, 2, ...)")
    
    # 호환성을 위한 더미 인자들 (dataset.py가 **kwargs로 받으므로 실제 사용되지는 않음)
    parser.add_argument("--frames", type=str, default="data/frames/")
    parser.add_argument("--wav", type=str, default="data/wav/")
    parser.add_argument("--merge_dialog", action='store_true')
    
    args = parser.parse_args()
    main(args)