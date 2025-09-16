# text2glm.py
# ─────────────────────────────────────────────────────────
# Text-only 분류 (video_id 전체 텍스트 + (video_id, person) 라벨)
# - DB: data/speech_segments.db
# - 샘플: (video_id, person)쌍
# - 분할: 7 : 1 : 2
# - 백본(ChatGLM2) 고정 + no_grad (헤드만 CPU 학습)  또는  LoRA만 학습(옵션)
# - 검증에서 "정확도" 최대 임계값 → 테스트 적용
# - 학습/검증/테스트 예측 CSV 저장(프롬프트 포함)
# - 테스트 종료 후 다양한 지표 출력(ACC/Prec/Rec/F1/ROC-AUC/PR-AUC/Confusion)
# - 프롬프트 템플릿(영문) 적용 + 콘솔 미리보기
# - SAFETY 수축 + 멀티GPU 장치 정합 + CUDA 컨텍스트 워밍업
# - 헤드 연산 float32 + logits clamp([-30, 30]) + tqdm 진행 바
# - ★ FIX: last_hidden_state가 [B,S,H]/[B,H]/[S,H] 어떤 형상이어도 안전한 풀링
# ─────────────────────────────────────────────────────────

import os
import json
import sqlite3
import random
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# tqdm (없으면 폴백)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, *args, **kwargs):
        return x if x is not None else range(0)

# =========================
# 설정
# =========================
DB_PATH    = "data/speech_segments.db"
MODEL_ID   = "THUDM/chatglm2-6b"
MAX_LEN    = 1024           # 메모리 빠듯하면 512/384/256으로 낮추세요
BATCH_SIZE = 1
LR         = 2e-5
EPOCHS     = 5
SEED       = 42

# 학습 모드
FREEZE_BACKBONE = True      # LoRA 모드에서도 베이스는 고정
USE_LORA        = True      # LoRA 사용: True / 헤드만 학습: False
LORA_R          = 8

# 헤드 전용 학습 모드에서만 사용 (LoRA=False일 때)
HEAD_DEVICE     = "cpu"     # 분류헤드/Optimizer를 CPU로 (GPU OOM 방지)

# 출력/로그
OUT_DIR             = "out_text2glm"
PREVIEW_PROMPT_N    = 3

# NaN/Overflow 가드
CLAMP_LOGITS_MIN    = -30.0
CLAMP_LOGITS_MAX    =  30.0
NAN_REPLACE         = dict(nan=0.0, posinf=1e4, neginf=-1e4)

random.seed(SEED)
torch.manual_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 유틸
# =========================
def to01(v):
    if isinstance(v, bool): return 1 if v else 0
    if isinstance(v, (int, float)): return 1 if float(v) >= 0.5 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1","true","t","yes","y","abnormal","anomaly","pos","positive"): return 1
        if s in ("0","false","f","no","n","normal","neg","negative"): return 0
        try: return 1 if float(s) >= 0.5 else 0
        except: return None
    return None

def split_7_1_2(pairs: List[Dict[str, Any]]):
    n = len(pairs)
    if n < 3:
        n_tr = max(1, int(n*0.8)); n_va = 0; n_te = n - n_tr
    else:
        n_tr = int(n*0.7); n_va = int(n*0.1); n_te = n - n_tr - n_va
        if n_tr == 0: n_tr = 1
        if n_va == 0 and n >= 3: n_va = 1
        n_te = n - n_tr - n_va
        if n_te == 0 and n >= 3:
            if n_tr > 1: n_tr -= 1
            elif n_va > 1: n_va -= 1
            n_te = n - n_tr - n_va
    return pairs[:n_tr], pairs[n_tr:n_tr+n_va], pairs[n_tr+n_va:]

def metrics_at_threshold(y_true, y_prob, thr=0.5):
    tp=tn=fp=fn=0
    for t,p in zip(y_true, y_prob):
        pred = 1.0 if p>=thr else 0.0
        if pred==1 and t==1: tp+=1
        elif pred==0 and t==0: tn+=1
        elif pred==1 and t==0: fp+=1
        else: fn+=1
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (tp+tn)/max(1,(tp+tn+fp+fn))
    return {"acc":acc,"prec":prec,"rec":rec,"f1":f1,"tp":tp,"tn":tn,"fp":fp,"fn":fn}

def find_best_threshold_acc(y_true, y_prob, show_progress: bool = False):
    best_thr, best_acc, best_m = 0.5, -1.0, None
    it = range(1, 100)
    if show_progress:
        it = tqdm(it, desc="Selecting threshold (ACC)", leave=False)
    for i in it:  # 0.01 ~ 0.99
        thr = i / 100.0
        m = metrics_at_threshold(y_true, y_prob, thr)
        if m["acc"] > best_acc:
            best_thr, best_acc, best_m = thr, m["acc"], m
    return best_thr, best_acc, best_m

# ROC-AUC 계산(수치 근사)
def _roc_curve_points(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    P = sum(1 for _,t in pairs if t==1)
    N = sum(1 for _,t in pairs if t==0)
    if P==0 or N==0: return None
    tp = fp = 0
    tpr_list = [0.0]; fpr_list = [0.0]; prev_score = None
    for score, t in pairs:
        if prev_score is None or score != prev_score:
            tpr_list.append(tp / P); fpr_list.append(fp / N); prev_score = score
        if t==1: tp += 1
        else: fp += 1
    tpr_list.append(tp / P); fpr_list.append(fp / N)
    return fpr_list, tpr_list

def roc_auc(y_true, y_score):
    pts = _roc_curve_points(y_true, y_score)
    if pts is None: return None
    fpr, tpr = pts
    auc = 0.0
    for i in range(1, len(fpr)):
        dx = fpr[i] - fpr[i-1]
        auc += dx * (tpr[i] + tpr[i-1]) / 2.0
    return max(0.0, min(1.0, auc))

def pr_auc(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    P = sum(1 for _,t in pairs if t==1)
    if P==0: return None
    tp = fp = 0; prev_recall = 0.0; ap = 0.0
    for score, t in pairs:
        if t==1: tp += 1
        else: fp += 1
        prec = tp / (tp + fp); rec  = tp / P
        ap += prec * max(0.0, rec - prev_recall); prev_recall = rec
    return max(0.0, min(1.0, ap))

# 프롬프트 템플릿(영문, 짧고 명확)
def build_prompt(person: str, script: str, video_id: str) -> str:
    person = (person or "").strip()
    script = (script or "").strip()
    prompt = (
        "Below is a YouTube video transcript about a specific person.\n"
        f"Person: {person}\n"
        "Transcript:\n"
        f"{script}\n\n"
        "Task: Predict whether this person's Google search volume will surge soon.\n"
        "Output a single label: 1 for surge, 0 for no surge."
    )
    return prompt

# =========================
# 데이터 적재
# =========================
def load_samples(db_path: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        seg = pd.read_sql_query("SELECT video_id, start_time, script FROM speech_segments", conn)
        vm  = pd.read_sql_query("SELECT video_id, persons_found FROM video_metadata", conn)

    seg = seg.dropna(subset=["script"])
    seg["video_id"] = seg["video_id"].astype(str)
    seg = seg.sort_values(["video_id","start_time"])

    texts_by_video = (
        seg.groupby("video_id")["script"]
           .apply(lambda s: "\n".join([x for x in s if isinstance(x,str) and x.strip()]))
           .to_dict()
    )

    samples: List[Dict[str, Any]] = []
    uid = 0
    for row in vm.itertuples(index=False):
        vid = str(getattr(row, "video_id"))
        raw = getattr(row, "persons_found")
        if vid not in texts_by_video: continue
        try:
            pf = json.loads(raw)
        except Exception:
            continue
        if not isinstance(pf, dict): continue

        video_text = (texts_by_video[vid] or "").strip()
        if not video_text: continue

        for person, lab in pf.items():
            lab01 = to01(lab)
            if lab01 is None: continue
            prompt = build_prompt(str(person), video_text, vid)
            samples.append({
                "id": uid,
                "video_id": vid,
                "person": str(person),
                "script": video_text,
                "prompt": prompt,
                "label": int(lab01),
            })
            uid += 1

    random.shuffle(samples)
    return samples

# =========================
# Dataset / collate
# =========================
class TextDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        text = it["prompt"]
        lab  = torch.tensor([float(it["label"])], dtype=torch.float)
        meta = {k: it[k] for k in ["id","video_id","person","label","prompt"]}
        return text, lab, meta

def collate(batch):
    texts, labels, metas = zip(*batch)
    return list(texts), torch.cat(labels, dim=0), list(metas)

# =========================
# 모델(헤드 전용 학습  또는  LoRA 학습)
# =========================
class GLMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.glm = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        # 입력 임베딩이 올라간 디바이스(초기 입력 투입용)
        self.base_device = self.glm.get_input_embeddings().weight.device

        # (옵션) LoRA — 메모리 여유 시
        self.use_lora = False
        if USE_LORA:
            try:
                from peft import LoraConfig, get_peft_model
                lcfg = LoraConfig(r=LORA_R, target_modules=["query_key_value"])
                self.glm = get_peft_model(self.glm, lcfg)
                self.use_lora = True
                print("[INFO] LoRA enabled.")
            except Exception as e:
                print("[WARN] LoRA 초기화 실패. 헤드 전용 학습으로 대체:", e)
                self.use_lora = False

        hidden = getattr(getattr(self.glm, "config", None), "hidden_size", 4096)
        if self.use_lora:
            # 헤드는 float32로, base_device(GPU)에 올려두고 forward에서 필요 시 장치만 맞춤
            self.cls = nn.Linear(hidden, 1).to(device=self.base_device, dtype=torch.float32)
        else:
            # 헤드 전용 학습: CPU float32
            self.cls = nn.Linear(hidden, 1).to(HEAD_DEVICE)

        # 가중치 고정 설정
        if FREEZE_BACKBONE:
            if self.use_lora:
                freeze_cnt = 0; train_cnt  = 0
                for n, p in self.glm.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True; train_cnt += p.numel()
                    else:
                        p.requires_grad = False; freeze_cnt += p.numel()
                print(f"[INFO] Backbone frozen ({freeze_cnt} params), LoRA trainable ({train_cnt} params).")
            else:
                for p in self.glm.parameters():
                    p.requires_grad = False
                print("[INFO] Backbone fully frozen (head-only training).")

        # 멀티GPU CUDA 컨텍스트 워밍업
        self._warmup_cuda_contexts()

    def _warmup_cuda_contexts(self):
        if not torch.cuda.is_available():
            return
        try:
            if isinstance(self.base_device, torch.device) and self.base_device.type == "cuda":
                torch.cuda.set_device(self.base_device)
            devices = set()
            for p in self.glm.parameters():
                if p.is_cuda:
                    devices.add(p.device)
            for d in sorted(devices, key=lambda x: (x.type, x.index)):
                if d.type == "cuda":
                    with torch.cuda.device(d):
                        _ = torch.empty(1, device=d)
        except Exception as e:
            print("[WARN] CUDA context warmup skipped:", e)

    def backbone(self):
        return getattr(self.glm, "transformer", None) or getattr(self.glm, "model", None) or self.glm

    def forward(self, input_ids, attention_mask, labels=None):
        # 1) 배치 차원 보장
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1: attention_mask = attention_mask.unsqueeze(0)

        B = attention_mask.size(0)

        # 2) 디바이스 정렬(백본 첫 디바이스로)
        input_ids = input_ids.to(self.base_device)
        attention_mask = attention_mask.to(self.base_device)

        # 3) 백본 순전파: LoRA 학습 시 그래프 필요 / 헤드 전용 학습 시 no_grad
        if self.use_lora:
            out = self.backbone()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                output_hidden_states=False,
            )
        else:
            with torch.no_grad():
                out = self.backbone()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                    output_hidden_states=False,
                )

        hs = out.last_hidden_state  # 기대: [B,S,H] 이지만 일부 환경/버전에 따라 [B,H] 또는 [S,H]로 나올 수 있음

        # 4) 견고한 풀링 → 항상 [B,H]를 얻는다
        if hs.dim() == 3:
            # 표준 경로: [B,S,H]
            mask = attention_mask.to(hs.device).unsqueeze(-1).to(hs.dtype)  # [B,S,1]
            summed = (hs * mask).sum(dim=1)                                 # [B,H]
            denom  = mask.sum(dim=1).clamp(min=1.0)                         # [B,1]
            pooled = summed / denom                                         # [B,H]
        elif hs.dim() == 2 and hs.size(0) == B:
            # 이미 [B,H] 형태 (모델이 마지막 토큰/평균을 반환하는 구현 케이스)
            pooled = hs                                                     # [B,H]
        elif hs.dim() == 2 and B == 1 and hs.size(0) == attention_mask.size(1):
            # [S,H] (배치=1) 형태: 마스크 평균 풀링 후 [1,H]
            m = attention_mask.to(hs.device).unsqueeze(0).unsqueeze(-1).to(hs.dtype)  # [1,S,1]
            pooled = (hs.unsqueeze(0) * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # [1,H]
        else:
            # 보수적 폴백: 토큰축 추정 후 평균
            try:
                H = hs.size(-1)
                pooled = hs.view(B, -1, H).mean(dim=1)                      # [B,H]
            except Exception:
                pooled = hs.mean(dim=-2) if hs.dim() >= 2 else hs           # 최대한 [H] → [1,H]
                if pooled.dim() == 1:
                    pooled = pooled.unsqueeze(0)

        # 안정화
        pooled = torch.nan_to_num(pooled, **NAN_REPLACE)                    # [B,H]

        # 5) 헤드 연산은 항상 float32로 (안정성)
        if self.use_lora:
            if (self.cls.weight.device != pooled.device) or (self.cls.weight.dtype != torch.float32):
                self.cls = self.cls.to(device=pooled.device, dtype=torch.float32)
            pooled32 = pooled.to(dtype=torch.float32)                       # [B,H] float32 (GPU)
            logits = self.cls(pooled32).squeeze(-1)                         # [B] float32
        else:
            pooled32_cpu = pooled.to(device=HEAD_DEVICE, dtype=torch.float32)
            logits = self.cls(pooled32_cpu).squeeze(-1)                     # [B] float32 (CPU)

        # ★ SAFETY: 만약 여전히 [B,S] 모양이 섞여 오면 마스크로 평균/수축
        if logits.dim() == 1 and logits.shape[0] != B:
            if B == 1 and logits.numel() == attention_mask.size(1):
                m = attention_mask[0].to(logits.device, logits.dtype)       # [S]
                logits = (logits * m).sum() / m.sum().clamp(min=1.0)
                logits = logits.view(1)
            else:
                S = attention_mask.size(1)
                if logits.numel() == B * S:
                    logits = logits.view(B, S)
                    m = attention_mask.to(logits.device, logits.dtype)
                    logits = (logits * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                else:
                    logits = logits.view(B, -1).mean(dim=1)

        if logits.dim() != 1:
            logits = logits.view(B, -1).mean(dim=1)

        # 6) 손실: float32 + clamp
        loss = None
        if labels is not None:
            labels_f32 = labels.view(-1).to(device=logits.device, dtype=torch.float32)
            logits_for_loss = torch.clamp(torch.nan_to_num(logits, **NAN_REPLACE),
                                          CLAMP_LOGITS_MIN, CLAMP_LOGITS_MAX)
            loss = nn.functional.binary_cross_entropy_with_logits(logits_for_loss, labels_f32)

        return logits, loss

# =========================
# CSV 저장 및 평가 유틸
# =========================
def save_csv(rows: List[Dict[str, Any]], path: str):
    df = pd.DataFrame(rows); df.to_csv(path, index=False, encoding="utf-8")
    print(f"💾 Saved CSV: {path} ({len(rows)} rows)")

def run_inference(split_name: str, model: GLMClassifier, loader: DataLoader, threshold: float) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for texts, y, metas in tqdm(loader, total=len(loader), desc=f"Inference[{split_name}]", unit="batch"):
            tok = model.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            ids, mask = tok.input_ids, tok.attention_mask
            logits, _ = model(ids, mask, labels=None)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            labels = y.view(-1).cpu().tolist()
            for prob, lab, meta, logit_val in zip(probs, labels, metas, logits.detach().cpu().tolist()):
                pred = 1 if prob >= threshold else 0
                rows.append({
                    "split": split_name,
                    "id": meta["id"],
                    "video_id": meta["video_id"],
                    "person": meta["person"],
                    "label": int(lab),
                    "logit": float(logit_val),
                    "prob": float(prob),
                    "pred": pred,
                    "threshold_used": threshold,
                    "prompt": meta["prompt"],
                })
    return rows

def preview_prompts(dataset: TextDataset, n: int = PREVIEW_PROMPT_N):
    print("\n===== Prompt preview (first {} samples) =====".format(min(n, len(dataset))))
    for i in range(min(n, len(dataset))):
        _, _, meta = dataset[i]
        print(f"[{i}] video_id={meta['video_id']} person={meta['person']}\n--- PROMPT START ---\n{meta['prompt']}\n--- PROMPT END ---\n")

def print_metrics(name: str, y_true: List[int], y_prob: List[float], thr: float):
    m = metrics_at_threshold(y_true, y_prob, thr)
    roc = roc_auc(y_true, y_prob); pra = pr_auc(y_true, y_prob)
    roc_str = f"{roc:.4f}" if roc is not None else "-"; pra_str = f"{pra:.4f}" if pra is not None else "-"
    print(
        f"[{name}] @thr={thr:.2f} "
        f"acc={m['acc']:.4f} prec={m['prec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} "
        f"roc_auc={roc_str} pr_auc={pra_str} "
        f"| cm(tp={m['tp']},tn={m['tn']},fp={m['fp']},fn={m['fn']})"
    )
    return m, roc, pra

# =========================
# Train / Eval
# =========================
def train():
    items = load_samples(DB_PATH)
    if len(items) == 0:
        raise RuntimeError("유효 (video 전체 텍스트, person 라벨) 샘플이 없습니다.")

    train_items, val_items, test_items = split_7_1_2(items)

    tr_ds, va_ds, te_ds = TextDataset(train_items), TextDataset(val_items), TextDataset(test_items)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, pin_memory=False)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, pin_memory=False)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, pin_memory=False)

    print(f"Samples: train={len(tr_ds)}, val={len(va_ds)}, test={len(te_ds)}")
    preview_prompts(tr_ds, PREVIEW_PROMPT_N)

    model = GLMClassifier()

    # 학습되는 파라미터만 옵티마이저에 전달
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar = tqdm(tr_ld, total=len(tr_ld), desc=f"Epoch {epoch} [train]", unit="batch")
        for texts, y, metas in pbar:
            tok = model.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            ids, mask = tok.input_ids, tok.attention_mask  # CPU 텐서

            optim.zero_grad(set_to_none=True)
            logits, loss = model(ids, mask, labels=y)

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss. Skipping step. (meta_id={metas[0]['id']}, vid={metas[0]['video_id']}, person={metas[0]['person']})")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optim.step()

            running += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        print(f"[Epoch {epoch}] train_loss={running / max(1, len(tr_ld)):.4f}")

    # ── 검증: 정확도 최대 임계값 선택
    model.eval()
    yv_true, yv_prob = [], []
    with torch.no_grad():
        for texts, y, _metas in tqdm(va_ld, total=len(va_ld), desc="Validating", unit="batch"):
            tok = model.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            ids, mask = tok.input_ids, tok.attention_mask
            logits, _ = model(ids, mask, labels=None)
            probs = torch.sigmoid(logits).cpu().tolist()
            yv_prob.extend(probs); yv_true.extend(y.view(-1).tolist())

    if len(yv_true) > 0:
        base_thr = 0.5
        print_metrics("VAL(base)", yv_true, yv_prob, base_thr)
        best_thr, _best_acc, _best_m = find_best_threshold_acc(yv_true, yv_prob, show_progress=True)
        print_metrics("VAL(best-acc)", yv_true, yv_prob, best_thr)
    else:
        best_thr = 0.5
        print("[VAL] 유효 검증 샘플이 없어 임계값 0.5 사용")

    # ── 전체 split에 대해 예측 CSV 저장
    train_rows = run_inference("train", model, tr_ld, best_thr)
    val_rows   = run_inference("val",   model, va_ld, best_thr)
    test_rows  = run_inference("test",  model, te_ld, best_thr)

    save_csv(train_rows, os.path.join(OUT_DIR, "train_out.csv"))
    save_csv(val_rows,   os.path.join(OUT_DIR, "val_out.csv"))
    save_csv(test_rows,  os.path.join(OUT_DIR, "test_out.csv"))

    # ── 테스트 지표 출력 (여러 지표)
    if len(test_rows) > 0:
        yt_true = [r["label"] for r in test_rows]
        yt_prob = [r["prob"]  for r in test_rows]
        print_metrics("TEST", yt_true, yt_prob, best_thr)
    else:
        print("[TEST] 유효 테스트 샘플 없음.")

    # 모델 + 임계값 저장
    torch.save({"model": model.state_dict(), "best_threshold": best_thr, "use_lora": USE_LORA},
               os.path.join(OUT_DIR, "glm_text_only_vid_person.pt"))
    print(f"✅ Saved: {os.path.join(OUT_DIR, 'glm_text_only_vid_person.pt')} (best_threshold={best_thr:.2f}, use_lora={USE_LORA})")

if __name__ == "__main__":
    train()
