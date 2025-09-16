# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBinaryLoss(nn.Module):
    """
    Binary Focal Loss
    - gamma: hard sample 강조(1.5 ~ 2.0 권장 시작)
    - alpha: 클래스 가중치(균형 5:5면 None 권장; 필요시 0.5)
    """
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B], targets: [B] {0,1}
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p   = torch.sigmoid(logits)
        pt  = targets * p + (1 - targets) * (1 - p)  # 정답확률
        modulating = (1 - pt).clamp_(self.eps, 1 - self.eps) ** self.gamma
        loss = modulating * bce
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AMBinaryCE(nn.Module):
    """
    Additive-Margin BCE
    - margin m을 로짓에 직접 부여해 결정경계 바깥으로 밀어냄
    - m: 0.1 ~ 0.4 권장 시작
    """
    def __init__(self, margin: float = 0.2, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 정답(1)에는 logit - m, 오답(0)에는 logit + m → 마진 강제
        z_m = torch.where(targets > 0.5, logits - self.margin, logits + self.margin)
        return F.binary_cross_entropy_with_logits(z_m, targets, reduction=self.reduction)


class AMFocalBinaryLoss(nn.Module):
    """
    AM-Focal (권장 기본: margin 적용 후 Focal 적용)
    - combine_mode="margin_then_focal": z'에 focal 적용 (추천)
    - combine_mode="sum": focal(z) + lam * am_bce(z)
    """
    def __init__(self,
                 gamma: float = 1.8,
                 margin: float = 0.2,
                 alpha: float | None = None,
                 lam: float = 0.3,
                 combine_mode: str = "margin_then_focal",
                 reduction: str = "mean"):
        super().__init__()
        self.focal = FocalBinaryLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.am    = AMBinaryCE(margin=margin, reduction=reduction)
        self.lam   = lam
        self.combine_mode = combine_mode
        self.reduction = reduction
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.combine_mode == "margin_then_focal":
            z_m = torch.where(targets > 0.5, logits - self.margin, logits + self.margin)
            return self.focal(z_m, targets)
        # 대안: 합성 방식 (가중합)
        return self.focal(logits, targets) + self.lam * self.am(logits, targets)
