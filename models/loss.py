import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

_NUMERIC_DTYPES = {
    torch.float16, torch.float32, torch.float64, torch.bfloat16,
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool,
}

class OnsetsAndFramesLoss(nn.Module):
    def __init__(
        self,
        lambda_on: float = 1.0,
        lambda_frame: float = 1.0,
        lambda_off: float = 0.5,     # set 0.0 if no offset head
        lambda_vel: float = 0.0,     # >0 if regressing velocity
        onset_emphasis: float = 2.0,
        pos_weight_on:  Any | None = None,
        pos_weight_frame: Any | None = None,
        pos_weight_off:  Any | None = None,
    ):
        super().__init__()
        self.lambda_on    = lambda_on
        self.lambda_frame = lambda_frame
        self.lambda_off   = lambda_off
        self.lambda_vel   = lambda_vel
        self.onset_emphasis = onset_emphasis

        # Store pos-weights as tensors (float32); cast to logits dtype/device at forward.
        self.pw_on  = self._sanitize_pos_weight(pos_weight_on,   "pos_weight_on")
        self.pw_frm = self._sanitize_pos_weight(pos_weight_frame,"pos_weight_frame")
        self.pw_off = self._sanitize_pos_weight(pos_weight_off,  "pos_weight_off")

        self.mse_vel = nn.MSELoss(reduction="none")

    @staticmethod
    def _sanitize_pos_weight(pw, name: str):
        if pw is None:
            return None
        if not torch.is_tensor(pw):
            pw = torch.as_tensor(pw)
        if pw.dtype not in _NUMERIC_DTYPES:
            raise TypeError(f"{name} must be numeric tensor; got dtype={pw.dtype}")
        if pw.ndim != 1:
            raise ValueError(f"{name} must be 1D [P]; got shape={tuple(pw.shape)}")
        return pw.to(torch.float32)

    @staticmethod
    def _need_numeric_tensor(name: str, t, like: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            raise TypeError(f"target['{name}'] must be a torch.Tensor, got {type(t)}")
        if t.dtype not in _NUMERIC_DTYPES:
            raise TypeError(f"target['{name}'] must be numeric tensor, got dtype={t.dtype}")
        # allow per-sample [T,P] when logits are [B,T,P]
        if t.dim() == 2 and like.dim() == 3:
            t = t.unsqueeze(0)
        t = t.to(dtype=like.dtype, device=like.device)
        if t.shape != like.shape:
            raise ValueError(f"{name}: target shape {tuple(t.shape)} must match logits {tuple(like.shape)}")
        return t

    @staticmethod
    def _pw_like(pw: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
        if pw is None:
            return None
        if pw.numel() != like.size(-1):
            raise ValueError(f"pos_weight length {pw.numel()} must match P={like.size(-1)}")
        return pw.to(dtype=like.dtype, device=like.device)

    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        on_logit    = pred["on"]            # [B,T,P] logits
        frame_logit = pred["frame"]         # [B,T,P] logits
        off_logit   = pred.get("off", None) # [B,T,P] logits or None
        vel_pred    = pred.get("vel", None) # [B,T,P] or [B,T] if you add it

        # Validate/align targets (will give a precise error if anything is not a numeric tensor)
        on_t    = self._need_numeric_tensor("on",    target["on"],    on_logit)
        frame_t = self._need_numeric_tensor("frame", target["frame"], frame_logit)
        off_t   = None
        if off_logit is not None and "off" in target and self.lambda_off > 0:
            off_t = self._need_numeric_tensor("off", target["off"], off_logit)

        # pos-weights on the right device/dtype for each head
        pw_on  = self._pw_like(self.pw_on,  on_logit)
        pw_frm = self._pw_like(self.pw_frm, frame_logit)
        pw_off = self._pw_like(self.pw_off, off_logit) if off_logit is not None else None

        # --- Onset loss ---
        lon_map = F.binary_cross_entropy_with_logits(on_logit, on_t, pos_weight=pw_on, reduction="none")
        lon = lon_map.mean()

        # --- Frame loss with onset emphasis ---
        lfr_map = F.binary_cross_entropy_with_logits(frame_logit, frame_t, pos_weight=pw_frm, reduction="none")
        w = 1.0 + self.onset_emphasis * on_t
        lfr = (lfr_map * w).sum() / w.sum().clamp_min(1.0)

        # --- Offset loss (optional) ---
        if off_logit is not None and off_t is not None and self.lambda_off > 0:
            lof_map = F.binary_cross_entropy_with_logits(off_logit, off_t, pos_weight=pw_off, reduction="none")
            lof = lof_map.mean()
        else:
            lof = on_logit.new_tensor(0.0)

        # --- Velocity regression (optional) ---
        if vel_pred is not None and "vel" in target and self.lambda_vel > 0:
            vel_t = self._need_numeric_tensor("vel", target["vel"], vel_pred)
            m = on_t
            lvel = (self.mse_vel(vel_pred, vel_t) * m).sum() / m.sum().clamp_min(1.0)
        else:
            lvel = on_logit.new_tensor(0.0)

        loss = self.lambda_on * lon + self.lambda_frame * lfr + self.lambda_off * lof + self.lambda_vel * lvel
        return loss, {"on": lon.detach(), "frame": lfr.detach(), "off": lof.detach(), "vel": lvel.detach()}
