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
        onset_emphasis: float = 2.0,  # Not used anymore, kept for compatibility
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
        lon = F.binary_cross_entropy_with_logits(on_logit, on_t, pos_weight=pw_on, reduction="mean")

        # --- Frame loss (simplified - remove complex onset emphasis weighting) ---
        lfr = F.binary_cross_entropy_with_logits(frame_logit, frame_t, pos_weight=pw_frm, reduction="mean")

        # --- Offset loss (optional) ---
        if off_logit is not None and off_t is not None and self.lambda_off > 0:
            lof = F.binary_cross_entropy_with_logits(off_logit, off_t, pos_weight=pw_off, reduction="mean")
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


class OnsetsAndFramesPaperLoss(nn.Module):
    """
    Loss function matching the original Onsets and Frames paper more closely.

    Key features:
    - Standard BCE for onset loss
    - Weighted BCE for frame loss, emphasizing early active frames after each onset
    - Optional BCE for offset loss (set lambda_off > 0 to enable)
    - No velocity loss (to match paper focus)

    The weighted frame loss approximates the paper's idea of emphasizing note beginnings
    by applying higher loss weight to frame predictions in the first few timesteps
    after each detected onset, but only where the frame target is actually active.
    This encourages the model to predict note starts more accurately without
    over-penalizing sustained note frames.

    Example usage:
        criterion = OnsetsAndFramesPaperLoss(lambda_on=1.0, lambda_frame=1.0,
                                             lambda_off=0.5, onset_frame_weight=5.0, onset_window=4)
        loss, metrics = criterion(pred, target)
    """

    def __init__(self, lambda_on: float = 1.0, lambda_frame: float = 1.0,
                 lambda_off: float = 0.0, onset_frame_weight: float = 5.0, onset_window: int = 4):
        super().__init__()
        self.lambda_on = lambda_on
        self.lambda_frame = lambda_frame
        self.lambda_off = lambda_off
        self.onset_frame_weight = onset_frame_weight
        self.onset_window = onset_window

    @staticmethod
    def _need_numeric_tensor(name: str, t, like: torch.Tensor) -> torch.Tensor:
        """Validate and align target tensor to match logits shape/dtype/device."""
        if not torch.is_tensor(t):
            raise TypeError(f"target['{name}'] must be a torch.Tensor, got {type(t)}")
        if t.dtype not in _NUMERIC_DTYPES:
            raise TypeError(f"target['{name}'] must be numeric tensor, got dtype={t.dtype}")
        # Allow per-sample [T,P] when logits are [B,T,P]
        if t.dim() == 2 and like.dim() == 3:
            t = t.unsqueeze(0)
        t = t.to(dtype=like.dtype, device=like.device)
        if t.shape != like.shape:
            raise ValueError(f"{name}: target shape {tuple(t.shape)} must match logits {tuple(like.shape)}")
        return t

    @staticmethod
    def _build_frame_weight_map(on_target: torch.Tensor, frame_target: torch.Tensor,
                                onset_frame_weight: float, onset_window: int) -> torch.Tensor:
        """
        Build per-element weight map for frame loss.

        Weights are set to onset_frame_weight for active frame targets in the
        onset_window timesteps following each onset, and 1.0 elsewhere.
        This emphasizes early note frames to better match the paper's approach.

        Args:
            on_target: [B, T, P] onset targets (0/1)
            frame_target: [B, T, P] frame targets (0/1)
            onset_frame_weight: weight multiplier for emphasized frames
            onset_window: number of timesteps after onset to emphasize

        Returns:
            weight: [B, T, P] weight tensor
        """
        B, T, P = on_target.shape
        weight = torch.ones_like(frame_target)

        # Vectorized approach: create all shifted onset masks
        onset_mask = (on_target == 1.0)  # [B, T, P]

        # Create shifted versions for all offsets at once
        offsets = torch.arange(onset_window, device=onset_mask.device, dtype=torch.long)
        shifted_onsets = []
        for offset in offsets:
            shifted = torch.roll(onset_mask, shifts=int(offset), dims=1)
            shifted[:, :int(offset), :] = 0  # Zero invalid parts
            shifted_onsets.append(shifted)
        shifted_onsets = torch.stack(shifted_onsets, dim=0)  # [onset_window, B, T, P]

        # Combine: any shift that has onset and frame is active gets weight
        combined_mask = shifted_onsets.any(dim=0) & (frame_target == 1.0)  # [B, T, P]
        weight = torch.where(combined_mask, torch.tensor(onset_frame_weight, device=weight.device, dtype=weight.dtype), weight)

        return weight

    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss.

        Args:
            pred: dict with "on", "frame", and optionally "off" logits, each [B, T, P]
            target: dict with "on", "frame", and optionally "off" targets, each [B, T, P] or [T, P]

        Returns:
            total_loss: scalar tensor
            metrics: dict with "on", "frame", and optionally "off" detached losses
        """
        on_logit = pred["on"]      # [B, T, P]
        frame_logit = pred["frame"]  # [B, T, P]
        off_logit = pred.get("off", None)  # [B, T, P] or None

        # Validate and align targets
        on_t = self._need_numeric_tensor("on", target["on"], on_logit)
        frame_t = self._need_numeric_tensor("frame", target["frame"], frame_logit)
        off_t = None
        if off_logit is not None and "off" in target and self.lambda_off > 0:
            off_t = self._need_numeric_tensor("off", target["off"], off_logit)

        # Onset loss: standard BCE
        onset_loss = F.binary_cross_entropy_with_logits(on_logit, on_t, reduction="mean")

        # Build frame weight map based on onsets and active frames
        weight = self._build_frame_weight_map(on_t, frame_t, self.onset_frame_weight, self.onset_window)

        # Weighted frame loss: apply weights to unreduced BCE, then mean
        frame_bce_map = F.binary_cross_entropy_with_logits(frame_logit, frame_t, reduction="none")
        weighted_frame_loss = (frame_bce_map * weight).mean()

        # Offset loss: standard BCE (optional)
        if off_logit is not None and off_t is not None and self.lambda_off > 0:
            offset_loss = F.binary_cross_entropy_with_logits(off_logit, off_t, reduction="mean")
        else:
            offset_loss = on_logit.new_tensor(0.0)

        # Total loss
        total_loss = (self.lambda_on * onset_loss +
                     self.lambda_frame * weighted_frame_loss +
                     self.lambda_off * offset_loss)

        metrics = {"on": onset_loss.detach(), "frame": weighted_frame_loss.detach()}
        if self.lambda_off > 0:
            metrics["off"] = offset_loss.detach()

        return total_loss, metrics
