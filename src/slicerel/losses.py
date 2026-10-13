from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .hausdorff_dt import HausdorffDTLossAdapter


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-5,
        ignore_index: Optional[int] = None,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("Dice expects logits [N,K,H,W] and target [N,H,W].")
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise ValueError("Logit/target spatial shape mismatch.")
        valid = torch.ones_like(target, dtype=torch.bool)
        safe_target = target
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            safe_target = target.masked_fill(~valid, 0)
        if safe_target.numel() and (safe_target.min() < 0 or safe_target.max() >= self.num_classes):
            raise ValueError("Target class is outside [0, num_classes).")

        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=probabilities.dtype)
        valid_mask = valid[:, None].to(dtype=probabilities.dtype)
        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class CompositeSegmentationLoss(nn.Module):
    """The thesis objective: weighted CE + Dice + HausdorffDT training proxy."""

    def __init__(
        self,
        num_classes: int,
        *,
        ce_weight: float,
        dice_weight: float,
        hausdorff_dt_weight: float,
        hausdorff_dt: HausdorffDTLossAdapter,
        ignore_index: Optional[int] = None,
        dice_include_background: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        weights = {
            "ce_weight": ce_weight,
            "dice_weight": dice_weight,
            "hausdorff_dt_weight": hausdorff_dt_weight,
        }
        for name, value in weights.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if ignore_index is not None:
            raise ValueError(
                "ignore_index is unsupported because HausdorffDT distance maps cannot "
                "apply the same ignored-pixel policy as CE and Dice."
            )
        if not isinstance(hausdorff_dt, nn.Module):
            raise TypeError("hausdorff_dt must be a torch.nn.Module.")

        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.hausdorff_dt_weight = float(hausdorff_dt_weight)
        self.ignore_index = ignore_index
        self.dice_include_background = bool(dice_include_background)
        self.dice = SoftDiceLoss(
            self.num_classes,
            ignore_index=ignore_index,
            include_background=self.dice_include_background,
        )
        self.hausdorff_dt = hausdorff_dt

    def _flatten_slices(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim == 4 and target.ndim == 3:
            flat_logits, flat_target = logits, target
        elif logits.ndim == 5 and target.ndim == 4:
            if logits.shape[:2] != target.shape[:2]:
                raise ValueError("Grouped logit/target batch and slice axes must match.")
            flat_logits = logits.reshape(-1, logits.shape[2], *logits.shape[-2:])
            flat_target = target.reshape(-1, *target.shape[-2:])
        else:
            raise ValueError(
                "Loss expects logits [N,K,H,W] with target [N,H,W], or grouped "
                "logits [B,S,K,H,W] with target [B,S,H,W]."
            )
        if flat_logits.shape[0] != flat_target.shape[0] or flat_logits.shape[-2:] != flat_target.shape[-2:]:
            raise ValueError("Logit/target spatial shape mismatch.")
        if flat_logits.shape[1] != self.num_classes:
            raise ValueError("Logit class axis does not match num_classes.")
        return flat_logits, flat_target

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat_logits, flat_target = self._flatten_slices(logits, target)
        ce = F.cross_entropy(flat_logits, flat_target)
        dice = self.dice(flat_logits, flat_target)
        hausdorff_dt = self.hausdorff_dt(flat_logits, flat_target)
        weighted_ce = self.ce_weight * ce
        weighted_dice = self.dice_weight * dice
        weighted_hausdorff_dt = self.hausdorff_dt_weight * hausdorff_dt
        total = weighted_ce + weighted_dice + weighted_hausdorff_dt
        return total, {
            "loss_total": total.detach(),
            "loss_ce": ce.detach(),
            "loss_dice": dice.detach(),
            "loss_hausdorff_dt": hausdorff_dt.detach(),
            "weighted_loss_ce": weighted_ce.detach(),
            "weighted_loss_dice": weighted_dice.detach(),
            "weighted_loss_hausdorff_dt": weighted_hausdorff_dt.detach(),
        }

    def reproduction_metadata(self) -> dict[str, object]:
        adapter_metadata = getattr(self.hausdorff_dt, "reproduction_metadata", None)
        return {
            "name": "ce_dice_hausdorff_dt",
            "num_classes": self.num_classes,
            "ce_weight": self.ce_weight,
            "dice_weight": self.dice_weight,
            "hausdorff_dt_weight": self.hausdorff_dt_weight,
            "ignore_index": self.ignore_index,
            "dice_include_background": self.dice_include_background,
            "hausdorff_dt": adapter_metadata() if callable(adapter_metadata) else None,
        }
