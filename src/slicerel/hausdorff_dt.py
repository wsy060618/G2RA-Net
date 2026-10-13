"""Differentiable Hausdorff distance-transform training loss.

This module deliberately does not implement the hard HD95 evaluation metric.
Each slice is treated as an independent two-dimensional distance-map domain.
"""

from __future__ import annotations

from typing import Optional

import monai
import numpy as np
import torch
from monai.losses import HausdorffDTLoss
from scipy.ndimage import distance_transform_edt
from torch import nn


class HausdorffDTLossAdapter(nn.Module):
    """Reproducible wrapper around MONAI's distance-transform loss."""

    def __init__(
        self,
        num_classes: int,
        *,
        package_version: str = "1.5.2",
        alpha: float = 2.0,
        include_background: bool = False,
        reduction: str = "mean",
        batch: bool = False,
        force_float32: bool = True,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if package_version != monai.__version__:
            raise RuntimeError(
                "MONAI package version mismatch: "
                f"configured {package_version!r}, installed {monai.__version__!r}."
            )
        if ignore_index is not None:
            raise ValueError(
                "ignore_index is unsupported by HausdorffDTLoss because its distance "
                "maps cannot mask ignored pixels reproducibly."
            )
        if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

        self.num_classes = int(num_classes)
        self.package_version = package_version
        self.alpha = float(alpha)
        self.include_background = bool(include_background)
        self.reduction = reduction
        self.batch = bool(batch)
        self.force_float32 = bool(force_float32)
        self.ignore_index = ignore_index
        self.loss = HausdorffDTLoss(
            alpha=self.alpha,
            include_background=self.include_background,
            to_onehot_y=True,
            sigmoid=False,
            softmax=True,
            reduction=self.reduction,
            batch=self.batch,
        )
        # Keep MONAI's loss formula while selecting one explicit, reproducible
        # EDT backend. MONAI's dynamic cuCIM optional-import fallback creates
        # traceback reference cycles on CUDA when cuCIM is unavailable.
        self.loss.distance_field = self._scipy_distance_field

    @staticmethod
    @torch.no_grad()
    def _scipy_distance_field(image: torch.Tensor) -> torch.Tensor:
        image_array = image.detach().float().cpu().numpy()
        field_array = np.zeros(image_array.shape, dtype=np.float32)
        for batch_index in range(len(image_array)):
            foreground = image_array[batch_index] > 0.5
            if foreground.any() and not foreground.all():
                field_array[batch_index] = distance_transform_edt(
                    foreground
                ) + distance_transform_edt(~foreground)
        return torch.as_tensor(field_array, device=image.device, dtype=image.dtype)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        flat_logits, flat_target = self._flatten_slices(logits, target)
        if flat_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Logit class axis has {flat_logits.shape[1]} channels, expected "
                f"num_classes={self.num_classes}."
            )
        if not target.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("target must contain integer class indices.")
        if target.numel() and (target.min() < 0 or target.max() >= self.num_classes):
            raise ValueError("Target class is outside [0, num_classes).")

        # Casting is differentiable. Keeping it inside an explicitly disabled
        # autocast region prevents the distance-transform term from silently
        # changing precision when the rest of training uses AMP.
        device_type = flat_logits.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            loss_logits = flat_logits.float() if self.force_float32 else flat_logits
            return self.loss(loss_logits, flat_target[:, None].long())

    @staticmethod
    def _flatten_slices(
        logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim == 4:
            if target.ndim != 3:
                raise ValueError("logits [N,K,H,W] require target [N,H,W].")
            flat_logits = logits
            flat_target = target
        elif logits.ndim == 5:
            if target.ndim != 4:
                raise ValueError("logits [B,S,K,H,W] require target [B,S,H,W].")
            if logits.shape[:2] != target.shape[:2]:
                raise ValueError(
                    "Logit/target batch/slice axes must match exactly for grouped input."
                )
            flat_logits = logits.reshape(-1, logits.shape[2], *logits.shape[-2:])
            flat_target = target.reshape(-1, *target.shape[-2:])
        else:
            raise ValueError(
                "logits must be [N,K,H,W] or grouped [B,S,K,H,W]."
            )
        if flat_logits.shape[0] != flat_target.shape[0] or flat_logits.shape[-2:] != flat_target.shape[-2:]:
            raise ValueError("Logit/target spatial shape mismatch.")
        return flat_logits, flat_target

    def reproduction_metadata(self) -> dict[str, object]:
        """Return checkpoint-safe semantics for this training component."""
        return {
            "name": "hausdorff_distance_transform",
            "implementation": "monai.losses.HausdorffDTLoss",
            "package": "monai",
            "package_version": self.package_version,
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "include_background": self.include_background,
            "reduction": self.reduction,
            "batch": self.batch,
            "force_float32": self.force_float32,
            "to_onehot_y": True,
            "softmax": True,
            "sigmoid": False,
            "domain": "flattened_2d",
            "ignore_index": self.ignore_index,
            "distance_transform_backend": "explicit_scipy_cpu",
            "cuda_post_forward_cycle_gc": False,
        }
