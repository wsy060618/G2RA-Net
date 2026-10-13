from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import yaml

from .dataset import VolumeWindowDataset
from .hausdorff_dt import HausdorffDTLossAdapter
from .losses import CompositeSegmentationLoss
from .models import SliceRelUNet


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_model(config: Mapping[str, Any]) -> SliceRelUNet:
    return SliceRelUNet(**dict(config["model"]))


def build_dataset(config: Mapping[str, Any], split: str) -> VolumeWindowDataset:
    model_config = config["model"]
    data_config = config["data"]
    return VolumeWindowDataset(
        Path(data_config["manifest"]),
        split,
        int(model_config["slice_count"]),
        slice_axis=int(data_config.get("slice_axis", 2)),
        group_stride=int(data_config.get("group_stride", 1)),
        target_size=data_config.get("target_size", (224, 224)),
        normalization=str(data_config.get("normalization", "minmax")),
        cache_volumes=int(data_config.get("cache_volumes", 2)),
    )


def build_loss(config: Mapping[str, Any]) -> CompositeSegmentationLoss:
    num_classes = int(config["model"]["num_classes"])
    loss_config = config["loss"]
    hausdorff = HausdorffDTLossAdapter(
        num_classes,
        package_version="1.5.2",
        alpha=float(loss_config.get("hausdorff_dt_alpha", 2.0)),
        include_background=bool(loss_config.get("hausdorff_dt_include_background", False)),
        force_float32=True,
    )
    return CompositeSegmentationLoss(
        num_classes,
        ce_weight=float(loss_config.get("ce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        hausdorff_dt_weight=float(loss_config.get("hausdorff_dt_weight", 0.1)),
        hausdorff_dt=hausdorff,
        dice_include_background=bool(loss_config.get("dice_include_background", False)),
    )


def foreground_dice(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    prediction = logits.argmax(dim=2)
    scores = []
    for class_id in range(1, num_classes):
        predicted = prediction == class_id
        expected = target == class_id
        denominator = predicted.sum() + expected.sum()
        if expected.any():
            scores.append((2.0 * (predicted & expected).sum() / denominator).item())
    return float(np.mean(scores)) if scores else float("nan")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
