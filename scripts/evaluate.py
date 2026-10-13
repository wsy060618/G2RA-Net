from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from slicerel.experiment import (
    build_dataset,
    build_model,
    load_config,
    resolve_device,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SliceRel-U-Net")
    parser.add_argument("--config", type=Path, default=Path("configs/example.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    return parser.parse_args()


def dice_per_case(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    prediction = logits.argmax(dim=1)
    by_class = {}
    for class_id in range(1, num_classes):
        predicted = prediction == class_id
        expected = target == class_id
        denominator = predicted.sum() + expected.sum()
        if expected.any():
            by_class[str(class_id)] = float((2.0 * (predicted & expected).sum() / denominator).item())
    return by_class


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config.get("device", "auto")))
    dataset = build_dataset(config, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.assert_compatible_metadata(checkpoint["model_metadata"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    num_classes = int(config["model"]["num_classes"])

    sums = {}
    counts = {}
    targets = {}
    for batch in loader:
        volume_id = batch["volume_id"][0]
        start = int(batch["start_slice"].item())
        depth = int(batch["volume_depth"].item())
        images = batch["image"].to(device)
        masks = batch["mask"][0].cpu()
        window_logits = model.forward_grouped(images)[0].float().cpu()
        if volume_id not in sums:
            height, width = window_logits.shape[-2:]
            sums[volume_id] = torch.zeros(depth, num_classes, height, width)
            counts[volume_id] = torch.zeros(depth, 1, 1, 1)
            targets[volume_id] = torch.zeros(depth, height, width, dtype=torch.long)
        stop = start + window_logits.shape[0]
        sums[volume_id][start:stop] += window_logits
        counts[volume_id][start:stop] += 1
        targets[volume_id][start:stop] = masks

    cases = {}
    all_scores = defaultdict(list)
    for volume_id in sorted(sums):
        if (counts[volume_id] == 0).any():
            raise RuntimeError(f"Some slices of {volume_id} were not covered; use group_stride <= slice_count")
        scores = dice_per_case(sums[volume_id] / counts[volume_id], targets[volume_id], num_classes)
        cases[volume_id] = {
            "dice_by_class": scores,
            "mean_foreground_dice": float(np.mean(list(scores.values()))) if scores else None,
        }
        for class_id, score in scores.items():
            all_scores[class_id].append(score)
    valid_case_means = [
        case["mean_foreground_dice"]
        for case in cases.values()
        if case["mean_foreground_dice"] is not None
    ]
    if not valid_case_means:
        raise RuntimeError("The selected split contains no foreground labels")
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "mean_foreground_dice": float(np.mean(valid_case_means)),
        "dice_by_class": {key: float(np.mean(values)) for key, values in sorted(all_scores.items())},
        "cases": cases,
    }
    save_json(args.output, result)
    print(f"mean_foreground_dice={result['mean_foreground_dice']:.5f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
