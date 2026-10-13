from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from slicerel.experiment import (
    build_dataset,
    build_loss,
    build_model,
    foreground_dice,
    load_config,
    resolve_device,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SliceRel-U-Net")
    parser.add_argument("--config", type=Path, default=Path("configs/example.yaml"))
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


@torch.no_grad()
def validate(model, criterion, loader, device, num_classes):
    model.eval()
    losses, scores = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model.forward_grouped(images)
        loss, _ = criterion(logits, masks)
        losses.append(float(loss.item()))
        score = foreground_dice(logits, masks, num_classes)
        if math.isfinite(score):
            scores.append(score)
    if not scores:
        raise RuntimeError("The validation split contains no foreground labels")
    return sum(losses) / len(losses), sum(scores) / len(scores)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    data_config = config["data"]
    training_config = config["training"]
    num_classes = int(config["model"]["num_classes"])

    train_data = build_dataset(config, "train")
    val_data = build_dataset(config, "val")
    train_loader = DataLoader(
        train_data,
        batch_size=int(data_config.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(data_config.get("validation_batch_size", 1)),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"].get("learning_rate", 1e-4)),
        weight_decay=float(config["optimizer"].get("weight_decay", 1e-4)),
    )
    start_epoch, best_dice = 0, float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.assert_compatible_metadata(checkpoint["model_metadata"])
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_val_dice", best_dice))

    output_dir = Path(training_config.get("output_dir", "runs/example"))
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(training_config.get("epochs", 100))
    accumulation = max(int(training_config.get("gradient_accumulation_steps", 1)), 1)
    clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
    use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    amp_name = str(training_config.get("amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    scaler_enabled = use_amp and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model.forward_grouped(images)
                loss, _ = criterion(logits, masks)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.item())

        val_loss, val_dice = validate(model, criterion, val_loader, device, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "val_foreground_dice": val_dice,
        }
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "best_val_dice": max(best_dice, val_dice),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_metadata": model.reproduction_metadata(),
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(checkpoint, output_dir / "best.pt")
        save_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={row['train_loss']:.5f} "
            f"val_loss={val_loss:.5f} val_dice={val_dice:.5f}"
        )


if __name__ == "__main__":
    main()
