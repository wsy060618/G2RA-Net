from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VolumeRecord:
    volume_id: str
    image_path: Path
    label_path: Path
    split: str
    spacing: Tuple[float, float, float]


def _resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def read_manifest(path: Path, split: str) -> List[VolumeRecord]:
    path = Path(path).expanduser().resolve()
    records: List[VolumeRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"volume_id", "image_path", "label_path", "split"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest columns must include: %s" % ", ".join(sorted(required)))
        for row in reader:
            if (row.get("split") or "").strip() != split:
                continue
            spacing = tuple(float(row.get(f"spacing_{axis}") or 1.0) for axis in range(3))
            records.append(
                VolumeRecord(
                    volume_id=(row.get("volume_id") or "").strip(),
                    image_path=_resolve_path(path, row.get("image_path") or ""),
                    label_path=_resolve_path(path, row.get("label_path") or ""),
                    split=split,
                    spacing=spacing,
                )
            )
    if not records:
        raise ValueError(f"No rows found for split={split!r} in {path}")
    if len({record.volume_id for record in records}) != len(records):
        raise ValueError("volume_id must be unique within each split")
    return records


def _read_single_array(path: Path) -> np.ndarray:
    name = path.name.lower()
    if name.endswith(".npy"):
        return np.asarray(np.load(path, allow_pickle=False))
    if name.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(f"NPZ file must contain exactly one array: {path}")
            return np.asarray(archive[archive.files[0]])
    raise ValueError(f"Unsupported volume file: {path}")


def _read_pair(record: VolumeRecord) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    if not record.image_path.exists() or not record.label_path.exists():
        raise FileNotFoundError(f"Missing image or label for {record.volume_id}")
    is_nifti = record.image_path.name.lower().endswith((".nii", ".nii.gz"))
    label_is_nifti = record.label_path.name.lower().endswith((".nii", ".nii.gz"))
    if is_nifti != label_is_nifti:
        raise ValueError(f"Image and label formats differ for {record.volume_id}")
    if is_nifti:
        import nibabel as nib

        image_nii = nib.as_closest_canonical(nib.load(str(record.image_path)))
        label_nii = nib.as_closest_canonical(nib.load(str(record.label_path)))
        if image_nii.shape != label_nii.shape or not np.allclose(image_nii.affine, label_nii.affine):
            raise ValueError(f"Image and label grids differ for {record.volume_id}")
        image = np.asarray(image_nii.get_fdata(dtype=np.float32), dtype=np.float32)
        raw_label = np.asarray(label_nii.get_fdata(dtype=np.float32))
        spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image_nii.affine))
    else:
        image = _read_single_array(record.image_path).astype(np.float32, copy=False)
        raw_label = _read_single_array(record.label_path)
        spacing = record.spacing
    if image.ndim != 3 or raw_label.ndim != 3 or image.shape != raw_label.shape:
        raise ValueError(f"Image and label must be aligned 3D volumes for {record.volume_id}")
    if not np.isfinite(image).all() or not np.isfinite(raw_label).all():
        raise ValueError(f"NaN or Inf found in {record.volume_id}")
    if not np.allclose(raw_label, np.rint(raw_label)):
        raise ValueError(f"Label must contain integer class IDs for {record.volume_id}")
    return image, np.rint(raw_label).astype(np.int64), spacing


class VolumeWindowDataset(Dataset):
    """Return consecutive 2D slices as [S, 1, H, W] windows."""

    def __init__(
        self,
        manifest: Path,
        split: str,
        slice_count: int,
        *,
        slice_axis: int = 2,
        group_stride: int = 1,
        target_size: Optional[Sequence[int]] = (224, 224),
        normalization: str = "minmax",
        cache_volumes: int = 2,
    ) -> None:
        if slice_count < 1 or group_stride < 1:
            raise ValueError("slice_count and group_stride must be positive")
        if slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")
        if normalization not in {"none", "minmax", "zscore"}:
            raise ValueError("normalization must be none, minmax, or zscore")
        self.records = read_manifest(Path(manifest), split)
        self.slice_count = int(slice_count)
        self.slice_axis = int(slice_axis)
        self.group_stride = int(group_stride)
        self.target_size = tuple(int(v) for v in target_size) if target_size else None
        self.normalization = normalization
        self.cache_volumes = max(int(cache_volumes), 0)
        self._cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]]" = OrderedDict()
        self.windows: List[Tuple[int, int]] = []
        for index, record in enumerate(self.records):
            image, label, spacing = self._load(index)
            depth = int(image.shape[0])
            if depth < self.slice_count:
                raise ValueError(f"{record.volume_id} has {depth} slices, fewer than {self.slice_count}")
            self.windows.extend(
                (index, start)
                for start in range(0, depth - self.slice_count + 1, self.group_stride)
            )

    def _load(self, index: int):
        if index in self._cache:
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        image, label, spacing_axes = _read_pair(self.records[index])
        order = (self.slice_axis,) + tuple(axis for axis in range(3) if axis != self.slice_axis)
        image = np.moveaxis(image, self.slice_axis, 0).astype(np.float32, copy=False)
        label = np.moveaxis(label, self.slice_axis, 0).astype(np.int64, copy=False)
        spacing = tuple(spacing_axes[axis] for axis in order)
        if self.normalization == "minmax":
            low, high = float(image.min()), float(image.max())
            image = (image - low) / max(high - low, 1e-6)
        elif self.normalization == "zscore":
            image = (image - float(image.mean())) / max(float(image.std()), 1e-6)
        value = (image, label, spacing)
        if self.cache_volumes:
            self._cache[index] = value
            while len(self._cache) > self.cache_volumes:
                self._cache.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record_index, start = self.windows[index]
        image, label, spacing = self._load(record_index)
        stop = start + self.slice_count
        image_tensor = torch.from_numpy(np.ascontiguousarray(image[start:stop])).unsqueeze(1)
        label_tensor = torch.from_numpy(np.ascontiguousarray(label[start:stop])).long()
        output_spacing = list(spacing)
        if self.target_size is not None and image_tensor.shape[-2:] != self.target_size:
            old_height, old_width = image_tensor.shape[-2:]
            image_tensor = F.interpolate(image_tensor, self.target_size, mode="bilinear", align_corners=False)
            label_tensor = F.interpolate(
                label_tensor[:, None].float(), self.target_size, mode="nearest"
            )[:, 0].long()
            output_spacing[1] *= old_height / self.target_size[0]
            output_spacing[2] *= old_width / self.target_size[1]
        record = self.records[record_index]
        return {
            "image": image_tensor,
            "mask": label_tensor,
            "volume_id": record.volume_id,
            "start_slice": start,
            "volume_depth": int(image.shape[0]),
            "spacing_dhw": torch.tensor(output_spacing, dtype=torch.float64),
        }

