from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .profiles import UNetProfile


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        normalization: str = "batch",
        *,
        profile: Optional[UNetProfile] = None,
    ) -> None:
        super().__init__()
        if profile is not None:
            padding = profile.conv_padding
            normalization = profile.normalization
        if normalization not in {"none", "batch"}:
            raise ValueError("normalization must be 'none' or 'batch'.")
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                bias=normalization == "none",
            )
        ]
        if normalization == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                bias=normalization == "none",
            )
        )
        if normalization == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(
            *layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        normalization: str = "batch",
        *,
        profile: Optional[UNetProfile] = None,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                padding=padding,
                normalization=normalization,
                profile=profile,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: Optional[bool] = None,
        *,
        profile: Optional[UNetProfile] = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.bilinear = (
            profile.upsampling == "bilinear" if profile is not None else bool(True if bilinear is None else bilinear)
        )
        if self.bilinear:
            self.up = nn.Identity()
            fused_channels = in_channels + skip_channels
        else:
            up_channels = out_channels if profile is not None else in_channels
            self.up = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2)
            fused_channels = up_channels + skip_channels
        self.conv = DoubleConv(
            fused_channels,
            out_channels,
            profile=profile,
        )

    @staticmethod
    def _center_crop(skip: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        height, width = skip.shape[-2:]
        target_height, target_width = target_size
        if target_height > height or target_width > width:
            raise ValueError("Cannot crop a skip tensor to a larger spatial size.")
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return skip[..., top : top + target_height, left : left + target_width]

    def upsample_and_concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.bilinear:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        else:
            x = self.up(x)
            if self.profile is not None and self.profile.crop_skip:
                skip = self._center_crop(skip, x.shape[-2:])
            elif x.shape[-2:] != skip.shape[-2:]:
                if self.profile is not None:
                    raise ValueError(
                        "Profile %s requires matching encoder and decoder spatial sizes."
                        % self.profile.name
                    )
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, x], dim=1)

    def fuse(self, concatenated: torch.Tensor) -> torch.Tensor:
        return self.conv(concatenated)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.upsample_and_concat(x, skip))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
