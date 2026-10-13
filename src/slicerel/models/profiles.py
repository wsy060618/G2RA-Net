"""Immutable U-Net backbone contracts used by SliceRel-U-Net.

The profiles deliberately describe incompatible historical architectures rather
than exposing a collection of mutable defaults.  A checkpoint therefore always
has one unambiguous backbone geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping


@dataclass(frozen=True)
class UNetProfile:
    """Complete structural contract for a five-level 2D U-Net backbone."""

    name: str
    widths: tuple[int, int, int, int, int]
    conv_padding: int
    normalization: str
    upsampling: str
    crop_skip: bool
    bilinear_keep_channels: bool


_PROFILES: Mapping[str, UNetProfile] = MappingProxyType(
    {
        "canonical_same_size_64": UNetProfile(
            name="canonical_same_size_64",
            widths=(64, 128, 256, 512, 1024),
            conv_padding=1,
            normalization="none",
            upsampling="transposed",
            crop_skip=False,
            bilinear_keep_channels=False,
        ),
        "original_2015_valid": UNetProfile(
            name="original_2015_valid",
            widths=(64, 128, 256, 512, 1024),
            conv_padding=0,
            normalization="none",
            upsampling="transposed",
            crop_skip=True,
            bilinear_keep_channels=False,
        ),
        "legacy_bn_bilinear_32": UNetProfile(
            name="legacy_bn_bilinear_32",
            widths=(32, 64, 128, 256, 512),
            conv_padding=1,
            normalization="batch",
            upsampling="bilinear",
            crop_skip=False,
            bilinear_keep_channels=True,
        ),
    }
)

VALID_UNET_PROFILE_NAMES = tuple(_PROFILES)


def resolve_unet_profile(name: str) -> UNetProfile:
    """Return the immutable profile named by ``name``.

    Returning the registry's frozen value rather than a copy makes the profile
    identity stable while preventing per-run mutation of architectural choices.
    """

    try:
        return _PROFILES[name]
    except KeyError as error:
        valid = ", ".join(VALID_UNET_PROFILE_NAMES)
        raise ValueError("Unknown U-Net profile %r. Valid profiles: %s." % (name, valid)) from error


def resolve_registered_unet_profile(profile: UNetProfile) -> UNetProfile:
    """Return the registry value only when ``profile`` is its exact contract."""

    registered = resolve_unet_profile(profile.name)
    if profile != registered:
        raise ValueError(
            "U-Net profile %r differs from its registered contract." % profile.name
        )
    return registered


def unet_profile_contract(profile: UNetProfile) -> Dict[str, object]:
    """Serialize every immutable field that changes backbone computation."""

    return {
        "name": profile.name,
        "widths": list(profile.widths),
        "conv_padding": profile.conv_padding,
        "normalization": profile.normalization,
        "upsampling": profile.upsampling,
        "crop_skip": profile.crop_skip,
        "bilinear_keep_channels": profile.bilinear_keep_channels,
    }
