from __future__ import annotations

import math
from typing import Optional

import torch


ADJACENCY_STRATEGIES = ("ordered_chain", "fully_connected", "identity")
NORMALIZATION_MODES = ("symmetric", "row", "none")


def _validate_weight(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("%s must be a finite non-negative weight." % name)
    return value


def build_weighted_slice_adjacency(
    slice_count: int,
    *,
    adjacent_weight: float,
    distant_weight: float,
    self_loop_weight: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the fixed weighted graph over ordered slice nodes only."""
    if slice_count < 1:
        raise ValueError("slice_count must be positive.")
    adjacent = _validate_weight("adjacent_weight", adjacent_weight)
    distant = _validate_weight("distant_weight", distant_weight)
    self_loop = _validate_weight("self_loop_weight", self_loop_weight)

    indices = torch.arange(slice_count, device=device)
    distance = (indices[:, None] - indices[None, :]).abs()
    adjacency = torch.full(
        (slice_count, slice_count),
        distant,
        device=device,
        dtype=dtype,
    )
    adjacency.masked_fill_(distance == 1, adjacent)
    adjacency.fill_diagonal_(self_loop)
    return adjacency


def build_global_node_adjacency(
    slice_count: int,
    *,
    adjacent_weight: float,
    distant_weight: float,
    global_weight: float,
    self_loop_weight: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an ``S+1`` graph with bidirectional slice/global edges."""
    global_edge = _validate_weight("global_weight", global_weight)
    self_loop = _validate_weight("self_loop_weight", self_loop_weight)
    slice_adjacency = build_weighted_slice_adjacency(
        slice_count,
        adjacent_weight=adjacent_weight,
        distant_weight=distant_weight,
        self_loop_weight=self_loop,
        device=device,
        dtype=dtype,
    )
    adjacency = torch.full(
        (slice_count + 1, slice_count + 1),
        global_edge,
        device=device,
        dtype=dtype,
    )
    adjacency[:slice_count, :slice_count] = slice_adjacency
    adjacency.fill_diagonal_(self_loop)
    return adjacency


def build_ordered_adjacency(
    num_slices: int,
    radius: int = 1,
    include_self: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a symmetric adjacency matrix for an ordered slice chain."""
    if num_slices < 1:
        raise ValueError("num_slices must be positive.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    indices = torch.arange(num_slices, device=device)
    distance = (indices[:, None] - indices[None, :]).abs()
    adjacency = (distance <= radius).to(dtype=dtype)
    if not include_self:
        adjacency.fill_diagonal_(0)
    return adjacency


def build_adjacency(
    num_slices: int,
    strategy: str = "ordered_chain",
    radius: int = 1,
    include_self: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an explicit graph topology for an ordered slice group.

    These strategies are engineering choices, not claims about the thesis's
    unspecified graph-construction rule. They are stored in the checkpoint.
    """
    if strategy not in ADJACENCY_STRATEGIES:
        raise ValueError("Unknown adjacency strategy: %s" % strategy)
    if strategy == "ordered_chain":
        return build_ordered_adjacency(num_slices, radius, include_self, device, dtype)
    if strategy == "fully_connected":
        adjacency = torch.ones((num_slices, num_slices), device=device, dtype=dtype)
        if not include_self:
            adjacency.fill_diagonal_(0)
        return adjacency
    if not include_self:
        raise ValueError("identity adjacency requires include_self=True.")
    return torch.eye(num_slices, device=device, dtype=dtype)


def normalize_adjacency(
    adjacency: torch.Tensor,
    mode: str = "symmetric",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize a square adjacency matrix with an explicit named rule."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square [S,S] tensor.")
    if mode not in NORMALIZATION_MODES:
        raise ValueError("Unknown adjacency normalization: %s" % mode)
    if mode == "none":
        return adjacency
    degree = adjacency.sum(dim=-1).clamp_min(eps)
    if mode == "row":
        return adjacency / degree[:, None]
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def resolve_adjacency(
    num_slices: int,
    radius: int,
    reference: torch.Tensor,
    adjacency: torch.Tensor = None,
    strategy: str = "ordered_chain",
    include_self: bool = True,
    normalization: str = "symmetric",
) -> torch.Tensor:
    if adjacency is None:
        adjacency = build_adjacency(
            num_slices=num_slices,
            strategy=strategy,
            radius=radius,
            include_self=include_self,
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        adjacency = adjacency.to(device=reference.device, dtype=reference.dtype)
    if adjacency.shape != (num_slices, num_slices):
        raise ValueError(
            "Expected adjacency shape (%d,%d), got %s."
            % (num_slices, num_slices, tuple(adjacency.shape))
        )
    return normalize_adjacency(adjacency, mode=normalization)
