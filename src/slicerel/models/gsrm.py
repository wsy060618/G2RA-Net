from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .adjacency import build_global_node_adjacency, normalize_adjacency


GLOBAL_NODE_INITIALIZATIONS = ("mean", "zero", "learned")
FINAL_THESIS_GLOBAL_NODE_EVIDENCE = {
    "source": "final_thesis_chapter_4_figure_4_2",
    "conflict": "algorithm_4_1_describes_S_by_S_adjacency_without_global_node",
    "resolution": "figure_4_2_global_node_graph_is_authoritative_for_research_reconstruction",
}


class GraphSliceRelationModule(nn.Module):
    """Fixed-S graph propagation over slice nodes and one global node."""

    def __init__(
        self,
        channels: int,
        slice_count: int,
        hidden_channels: Optional[int] = None,
        *,
        adjacent_weight: float,
        distant_weight: float,
        global_weight: float,
        self_loop_weight: float,
        normalization: str,
        global_node_init: str,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        if slice_count < 1:
            raise ValueError("slice_count must be positive.")
        hidden = channels if hidden_channels is None else hidden_channels
        if hidden < 1:
            raise ValueError("hidden_channels must be positive.")
        if global_node_init not in GLOBAL_NODE_INITIALIZATIONS:
            raise ValueError(
                "global_node_init must be one of %s." % (GLOBAL_NODE_INITIALIZATIONS,)
            )
        self.channels = channels
        self.slice_count = slice_count
        self.hidden_channels = hidden
        self.global_node_init = global_node_init
        self.adjacent_weight = float(adjacent_weight)
        self.distant_weight = float(distant_weight)
        self.global_weight = float(global_weight)
        self.self_loop_weight = float(self_loop_weight)
        self.normalization = str(normalization)
        self.linear1 = nn.Linear(channels, hidden, bias=False)
        self.linear2 = nn.Linear(hidden, channels, bias=False)
        raw_adjacency = build_global_node_adjacency(
            slice_count,
            adjacent_weight=adjacent_weight,
            distant_weight=distant_weight,
            global_weight=global_weight,
            self_loop_weight=self_loop_weight,
        )
        self.register_buffer(
            "adjacency", normalize_adjacency(raw_adjacency, mode=normalization)
        )
        if global_node_init == "learned":
            self.learned_global_node = nn.Parameter(torch.zeros(1, 1, channels))
        else:
            self.register_parameter("learned_global_node", None)

    def reproduction_contract(self) -> Dict[str, object]:
        """Return the checkpoint-visible final-thesis GSRM semantics."""
        return {
            "topology": "fixed_weighted_ordered_slice_graph_with_global_node",
            "node_count": self.slice_count + 1,
            "adjacency_layout": "S+1,S+1",
            "global_node_position": "last",
            "weights": {
                "adjacent": self.adjacent_weight,
                "distant": self.distant_weight,
                "global": self.global_weight,
                "self_loop": self.self_loop_weight,
            },
            "global_node_initialization": self.global_node_init,
            "normalization": self.normalization,
            "linear_bias": False,
            "activation": "relu",
            "propagation_layers": 2,
            "global_node_participation": "both_propagation_layers",
            "output_modulation_nodes": "slice_only",
            "final_thesis_evidence": dict(FINAL_THESIS_GLOBAL_NODE_EVIDENCE),
        }

    def _initial_global_node(self, descriptors: torch.Tensor) -> torch.Tensor:
        if self.global_node_init == "mean":
            return descriptors.mean(dim=1, keepdim=True)
        if self.global_node_init == "zero":
            return descriptors.new_zeros((descriptors.shape[0], 1, self.channels))
        return self.learned_global_node.expand(descriptors.shape[0], -1, -1)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if x.ndim != 5:
            raise ValueError("GSRM expects [B,S,C,H,W].")
        batch, slices, channels, _, _ = x.shape
        if slices != self.slice_count:
            raise ValueError(
                "GSRM slice mismatch: expected %d, got %d."
                % (self.slice_count, slices)
            )
        if channels != self.channels:
            raise ValueError("GSRM channel mismatch: expected %d, got %d." % (self.channels, channels))

        slice_descriptors = x.mean(dim=(-2, -1))
        initial_global_node = self._initial_global_node(slice_descriptors)
        descriptors = torch.cat((slice_descriptors, initial_global_node), dim=1)
        adjacency = self.adjacency.to(device=x.device, dtype=x.dtype)

        projected = self.linear1(descriptors)
        aggregated = torch.einsum("st,bth->bsh", adjacency, projected)
        message1 = F.relu(aggregated, inplace=False)
        aggregated2 = torch.einsum("st,bth->bsh", adjacency, message1)
        message2 = self.linear2(F.relu(aggregated2, inplace=False))
        output = x * (1.0 + message2[:, :slices, :, None, None])

        if not return_aux:
            return output
        return output, {
            "adjacency": adjacency,
            "descriptors": descriptors,
            "initial_global_node": initial_global_node,
            "message1": message1,
            "message2": message2,
            "updated_global_node": message2[:, -1:, :],
        }
