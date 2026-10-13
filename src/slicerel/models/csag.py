from __future__ import annotations

import math
from typing import Any, Dict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class CrossSliceAttentionGate(nn.Module):
    """Fixed-slice CS-AG with semantic and whole-adjacency gating."""

    def __init__(
        self,
        channels: int,
        slice_count: int,
        slice_adjacency: torch.Tensor,
        *,
        attention_dim: int,
        adjacency_mlp_hidden: int,
        adjacency_mlp_activation: str,
        adjacency_mlp_output_activation: str,
        qkv_bias: bool,
        spatial_reduction: int,
        gate_adapter: str = "direct_flat_to_slice_channel",
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        if slice_count < 1:
            raise ValueError("slice_count must be positive.")
        if attention_dim < 1:
            raise ValueError("attention_dim must be positive.")
        if adjacency_mlp_hidden < 1:
            raise ValueError("adjacency_mlp_hidden must be positive.")
        if spatial_reduction < 1:
            raise ValueError("spatial_reduction must be positive.")
        if adjacency_mlp_activation != "relu":
            raise ValueError("adjacency_mlp_activation must be 'relu'.")
        if adjacency_mlp_output_activation != "sigmoid":
            raise ValueError("adjacency_mlp_output_activation must be 'sigmoid'.")
        if gate_adapter not in {"direct_flat_to_slice_channel", "row_projection"}:
            raise ValueError(
                "gate_adapter must be 'direct_flat_to_slice_channel' or "
                "the explicit legacy diagnostic adapter 'row_projection'."
            )
        if not isinstance(qkv_bias, bool):
            raise ValueError("qkv_bias must be a bool.")
        if not isinstance(slice_adjacency, torch.Tensor):
            raise TypeError("slice_adjacency must be a torch.Tensor.")
        if slice_adjacency.shape != (slice_count, slice_count):
            raise ValueError(
                "slice_adjacency shape must be (%d, %d), got %s."
                % (slice_count, slice_count, tuple(slice_adjacency.shape))
            )
        if not torch.isfinite(slice_adjacency).all():
            raise ValueError("slice_adjacency must contain only finite values.")
        if torch.any(slice_adjacency < 0):
            raise ValueError("slice_adjacency must be non-negative.")

        self.channels = channels
        self.slice_count = slice_count
        self.attention_dim = attention_dim
        self.adjacency_mlp_hidden = adjacency_mlp_hidden
        self.adjacency_mlp_activation = adjacency_mlp_activation
        self.adjacency_mlp_output_activation = adjacency_mlp_output_activation
        self.qkv_bias = qkv_bias
        self.spatial_reduction = spatial_reduction
        self.gate_adapter = gate_adapter
        self.register_buffer("slice_adjacency", slice_adjacency.detach().clone())

        self.query = nn.Linear(channels, attention_dim, bias=qkv_bias)
        self.key = nn.Linear(channels, attention_dim, bias=qkv_bias)
        self.value = nn.Linear(channels, attention_dim, bias=qkv_bias)
        self.output_projection = nn.Linear(attention_dim, channels, bias=False)

        matrix_size = slice_count * slice_count
        adapter_output = (
            slice_count * channels
            if gate_adapter == "direct_flat_to_slice_channel"
            else matrix_size
        )
        self.adjacency_mlp = nn.Sequential(
            nn.Linear(matrix_size, adjacency_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(adjacency_mlp_hidden, adapter_output),
            nn.Sigmoid(),
        )
        if gate_adapter == "row_projection":
            self.row_projection = nn.Linear(slice_count, channels)

        spatial_hidden = max(channels // spatial_reduction, 1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, spatial_hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def adapter_contract(self) -> Dict[str, Any]:
        """Return the checkpoint-visible mapping performed by the fixed gate."""
        matrix_size = self.slice_count * self.slice_count
        if self.gate_adapter == "direct_flat_to_slice_channel":
            return {
                "name": self.gate_adapter,
                "input_layout": "S,S",
                "flatten_order": "row_major",
                "flattened_input_features": matrix_size,
                "hidden_features": self.adjacency_mlp_hidden,
                "hidden_activation": self.adjacency_mlp_activation,
                "linear_bias": True,
                "output_features": self.slice_count * self.channels,
                "output_activation": self.adjacency_mlp_output_activation,
                "output_layout": "B,S,C",
            }
        return {
            "name": self.gate_adapter,
            "profile": "legacy_diagnostic",
            "input_layout": "S,S",
            "flatten_order": "row_major",
            "flattened_input_features": matrix_size,
            "hidden_features": self.adjacency_mlp_hidden,
            "hidden_activation": self.adjacency_mlp_activation,
            "linear_bias": True,
            "mlp_output_features": matrix_size,
            "mlp_output_activation": self.adjacency_mlp_output_activation,
            "row_projection": "Linear(S,C)",
            "output_layout": "B,S,C",
        }

    def semantic_contract(self) -> Dict[str, Any]:
        """Return the checkpoint-visible paper formula for semantic attention."""
        return {
            "channels": self.channels,
            "attention_dim": self.attention_dim,
            "descriptor_layout": "B,S,C",
            "query_projection": "Linear(C,d)",
            "key_projection": "Linear(C,d)",
            "value_projection": "Linear(C,d)",
            "value_in_features": self.value.in_features,
            "value_out_features": self.value.out_features,
            "attention": "softmax(QK^T/sqrt(d))",
            "latent_context_layout": "B,S,d",
            "output_projection": "Linear(d,C)",
            "output_projection_in_features": self.output_projection.in_features,
            "output_projection_out_features": self.output_projection.out_features,
            "output_projection_bias": False,
            "attention_gate_layout": "B,S,C",
        }

    @staticmethod
    def _linear_float64(module: nn.Linear, value: torch.Tensor) -> torch.Tensor:
        bias = None if module.bias is None else module.bias.double()
        return F.linear(value, module.weight.double(), bias)

    @staticmethod
    def _conv_float64(module: nn.Conv2d, value: torch.Tensor) -> torch.Tensor:
        bias = None if module.bias is None else module.bias.double()
        return F.conv2d(
            value,
            module.weight.double(),
            bias,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )

    def _forward_float64(
        self, x: torch.Tensor, *, return_aux: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Re-evaluate the unchanged gate formula when float32-range math overflows."""
        original_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.double()
            batch, slices, channels, height, width = work.shape
            descriptors = work.mean(dim=(-2, -1))
            query = self._linear_float64(self.query, descriptors)
            key = self._linear_float64(self.key, descriptors)
            value = self._linear_float64(self.value, descriptors)
            logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
                self.attention_dim
            )
            semantic_weights = torch.softmax(logits, dim=-1)
            semantic_latent = torch.matmul(semantic_weights, value)
            semantic_context = self._linear_float64(
                self.output_projection, semantic_latent
            )

            flattened_adjacency = self.slice_adjacency.double().reshape(1, -1)
            fixed_gate = torch.sigmoid(
                self._linear_float64(
                    self.adjacency_mlp[2],
                    F.relu(
                        self._linear_float64(
                            self.adjacency_mlp[0], flattened_adjacency
                        )
                    ),
                )
            )
            if self.gate_adapter == "direct_flat_to_slice_channel":
                adjacency_channel_gate = fixed_gate.reshape(
                    1, self.slice_count, self.channels
                ).expand(batch, -1, -1)
                adjacency_gate_matrix = None
            else:
                adjacency_gate_matrix = fixed_gate.reshape(
                    1, self.slice_count, self.slice_count
                ).expand(batch, -1, -1)
                adjacency_channel_gate = self._linear_float64(
                    self.row_projection, adjacency_gate_matrix
                )

            modulated = work * (
                1.0
                + semantic_context[..., None, None]
                + adjacency_channel_gate[..., None, None]
            )
            flat = modulated.reshape(batch * slices, channels, height, width)
            spatial_attention = torch.sigmoid(
                self._conv_float64(
                    self.spatial_attention[2],
                    F.relu(
                        self._conv_float64(self.spatial_attention[0], flat)
                    ),
                )
            ).reshape(batch, slices, 1, height, width)
            output = (modulated * spatial_attention).to(original_dtype)

        if not torch.isfinite(output).all():
            raise FloatingPointError(
                "CS-AG output remains non-finite after float64 formula fallback."
            )
        if not return_aux:
            return output
        aux = {
            "adjacency": self.slice_adjacency,
            "descriptors": descriptors.to(original_dtype),
            "query": query.to(original_dtype),
            "key": key.to(original_dtype),
            "value": value.to(original_dtype),
            "semantic_weights": semantic_weights.to(original_dtype),
            "semantic_latent": semantic_latent.to(original_dtype),
            "semantic_context": semantic_context.to(original_dtype),
            "spatial_attention": spatial_attention.to(original_dtype),
        }
        if self.gate_adapter == "direct_flat_to_slice_channel":
            aux["adjacency_channel_gate"] = adjacency_channel_gate.to(original_dtype)
            aux["adapter_contract"] = self.adapter_contract()
        else:
            aux["adjacency_gate_matrix"] = adjacency_gate_matrix.to(original_dtype)
            aux["row_projected_gate"] = adjacency_channel_gate.to(original_dtype)
        return output, aux

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        if x.ndim != 5:
            raise ValueError("CS-AG expects [B,S,C,H,W].")
        batch, slices, channels, height, width = x.shape
        if slices != self.slice_count:
            raise ValueError(
                "CS-AG configured slice_count=%d, got runtime S=%d."
                % (self.slice_count, slices)
            )
        if channels != self.channels:
            raise ValueError(
                "CS-AG channel mismatch: expected %d, got %d."
                % (self.channels, channels)
            )

        descriptors = x.mean(dim=(-2, -1))
        query = self.query(descriptors)
        key = self.key(descriptors)
        value = self.value(descriptors)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.attention_dim
        )
        semantic_weights = torch.softmax(logits, dim=-1)
        semantic_latent = torch.matmul(semantic_weights, value)
        semantic_context = self.output_projection(semantic_latent)

        flattened_adjacency = self.slice_adjacency.reshape(1, -1)
        fixed_gate = self.adjacency_mlp(flattened_adjacency)
        if self.gate_adapter == "direct_flat_to_slice_channel":
            adjacency_channel_gate = fixed_gate.reshape(
                1, self.slice_count, self.channels
            ).expand(batch, -1, -1)
            adjacency_gate_matrix = None
        else:
            adjacency_gate_matrix = fixed_gate.reshape(
                1, self.slice_count, self.slice_count
            ).expand(batch, -1, -1)
            adjacency_channel_gate = self.row_projection(adjacency_gate_matrix)

        modulated = x * (
            1.0
            + semantic_context[..., None, None]
            + adjacency_channel_gate[..., None, None]
        )
        flat = modulated.reshape(batch * slices, channels, height, width)
        spatial_attention = self.spatial_attention(flat).reshape(
            batch, slices, 1, height, width
        )
        output = modulated * spatial_attention

        if not torch.isfinite(output).all():
            return self._forward_float64(x, return_aux=return_aux)

        if not return_aux:
            return output
        aux = {
            "adjacency": self.slice_adjacency,
            "descriptors": descriptors,
            "query": query,
            "key": key,
            "value": value,
            "semantic_weights": semantic_weights,
            "semantic_latent": semantic_latent,
            "semantic_context": semantic_context,
            "spatial_attention": spatial_attention,
        }
        if self.gate_adapter == "direct_flat_to_slice_channel":
            aux["adjacency_channel_gate"] = adjacency_channel_gate
            aux["adapter_contract"] = self.adapter_contract()
        else:
            aux["adjacency_gate_matrix"] = adjacency_gate_matrix
            aux["row_projected_gate"] = adjacency_channel_gate
        return output, aux
