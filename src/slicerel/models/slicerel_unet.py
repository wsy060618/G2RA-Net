from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .adjacency import build_weighted_slice_adjacency, normalize_adjacency
from .blocks import DoubleConv, Down, OutConv, Up
from .csag import CrossSliceAttentionGate
from .gsrm import GraphSliceRelationModule
from .profiles import (
    UNetProfile,
    resolve_registered_unet_profile,
    resolve_unet_profile,
    unet_profile_contract,
)


CSAG_SCALE_NAMES = ("bottleneck", "up1", "up2", "up3", "up4")
VARIANTS = ("unet", "gsrm_only", "csag_only", "slicerel_unet")
APPROVED_CSAG_POSITION = "post_concat_pre_fuse"
TensorOrOutput = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]


class SliceRelUNet(nn.Module):
    """Fixed-S SliceRel-U-Net assembled from an immutable backbone profile."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        slice_count: int,
        *,
        backbone_profile: Union[str, UNetProfile] = "legacy_bn_bilinear_32",
        variant: Optional[str] = None,
        graph_hidden_channels: Optional[int] = None,
        adjacent_weight: float = 1.0,
        distant_weight: float = 0.1,
        global_weight: float = 1.0,
        self_loop_weight: float = 1.0,
        adjacency_normalization: str = "symmetric",
        global_node_init: str = "mean",
        attention_dim: Optional[int] = 64,
        attention_dim_rule: str = "fixed",
        adjacency_mlp_hidden: Optional[int] = None,
        adjacency_mlp_activation: str = "relu",
        adjacency_mlp_output_activation: str = "sigmoid",
        qkv_bias: bool = False,
        spatial_reduction: int = 4,
        csag_position: str = APPROVED_CSAG_POSITION,
        gate_adapter: str = "direct_flat_to_slice_channel",
        **legacy_graph: object,
    ) -> None:
        super().__init__()
        if slice_count < 1:
            raise ValueError("slice_count must be positive.")
        if isinstance(backbone_profile, str):
            profile = resolve_unet_profile(backbone_profile)
        elif isinstance(backbone_profile, UNetProfile):
            profile = resolve_registered_unet_profile(backbone_profile)
        else:
            raise TypeError("backbone_profile must be a profile name or UNetProfile.")

        if variant is None:
            raise ValueError("variant is required; migrate legacy module switches explicitly.")
        if variant not in VARIANTS:
            raise ValueError("variant must be one of %s." % (VARIANTS,))
        if csag_position != APPROVED_CSAG_POSITION:
            raise ValueError("csag_position must be 'post_concat_pre_fuse'.")
        if legacy_graph:
            raise TypeError(
                "Obsolete graph options %s are ignored by the weighted graph; migrate to "
                "adjacent_weight, distant_weight, global_weight, and self_loop_weight."
                % ", ".join(sorted(legacy_graph))
            )

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.slice_count = int(slice_count)
        self.profile = profile
        self.backbone_profile = profile.name
        self.variant = variant
        self.use_gsrm = variant in {"gsrm_only", "slicerel_unet"}
        self.use_csag = variant in {"csag_only", "slicerel_unet"}
        self.csag_scales = CSAG_SCALE_NAMES if self.use_csag else ()
        self.csag_position = csag_position
        if attention_dim_rule not in {"fixed", "quarter_channels"}:
            raise ValueError("attention_dim_rule must be 'fixed' or 'quarter_channels'.")
        if attention_dim_rule == "fixed" and (attention_dim is None or int(attention_dim) < 1):
            raise ValueError("fixed attention_dim_rule requires a positive attention_dim.")
        if attention_dim_rule == "quarter_channels" and attention_dim is not None:
            raise ValueError("quarter_channels attention_dim_rule requires attention_dim=null.")
        hidden = profile.widths[4] if graph_hidden_channels is None else int(graph_hidden_channels)
        matrix_hidden = slice_count * slice_count if adjacency_mlp_hidden is None else int(adjacency_mlp_hidden)
        self.graph_config = {
            "hidden_channels": hidden,
            "adjacent_weight": float(adjacent_weight),
            "distant_weight": float(distant_weight),
            "global_weight": float(global_weight),
            "self_loop_weight": float(self_loop_weight),
            "normalization": str(adjacency_normalization),
            "global_node_init": str(global_node_init),
        }
        self.csag_config = {
            "attention_dim": None if attention_dim is None else int(attention_dim),
            "adjacency_mlp_hidden": matrix_hidden,
            "adjacency_mlp_activation": str(adjacency_mlp_activation),
            "adjacency_mlp_output_activation": str(adjacency_mlp_output_activation),
            "qkv_bias": bool(qkv_bias),
            "gate_adapter": str(gate_adapter),
            "spatial_reduction": int(spatial_reduction),
            "scales": list(self.csag_scales),
            "position": csag_position,
        }

        widths = profile.widths
        self.inc = DoubleConv(in_channels, widths[0], profile=profile)
        self.down1 = Down(widths[0], widths[1], profile=profile)
        self.down2 = Down(widths[1], widths[2], profile=profile)
        self.down3 = Down(widths[2], widths[3], profile=profile)
        self.down4 = Down(widths[3], widths[4], profile=profile)
        self.gsrm = (
            GraphSliceRelationModule(
                widths[4], slice_count, hidden,
                adjacent_weight=adjacent_weight,
                distant_weight=distant_weight,
                global_weight=global_weight,
                self_loop_weight=self_loop_weight,
                normalization=adjacency_normalization,
                global_node_init=global_node_init,
            )
            if self.use_gsrm else None
        )
        if self.gsrm is not None:
            self.graph_config["contract"] = self.gsrm.reproduction_contract()
            self.graph_config["final_thesis_evidence"] = dict(
                self.graph_config["contract"]["final_thesis_evidence"]
            )

        slice_adjacency = normalize_adjacency(
            build_weighted_slice_adjacency(
                slice_count,
                adjacent_weight=adjacent_weight,
                distant_weight=distant_weight,
                self_loop_weight=self_loop_weight,
            ),
            mode=adjacency_normalization,
        )

        resolved_attention_dims: Dict[str, int] = {}
        adapter_contracts: Dict[str, Dict[str, Any]] = {}
        semantic_contracts: Dict[str, Dict[str, Any]] = {}

        def make_csag(name: str, channels: int) -> CrossSliceAttentionGate:
            resolved_attention_dim = (
                max(channels // 4, 1)
                if attention_dim_rule == "quarter_channels"
                else int(attention_dim)
            )
            resolved_attention_dims[name] = resolved_attention_dim
            gate = CrossSliceAttentionGate(
                channels, slice_count, slice_adjacency,
                attention_dim=resolved_attention_dim,
                adjacency_mlp_hidden=matrix_hidden,
                adjacency_mlp_activation=adjacency_mlp_activation,
                adjacency_mlp_output_activation=adjacency_mlp_output_activation,
                qkv_bias=qkv_bias,
                spatial_reduction=spatial_reduction,
                gate_adapter=gate_adapter,
            )
            adapter_contracts[name] = gate.adapter_contract()
            semantic_contracts[name] = gate.semantic_contract()
            return gate

        self.bottleneck_csag = make_csag("bottleneck", widths[4]) if "bottleneck" in self.csag_scales else None
        self.up1 = Up(widths[4], widths[3], widths[3], profile=profile)
        self.up2 = Up(widths[3], widths[2], widths[2], profile=profile)
        self.up3 = Up(widths[2], widths[1], widths[1], profile=profile)
        self.up4 = Up(widths[1], widths[0], widths[0], profile=profile)
        self._ups = (self.up1, self.up2, self.up3, self.up4)
        self._decoder_names = ("up1", "up2", "up3", "up4")
        concat_channels = (
            widths[3] + widths[3],
            widths[2] + widths[2],
            widths[1] + widths[1],
            widths[0] + widths[0],
        ) if profile.upsampling == "transposed" else (
            widths[4] + widths[3],
            widths[3] + widths[2],
            widths[2] + widths[1],
            widths[1] + widths[0],
        )
        self.decoder_csag = nn.ModuleDict(
            {
                name: make_csag(name, channels)
                for name, channels in zip(self._decoder_names, concat_channels)
                if name in self.csag_scales
            }
        )
        self.csag_config["attention_dim_rule"] = attention_dim_rule
        self.csag_config["resolved_attention_dim_by_scale"] = resolved_attention_dims
        self.csag_config["adapter_contract_by_scale"] = adapter_contracts
        self.csag_config["semantic_contract_by_scale"] = semantic_contracts
        self.outc = OutConv(widths[0], num_classes)

    @staticmethod
    def _group(feature: torch.Tensor, batch: int, slices: int) -> torch.Tensor:
        return feature.reshape(batch, slices, *feature.shape[1:])

    @staticmethod
    def _ungroup(feature: torch.Tensor) -> torch.Tensor:
        return feature.reshape(feature.shape[0] * feature.shape[1], *feature.shape[2:])

    @staticmethod
    def _record_shape(trace, path, feature: torch.Tensor, layout: str) -> None:
        """Record a grouped tensor shape under a stable diagnostic namespace."""
        if trace is None:
            return
        record = trace
        for part in path[:-1]:
            record = record.setdefault(part, {})
        record[path[-1]] = {
            "shape": list(feature.shape),
            "layout": layout,
            "dtype": str(feature.dtype).replace("torch.", ""),
        }

    @staticmethod
    def _capture(features, path, feature: torch.Tensor) -> None:
        if features is None:
            return
        target = features
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target[path[-1]] = feature.detach()

    def _decode_stage(self, name, up, decoder_feature, skip, batch, slices, return_aux, trace, features):
        concatenated = up.upsample_and_concat(decoder_feature, skip)
        grouped = self._group(concatenated, batch, slices)
        self._record_shape(trace, ("decoder", name, "post_concat"), grouped, "B,S,C,H,W")
        self._capture(features, ("decoder", name, "post_concat"), grouped)
        aux = None
        if name in self.decoder_csag:
            if return_aux:
                grouped, aux = self.decoder_csag[name](grouped, return_aux=True)
            else:
                grouped = self.decoder_csag[name](grouped)
            concatenated = self._ungroup(grouped)
            self._record_shape(trace, ("decoder", name, "post_csag"), grouped, "B,S,C,H,W")
            self._capture(features, ("decoder", name, "post_csag"), grouped)
        fused = up.fuse(concatenated)
        grouped_fused = self._group(fused, batch, slices)
        self._record_shape(trace, ("decoder", name, "post_fuse"), grouped_fused, "B,S,C,H,W")
        self._capture(features, ("decoder", name, "post_fuse"), grouped_fused)
        return fused, aux

    def reproduction_metadata(self) -> Dict[str, object]:
        metadata = {
            "architecture": "SliceRelUNet",
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "slice_count": self.slice_count,
            "backbone_profile": self.backbone_profile,
            "backbone_profile_contract": unet_profile_contract(self.profile),
            "variant": self.variant,
            "graph": dict(self.graph_config),
            "csag": dict(self.csag_config),
            # Compatibility keys make old checkpoints fail clearly rather than silently.
            "use_gsrm": self.use_gsrm,
            "use_csag": self.use_csag,
            "csag_scales": list(self.csag_scales),
            "csag_position": self.csag_position,
            "diagnostics": {
                "return_aux": True,
                "trace": True,
                "gsrm": self.gsrm is not None,
                "csag": self.use_csag,
            },
        }
        return metadata

    def assert_compatible_metadata(self, metadata: Dict[str, object]) -> None:
        expected = self.reproduction_metadata()
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError("Checkpoint architecture metadata differs for: %s." % ", ".join(mismatches))

    def forward(self, x, slice_count=None, return_aux=False, return_trace=False, capture_features=False):
        if x.ndim != 4:
            raise ValueError("SliceRelUNet.forward expects [N,C,H,W].")
        if x.shape[1] != self.in_channels:
            raise ValueError("Input channel mismatch.")
        slices = self.slice_count if slice_count is None else int(slice_count)
        if slices != self.slice_count:
            raise ValueError(
                "SliceRelUNet expected %d, got %d; configured slice_count=%d, runtime S=%d."
                % (self.slice_count, slices, self.slice_count, slices)
            )
        if x.shape[0] % slices:
            raise ValueError("N must be divisible by slice_count.")
        batch = x.shape[0] // slices
        trace = {} if (return_trace or return_aux or capture_features) else None
        features = {} if capture_features else None
        grouped_input = self._group(x, batch, slices)
        self._record_shape(trace, ("input", "slices"), grouped_input, "B,S,C,H,W")
        self._capture(features, ("input", "slices"), grouped_input)
        x1 = self.inc(x)
        x1_grouped = self._group(x1, batch, slices)
        self._record_shape(trace, ("encoder", "inc"), x1_grouped, "B,S,C,H,W"); self._capture(features, ("encoder", "inc"), x1_grouped)
        x2 = self.down1(x1)
        x2_grouped = self._group(x2, batch, slices)
        self._record_shape(trace, ("encoder", "down1"), x2_grouped, "B,S,C,H,W"); self._capture(features, ("encoder", "down1"), x2_grouped)
        x3 = self.down2(x2)
        x3_grouped = self._group(x3, batch, slices)
        self._record_shape(trace, ("encoder", "down2"), x3_grouped, "B,S,C,H,W"); self._capture(features, ("encoder", "down2"), x3_grouped)
        x4 = self.down3(x3)
        x4_grouped = self._group(x4, batch, slices)
        self._record_shape(trace, ("encoder", "down3"), x4_grouped, "B,S,C,H,W"); self._capture(features, ("encoder", "down3"), x4_grouped)
        x5 = self.down4(x4)
        x5_grouped = self._group(x5, batch, slices)
        self._record_shape(trace, ("encoder", "down4"), x5_grouped, "B,S,C,H,W"); self._capture(features, ("encoder", "down4"), x5_grouped)
        grouped = self._group(x5, batch, slices)
        self._record_shape(trace, ("bottleneck", "grouped_input"), grouped, "B,S,C,H,W")
        gsrm_aux = bottleneck_aux = None
        if self.gsrm is not None:
            if return_aux: grouped, gsrm_aux = self.gsrm(grouped, return_aux=True)
            else: grouped = self.gsrm(grouped)
            self._record_shape(trace, ("bottleneck", "post_gsrm"), grouped, "B,S,C,H,W"); self._capture(features, ("bottleneck", "post_gsrm"), grouped)
        if self.bottleneck_csag is not None:
            if return_aux: grouped, bottleneck_aux = self.bottleneck_csag(grouped, return_aux=True)
            else: grouped = self.bottleneck_csag(grouped)
            self._record_shape(trace, ("bottleneck", "post_csag"), grouped, "B,S,C,H,W"); self._capture(features, ("bottleneck", "post_csag"), grouped)
        x5 = self._ungroup(grouped)
        d1, a1 = self._decode_stage("up1", self.up1, x5, x4, batch, slices, return_aux, trace, features)
        d2, a2 = self._decode_stage("up2", self.up2, d1, x3, batch, slices, return_aux, trace, features)
        d3, a3 = self._decode_stage("up3", self.up3, d2, x2, batch, slices, return_aux, trace, features)
        d4, a4 = self._decode_stage("up4", self.up4, d3, x1, batch, slices, return_aux, trace, features)
        logits = self.outc(d4)
        grouped_logits = self._group(logits, batch, slices)
        self._record_shape(trace, ("output", "logits"), grouped_logits, "B,S,K,H,W"); self._capture(features, ("output", "logits"), grouped_logits)
        if not (return_aux or return_trace or capture_features): return logits
        payload = {
            "gsrm": gsrm_aux,
            "attention": {
                "bottleneck": bottleneck_aux,
                "up1": a1,
                "up2": a2,
                "up3": a3,
                "up4": a4,
            },
            "shape_trace": trace,
        }
        if features is not None: payload["features"] = features
        return logits, payload

    def forward_grouped(self, x: torch.Tensor, **kwargs) -> TensorOrOutput:
        if x.ndim != 5:
            raise ValueError("forward_grouped expects [B,S,C,H,W].")
        batch, slices, channels, height, width = x.shape
        result = self.forward(x.reshape(batch * slices, channels, height, width), slice_count=slices, **kwargs)
        if isinstance(result, tuple):
            logits, payload = result
            return logits.reshape(batch, slices, self.num_classes, *logits.shape[-2:]), payload
        return result.reshape(batch, slices, self.num_classes, *result.shape[-2:])
