from __future__ import annotations

import math

import torch
from torch import nn

from .model_registry import (
    CHANNEL_POOLERS,
    CONTEXT_ENCODERS,
    MODELS,
    SPATIAL_ENCODERS,
    TEMPORAL_ENCODERS,
)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(value, float(scale))


@TEMPORAL_ENCODERS.register("multiscale")
class MultiScaleTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_streams: int,
        model_dim: int,
        branch_width: int = 24,
        kernels: list[int] | tuple[int, ...] = (7, 25, 51),
        dilations: list[int] | tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have equal length")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_streams, branch_width, kernel_size=kernel, stride=2, padding=kernel // 2),
                    nn.BatchNorm1d(branch_width),
                    nn.GELU(),
                    nn.Conv1d(
                        branch_width,
                        branch_width,
                        kernel_size=5,
                        padding=2 * dilation,
                        dilation=dilation,
                        groups=branch_width,
                    ),
                    nn.BatchNorm1d(branch_width),
                    nn.GELU(),
                )
                for kernel, dilation in zip(kernels, dilations)
            ]
        )
        self.project = nn.Sequential(
            nn.Conv1d(branch_width * len(kernels), model_dim, kernel_size=1),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(signal) for branch in self.branches], dim=1))


@SPATIAL_ENCODERS.register("identity")
class IdentitySpatialEncoder(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(self, nodes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return nodes * mask.unsqueeze(-1), {}


@SPATIAL_ENCODERS.register("set_attention")
class SetAttentionSpatialEncoder(nn.Module):
    """Permutation-equivariant electrode encoder without electrode identities."""

    def __init__(self, model_dim: int, heads: int = 4, layers: int = 2, dropout: float = 0.3, **_: object) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, nodes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.encoder(nodes, src_key_padding_mask=~mask)
        return output * mask.unsqueeze(-1), {}


@SPATIAL_ENCODERS.register("sparse_graph")
class SparseGraphSpatialEncoder(nn.Module):
    def __init__(self, model_dim: int, heads: int = 4, top_k: int = 12, dropout: float = 0.3, **_: object) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = model_dim // heads
        self.top_k = top_k
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, nodes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, channels, dim = nodes.shape
        qkv = self.qkv(nodes).reshape(batch, channels, 3, self.heads, self.head_dim)
        query, key, value = (item.transpose(1, 2) for item in qkv.unbind(dim=2))
        logits = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        if 0 < self.top_k < channels:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            sparse = torch.full_like(logits, torch.finfo(logits.dtype).min)
            sparse.scatter_(-1, top_indices, top_values)
            logits = sparse
        weights = self.dropout(torch.softmax(logits, dim=-1))
        attended = (weights @ value).transpose(1, 2).reshape(batch, channels, dim)
        output = self.norm(nodes + self.output(attended)) * mask.unsqueeze(-1)
        return output, {"graph_attention": weights}


@CHANNEL_POOLERS.register("dynamic_set")
class DynamicSetChannelPool(nn.Module):
    """Time-varying, permutation-invariant channel pooling."""

    def __init__(self, model_dim: int, hidden_dim: int | None = None, **_: object) -> None:
        super().__init__()
        hidden = hidden_dim or max(16, model_dim // 2)
        self.static_score = nn.Sequential(nn.Linear(model_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.dynamic_score = nn.Conv1d(model_dim, 1, kernel_size=1)

    def forward(
        self, temporal: torch.Tensor, spatial_nodes: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, dim, times = temporal.shape
        static = self.static_score(spatial_nodes).squeeze(-1).unsqueeze(-1)
        dynamic = self.dynamic_score(temporal.reshape(batch * channels, dim, times)).reshape(batch, channels, times)
        logits = (static + dynamic).masked_fill(~mask.unsqueeze(-1), torch.finfo(temporal.dtype).min)
        weights = torch.softmax(logits, dim=1)
        sequence = torch.einsum("bct,bcdt->btd", weights, temporal)
        return sequence, weights


@CHANNEL_POOLERS.register("masked_mean")
class MaskedMeanChannelPool(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(
        self, temporal: torch.Tensor, spatial_nodes: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del spatial_nodes
        weights = mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1)
        sequence = torch.einsum("bc,bcdt->btd", weights, temporal)
        return sequence, weights.unsqueeze(-1).expand(-1, -1, temporal.shape[-1])


@CONTEXT_ENCODERS.register("transformer")
class TransformerContextEncoder(nn.Module):
    def __init__(self, model_dim: int, heads: int = 4, layers: int = 2, dropout: float = 0.3, **_: object) -> None:
        super().__init__()
        self.local_position = nn.Conv1d(model_dim, model_dim, kernel_size=7, padding=3, groups=model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        local = self.local_position(sequence.transpose(1, 2)).transpose(1, 2)
        return self.encoder(sequence + local)


@CONTEXT_ENCODERS.register("temporal_conv")
class TemporalConvContextEncoder(nn.Module):
    def __init__(self, model_dim: int, layers: int = 3, dropout: float = 0.3, **_: object) -> None:
        super().__init__()
        blocks = []
        for layer_index in range(layers):
            dilation = 2**layer_index
            blocks.extend(
                [
                    nn.Conv1d(model_dim, model_dim, 5, padding=2 * dilation, dilation=dilation, groups=model_dim),
                    nn.Conv1d(model_dim, model_dim, 1),
                    nn.BatchNorm1d(model_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.network(sequence.transpose(1, 2)).transpose(1, 2)


class AttentionTimePool(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(model_dim, model_dim // 2), nn.Tanh(), nn.Linear(model_dim // 2, 1))

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=-1)
        return torch.einsum("bt,btd->bd", weights, sequence), weights


@MODELS.register("modular_phonology")
class ModularPhonologyNetwork(nn.Module):
    def __init__(
        self,
        task_sizes: dict[str, int],
        n_domains: int,
        model_dim: int = 96,
        input_streams: int = 2,
        temporal: dict | str = "multiscale",
        spatial: dict | str = "set_attention",
        channel_pool: dict | str = "dynamic_set",
        context: dict | str = "transformer",
        dropout: float = 0.3,
        channel_dropout: float = 0.2,
        projection_dim: int = 64,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.channel_dropout = float(channel_dropout)
        self.temporal = TEMPORAL_ENCODERS.build(
            temporal, input_streams=input_streams, model_dim=model_dim, dropout=dropout
        )
        self.spatial = SPATIAL_ENCODERS.build(spatial, model_dim=model_dim, dropout=dropout)
        self.channel_pool = CHANNEL_POOLERS.build(channel_pool, model_dim=model_dim)
        self.context = CONTEXT_ENCODERS.build(context, model_dim=model_dim, dropout=dropout)
        self.time_pool = AttentionTimePool(model_dim)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, size))
                for name, size in task_sizes.items()
            }
        )
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, projection_dim)
        )
        self.domain_head = (
            nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, n_domains))
            if n_domains > 1
            else None
        )

    def _drop_channels(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.channel_dropout <= 0:
            return mask
        keep = torch.rand(mask.shape, device=mask.device) >= self.channel_dropout
        output = mask & keep
        empty = ~output.any(dim=1)
        if empty.any():
            first_valid = mask.float().argmax(dim=1)
            output[empty, first_valid[empty]] = True
        return output

    def forward(
        self, signal: torch.Tensor, channel_mask: torch.Tensor, grl_scale: float = 0.0
    ) -> dict[str, torch.Tensor]:
        batch, channels, streams, times = signal.shape
        active_mask = self._drop_channels(channel_mask)
        signal = signal * active_mask[:, :, None, None]
        temporal = self.temporal(signal.reshape(batch * channels, streams, times))
        _, dim, reduced_time = temporal.shape
        temporal = temporal.reshape(batch, channels, dim, reduced_time)
        nodes = temporal.mean(dim=-1)
        spatial_nodes, spatial_outputs = self.spatial(nodes, active_mask)
        sequence, channel_attention = self.channel_pool(temporal, spatial_nodes, active_mask)
        sequence = self.context(sequence)
        embedding, temporal_attention = self.time_pool(sequence)
        output = {name: head(embedding) for name, head in self.heads.items()}
        output.update(
            {
                "embedding": embedding,
                "contrastive_embedding": nn.functional.normalize(self.projection(embedding), dim=-1),
                "channel_attention": channel_attention,
                "temporal_attention": temporal_attention,
            }
        )
        output.update(spatial_outputs)
        if self.domain_head is not None:
            output["domain"] = self.domain_head(gradient_reverse(embedding, grl_scale))
        return output

