from __future__ import annotations

import math

import torch
from torch import nn


class MultiScaleTemporalStem(nn.Module):
    """Shared per-electrode temporal encoder with three receptive fields."""

    def __init__(self, input_streams: int, branch_width: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        kernels = (7, 25, 51)
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
                for kernel, dilation in zip(kernels, (1, 2, 4))
            ]
        )
        self.project = nn.Sequential(
            nn.Conv1d(branch_width * len(kernels), model_dim, kernel_size=1),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class AdaptiveGraphAttention(nn.Module):
    """Trial-adaptive sparse attention over an unordered set of electrodes."""

    def __init__(self, model_dim: int, heads: int, top_k: int, dropout: float) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by graph heads")
        self.heads = heads
        self.head_dim = model_dim // heads
        self.top_k = top_k
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, nodes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, dim = nodes.shape
        qkv = self.qkv(nodes).reshape(batch, channels, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))  # [B,H,C,D]
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        valid_keys = mask[:, None, None, :]
        logits = logits.masked_fill(~valid_keys, torch.finfo(logits.dtype).min)
        if self.top_k > 0 and self.top_k < channels:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            sparse = torch.full_like(logits, torch.finfo(logits.dtype).min)
            sparse.scatter_(-1, top_indices, top_values)
            logits = sparse
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, channels, dim)
        attended = attended * mask.unsqueeze(-1)
        return self.norm(nodes + self.output(attended)), weights


class AttentionPool(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.Tanh(),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, nodes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(nodes).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bc,bcd->bd", weights, nodes)
        return pooled, weights


class MultiScaleTCN(nn.Module):
    """A non-graph baseline using the same multi-scale temporal stem."""

    def __init__(self, input_streams: int = 2, n_classes: int = 4, branch_width: int = 24, model_dim: int = 96, dropout: float = 0.35, **_: object) -> None:
        super().__init__()
        self.stem = MultiScaleTemporalStem(input_streams, branch_width, model_dim, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, n_classes))

    def forward(self, signal: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, channels, streams, times = signal.shape
        features = self.stem(signal.reshape(batch * channels, streams, times))
        features = features.mean(dim=-1).reshape(batch, channels, -1)
        mask = channel_mask.unsqueeze(-1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return {"tone": self.classifier(pooled), "embedding": pooled}


class STAGMandarin(nn.Module):
    """Multi-scale temporal, adaptive electrode-graph, Transformer decoder."""

    def __init__(
        self,
        input_streams: int = 2,
        n_classes: int = 4,
        branch_width: int = 24,
        model_dim: int = 96,
        graph_heads: int = 4,
        graph_top_k: int = 12,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.35,
        **_: object,
    ) -> None:
        super().__init__()
        self.stem = MultiScaleTemporalStem(input_streams, branch_width, model_dim, dropout)
        self.graph = AdaptiveGraphAttention(model_dim, graph_heads, graph_top_k, dropout)
        self.channel_pool = AttentionPool(model_dim)
        self.temporal_position = nn.Conv1d(model_dim, model_dim, kernel_size=7, padding=3, groups=model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.temporal_pool = AttentionPool(model_dim)
        self.tone_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, n_classes),
        )

    def forward(self, signal: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, channels, streams, times = signal.shape
        temporal = self.stem(signal.reshape(batch * channels, streams, times))
        _, dim, reduced_time = temporal.shape
        temporal = temporal.reshape(batch, channels, dim, reduced_time)

        node_summary = temporal.mean(dim=-1)
        graph_nodes, graph_weights = self.graph(node_summary, channel_mask)
        _, channel_weights = self.channel_pool(graph_nodes, channel_mask)
        sequence = torch.einsum("bc,bcdt->btd", channel_weights, temporal)
        position = self.temporal_position(sequence.transpose(1, 2)).transpose(1, 2)
        sequence = self.transformer(sequence + position)
        time_mask = torch.ones(batch, reduced_time, dtype=torch.bool, device=signal.device)
        embedding, temporal_weights = self.temporal_pool(sequence, time_mask)
        return {
            "tone": self.tone_head(embedding),
            "embedding": embedding,
            "channel_attention": channel_weights,
            "graph_attention": graph_weights,
            "temporal_attention": temporal_weights,
        }


def build_model(config: dict) -> nn.Module:
    name = config.get("name", "stag").lower()
    kwargs = {key: value for key, value in config.items() if key != "name"}
    if name == "stag":
        return STAGMandarin(**kwargs)
    if name in {"tcn", "multiscale_tcn"}:
        return MultiScaleTCN(**kwargs)
    raise ValueError(f"unknown model: {name}")

