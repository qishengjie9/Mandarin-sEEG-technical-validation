import torch

from mandarin_seeg.model_registry import build_model as build_cross_participant_model
from mandarin_seeg.models import STAGMandarin


def test_within_participant_model_shapes() -> None:
    model = STAGMandarin(
        input_streams=2,
        n_classes=4,
        branch_width=8,
        model_dim=32,
        graph_heads=4,
        graph_top_k=4,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.1,
    )
    signal = torch.randn(2, 8, 2, 250)
    mask = torch.tensor([[1] * 8, [1] * 6 + [0] * 2], dtype=torch.bool)
    output = model(signal, mask)
    assert output["tone"].shape == (2, 4)
    assert output["embedding"].shape == (2, 32)
    assert output["channel_attention"].shape == (2, 8)


def test_cross_participant_model_shapes() -> None:
    config = {
        "name": "modular_phonology",
        "model_dim": 32,
        "input_streams": 2,
        "dropout": 0.1,
        "channel_dropout": 0.1,
        "projection_dim": 16,
        "temporal": {
            "name": "multiscale",
            "branch_width": 8,
            "kernels": [7, 25, 51],
            "dilations": [1, 2, 4],
        },
        "spatial": {"name": "set_attention", "heads": 4, "layers": 1},
        "channel_pool": {"name": "dynamic_set"},
        "context": {"name": "transformer", "heads": 4, "layers": 1},
    }
    model = build_cross_participant_model(
        config,
        task_sizes={"tone": 4, "initial": 22, "final": 39},
        n_domains=3,
    )
    signal = torch.randn(2, 8, 2, 250)
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3], dtype=torch.bool)
    output = model(signal, mask, grl_scale=0.5)
    assert output["tone"].shape == (2, 4)
    assert output["initial"].shape == (2, 22)
    assert output["final"].shape == (2, 39)
    assert output["domain"].shape == (2, 3)
    assert output["contrastive_embedding"].shape == (2, 16)
