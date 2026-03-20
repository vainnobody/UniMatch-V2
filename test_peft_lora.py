from types import SimpleNamespace

import torch
from torch import nn

from model.peft_lora import (
    WarpLinearWithLora,
    build_optimizer,
    build_peft_model,
    resolve_peft_cfg,
)


def make_args(**kwargs):
    defaults = {
        "peft_target_modules": None,
        "freeze_backbone": None,
        "lora_r": None,
        "lora_alpha": None,
        "lora_dropout": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)


class DummyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()
        self.mlp = DummyMlp()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([DummyBlock()])
        self.head = nn.Linear(8, 2)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


def test_resolve_peft_cfg_uses_defaults():
    cfg = {}
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    assert peft_cfg["method"] == "lora"
    assert peft_cfg["target_modules"] == ["qkv", "proj", "fc1", "fc2"]
    assert peft_cfg["freeze_backbone"] is True


def test_resolve_peft_cfg_applies_cli_overrides():
    cfg = {"peft": {"r": 8, "freeze_backbone": True}}
    peft_cfg = resolve_peft_cfg(
        cfg,
        make_args(
            peft_target_modules=["attn"],
            freeze_backbone=False,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        ),
    )
    assert peft_cfg["target_modules"] == ["attn"]
    assert peft_cfg["freeze_backbone"] is False
    assert peft_cfg["r"] == 16
    assert peft_cfg["lora_alpha"] == 32
    assert abs(peft_cfg["lora_dropout"] - 0.05) < 1e-8


def test_build_peft_model_wraps_target_linear_layers():
    model = DummyModel()
    peft_cfg = resolve_peft_cfg({"peft": {"target_modules": ["attn", "mlp"]}}, make_args())
    model = build_peft_model(model, peft_cfg)
    block = model.backbone.blocks[0]
    assert isinstance(block.attn.qkv, WarpLinearWithLora)
    assert isinstance(block.attn.proj, WarpLinearWithLora)
    assert isinstance(block.mlp.fc1, WarpLinearWithLora)
    assert isinstance(block.mlp.fc2, WarpLinearWithLora)


def test_build_optimizer_splits_backbone_and_non_backbone_params():
    model = DummyModel()
    peft_cfg = resolve_peft_cfg({"peft": {"target_modules": ["mlp"]}}, make_args(freeze_backbone=False))
    model = build_peft_model(model, peft_cfg)
    optimizer = build_optimizer(model, {"lr": 1e-4, "lr_multi": 10.0})

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-3
    assert len(optimizer.param_groups[0]["params"]) > 0
    assert len(optimizer.param_groups[1]["params"]) > 0


def test_freeze_backbone_keeps_only_lora_and_head_trainable():
    model = DummyModel()
    peft_cfg = resolve_peft_cfg({"peft": {"target_modules": ["mlp"], "freeze_backbone": True}}, make_args())
    model = build_peft_model(model, peft_cfg)

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert any("adapter" in name for name in trainable)
    assert any(name.startswith("head.") for name in trainable)
    assert not any(name.endswith("base_layer.weight") for name in trainable)
