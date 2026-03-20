import math
import re
from copy import deepcopy

import torch
from torch import nn


DEFAULT_PEFT_CFG = {
    "method": "lora",
    "target_modules": ["qkv", "proj", "fc1", "fc2"],
    "freeze_backbone": True,
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
}

HIGH_LEVEL_TO_SUBMODULES = {
    "attn": ["qkv", "proj"],
    "mlp": ["fc1", "fc2"],
}


def _normalize_target_modules(value):
    if value is None:
        return list(DEFAULT_PEFT_CFG["target_modules"])
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(DEFAULT_PEFT_CFG["target_modules"])
        if "," in stripped:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return stripped
    if isinstance(value, (list, tuple)):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if not normalized:
            return list(DEFAULT_PEFT_CFG["target_modules"])
        if len(normalized) == 1 and any(ch in normalized[0] for ch in "^$.*+?[](){}|\\"):
            return normalized[0]
        return normalized
    raise TypeError(f"Unsupported target_modules type: {type(value)!r}")


def resolve_peft_cfg(cfg, args):
    raw_yaml_peft = cfg.get("peft", {})
    peft_cfg = dict(DEFAULT_PEFT_CFG)
    peft_cfg.update(raw_yaml_peft)

    if "method" in peft_cfg and str(peft_cfg["method"]).lower() != "lora":
        raise ValueError("Only LoRA is supported in UniMatch-V2 PEFT for now.")
    peft_cfg["method"] = "lora"

    if getattr(args, "peft_target_modules", None) is not None:
        peft_cfg["target_modules"] = args.peft_target_modules
    if getattr(args, "freeze_backbone", None) is not None:
        peft_cfg["freeze_backbone"] = args.freeze_backbone
    if getattr(args, "lora_r", None) is not None:
        peft_cfg["r"] = args.lora_r
    if getattr(args, "lora_alpha", None) is not None:
        peft_cfg["lora_alpha"] = args.lora_alpha
    if getattr(args, "lora_dropout", None) is not None:
        peft_cfg["lora_dropout"] = args.lora_dropout

    peft_cfg["target_modules"] = _normalize_target_modules(peft_cfg.get("target_modules"))
    cfg["peft"] = peft_cfg
    return peft_cfg


class LoraAdapter(nn.Module):
    def __init__(self, in_features, out_features, r=32, lora_alpha=64, dropout=0.1):
        super().__init__()
        self.r = r
        self.scaling = lora_alpha / max(r, 1)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class WarpLinearWithLora(nn.Module):
    def __init__(self, base_layer, adapter):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x, *args, **kwargs):
        return self.base_layer(x, *args, **kwargs) + self.adapter(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)


def _expand_target_modules(targets):
    if isinstance(targets, str):
        return targets
    expanded = []
    for target in targets:
        target = str(target)
        if target in HIGH_LEVEL_TO_SUBMODULES:
            expanded.extend(HIGH_LEVEL_TO_SUBMODULES[target])
            expanded.append(target)
        else:
            expanded.append(target)
    result = []
    for item in expanded:
        if item not in result:
            result.append(item)
    return result


def _matches_module_key(key, targets):
    if isinstance(targets, str):
        return re.fullmatch(targets, key) is not None
    return any(key.endswith(target_key) for target_key in targets)


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def apply_lora(model, peft_cfg):
    targets = _expand_target_modules(peft_cfg["target_modules"])
    wrapped = False
    visited = set()
    key_list = [key for key, _ in model.named_modules() if key]

    for key in key_list:
        if key in visited or not _matches_module_key(key, targets):
            continue

        parent, target, target_name = _get_submodules(model, key)

        if target_name in HIGH_LEVEL_TO_SUBMODULES:
            for child_name in HIGH_LEVEL_TO_SUBMODULES[target_name]:
                child = getattr(target, child_name, None)
                if not isinstance(child, nn.Linear):
                    continue
                setattr(
                    target,
                    child_name,
                    WarpLinearWithLora(
                        child,
                        LoraAdapter(
                            child.in_features,
                            child.out_features,
                            r=peft_cfg["r"],
                            lora_alpha=peft_cfg["lora_alpha"],
                            dropout=peft_cfg["lora_dropout"],
                        ),
                    ),
                )
                visited.add(f"{key}.{child_name}")
                wrapped = True
            visited.add(key)
            continue

        if not isinstance(target, nn.Linear):
            visited.add(key)
            continue

        setattr(
            parent,
            target_name,
            WarpLinearWithLora(
                target,
                LoraAdapter(
                    target.in_features,
                    target.out_features,
                    r=peft_cfg["r"],
                    lora_alpha=peft_cfg["lora_alpha"],
                    dropout=peft_cfg["lora_dropout"],
                ),
            ),
        )
        visited.add(key)
        wrapped = True

    if not wrapped:
        raise ValueError(
            f"Target modules {peft_cfg['target_modules']} not found in the base model."
        )
    return model


def build_peft_model(model, peft_cfg):
    if peft_cfg.get("freeze_backbone", True) and hasattr(model, "lock_backbone"):
        model.lock_backbone()
    elif peft_cfg.get("freeze_backbone", True):
        for param in model.backbone.parameters():
            param.requires_grad = False

    return apply_lora(model, peft_cfg)


def build_optimizer(model, cfg):
    trainable_backbone_params = []
    trainable_non_backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            trainable_backbone_params.append(param)
        else:
            trainable_non_backbone_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": trainable_backbone_params, "lr": cfg["lr"]},
            {"params": trainable_non_backbone_params, "lr": cfg["lr"] * cfg["lr_multi"]},
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def show_trainable_parameters(model, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = 100 * trainable_params / total_params if total_params > 0 else 0

    logger.info("Trainable params: %s / %s (%.2f%%)", trainable_params, total_params, percentage)


def clone_ema_model(model):
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    return model_ema
