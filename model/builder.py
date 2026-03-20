import torch

from model.semseg.dpt import DPT
from model.semseg.dpt_scalematch import DPT_ScaleMatch
from model.semseg.upernet import UperNet
from model.semseg.upernet_scalematch import UperNet_ScaleMatch


MODEL_CONFIGS = {
    'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def get_backbone_info(backbone_name):
    backbone_version, encoder_size = backbone_name.split('_', 1)
    if backbone_version not in {'dinov2', 'dinov3'}:
        raise ValueError(f"Unsupported backbone version: {backbone_version}")
    if encoder_size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported backbone size: {encoder_size}")
    patch_size = 14 if backbone_version == 'dinov2' else 16
    return backbone_version, encoder_size, patch_size


def _unwrap_checkpoint_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    for key in ('teacher', 'student', 'model', 'state_dict'):
        candidate = state_dict.get(key)
        if isinstance(candidate, dict):
            state_dict = candidate
            break

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.') :]
        if key.startswith('backbone.'):
            key = key[len('backbone.') :]
        cleaned[key] = value
    return cleaned


def _build_standard_model(cfg):
    backbone_version, encoder_size, patch_size = get_backbone_info(cfg['backbone'])
    model_name = cfg.get('model', 'dpt').lower()
    model_kwargs = {**MODEL_CONFIGS[encoder_size], 'nclass': cfg['nclass']}
    if model_name == 'dpt':
        model = DPT(**model_kwargs, backbone_version=backbone_version)
        eval_multiplier = patch_size
    elif model_name == 'upernet':
        model = UperNet(
            encoder_size=model_kwargs['encoder_size'],
            nclass=cfg['nclass'],
            fpn_channels=cfg.get('fpn_channels', 256),
            backbone_version=backbone_version,
        )
        eval_multiplier = None
    else:
        raise ValueError(f"Unsupported model: {cfg.get('model')}")
    return model, eval_multiplier


def _build_scalematch_model(cfg):
    backbone_version, encoder_size, _ = get_backbone_info(cfg['backbone'])
    model_name = cfg.get('model', 'dpt').lower()
    model_kwargs = {**MODEL_CONFIGS[encoder_size], 'nclass': cfg['nclass']}
    if model_name == 'dpt':
        model = DPT_ScaleMatch(**model_kwargs, backbone_version=backbone_version)
    elif model_name == 'upernet':
        model = UPerNet_ScaleMatch(
            encoder_size=model_kwargs['encoder_size'],
            nclass=cfg['nclass'],
            fpn_channels=cfg.get('fpn_channels', 256),
            backbone_version=backbone_version,
        )
    else:
        raise ValueError(f"Unsupported ScaleMatch model: {cfg.get('model')}")
    return model


def build_model_from_cfg(cfg):
    return _build_standard_model(cfg)


def build_scalematch_model_from_cfg(cfg):
    return _build_scalematch_model(cfg)


def load_backbone_weights(model, cfg):
    pretrained_path = f'./pretrained/{cfg["backbone"]}.pth'
    state_dict = torch.load(pretrained_path, map_location='cpu')
    state_dict = _unwrap_checkpoint_state_dict(state_dict)
    return model.backbone.load_state_dict(state_dict, strict=False)
