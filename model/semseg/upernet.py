import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6), dropout=0.1):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        ppm_outs = [x] + [
            F.interpolate(stage(x), size=x.shape[2:], mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(ppm_outs, dim=1))


class Feature2Pyramid(nn.Module):
    def __init__(self, embed_dim, rescales=(4, 2, 1, 0.5)):
        super().__init__()
        self.rescales = rescales
        self.ops = nn.ModuleList()
        for r in rescales:
            if r == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2, bias=False),
                        nn.BatchNorm2d(embed_dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    )
                )
            elif r == 2:
                self.ops.append(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            elif r == 1:
                self.ops.append(nn.Identity())
            elif r == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise KeyError(f"Invalid rescale factor: {r}")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        return tuple(op(feat) for op, feat in zip(self.ops, inputs))


class UPerNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        ppm_channels=512,
        fpn_channels=512,
        num_classes=21,
        dropout=0.1,
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=False),
                )
                for ch in in_channels[:-1]
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=False),
                )
                for _ in in_channels[:-1]
            ]
        )
        self.ppm = PPM(in_channels[-1], ppm_channels, pool_scales=(1, 2, 3, 6), dropout=dropout)
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_channels, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=dropout),
        )
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

    def forward(self, feats, return_feats=False):
        laterals = [l_conv(feats[i]) for i, l_conv in enumerate(self.lateral_convs)]
        top = self.ppm_bottleneck(self.ppm(feats[-1]))
        laterals.append(top)

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )
            laterals[i - 1] = laterals[i - 1] + up

        fpn_outs = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        fpn_outs.append(laterals[-1])
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False
            )

        out = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        logits = self.classifier(out)
        if return_feats:
            return logits, out
        return logits


class UperNet(nn.Module):
    def __init__(
        self,
        encoder_size="base",
        nclass=21,
        fpn_channels=256,
        use_bn=True,
        backbone_version="dinov2",
        **kwargs,
    ):
        del use_bn, kwargs
        super().__init__()
        self.intermediate_layer_idx_v2 = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        self.intermediate_layer_idx_v3 = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [5, 11, 17, 23],
            "so400m": [6, 13, 20, 26],
            "huge": [7, 15, 23, 31],
            "giant": [9, 19, 29, 39],
        }

        self.encoder_size = encoder_size
        self.backbone_version = backbone_version
        if backbone_version == "dinov2":
            self.backbone = DINOv2(model_name=encoder_size)
            self.intermediate_layer_idx = self.intermediate_layer_idx_v2
        elif backbone_version == "dinov3":
            self.backbone = DINOv3(model_name=encoder_size)
            self.intermediate_layer_idx = self.intermediate_layer_idx_v3
        else:
            raise ValueError(f"Unknown backbone version: {backbone_version}.")

        embed_dim = self.backbone.embed_dim
        self.neck = Feature2Pyramid(embed_dim=embed_dim)
        self.decoder = UPerNetDecoder(
            in_channels=[embed_dim] * 4,
            fpn_channels=fpn_channels,
            num_classes=nclass,
        )
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    @property
    def head(self):
        return nn.ModuleList([self.neck, self.decoder])

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _extract_feature_maps(self, x):
        patch_size = self.backbone.patch_size
        bsz, _, h, w = x.shape
        patch_h, patch_w = h // patch_size, w // patch_size
        features = self.backbone.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder_size])
        return tuple(
            feat.permute(0, 2, 1).reshape(bsz, -1, patch_h, patch_w).float().contiguous()
            for feat in features
        )

    def forward(self, x, comp_drop=False):
        feats = self._extract_feature_maps(x)
        if comp_drop:
            bs, dim = feats[0].shape[0], feats[0].shape[1]
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(x.device) * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            kept_indexes = torch.randperm(bs // 2, device=x.device)[: int(bs // 2 * 0.5)]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2), dim=0)
            feats = tuple(
                feat * dropout_mask.unsqueeze(-1).unsqueeze(-1).to(feat.device) for feat in feats
            )

        pyramid_feats = self.neck(feats)
        logits = self.decoder(pyramid_feats)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
