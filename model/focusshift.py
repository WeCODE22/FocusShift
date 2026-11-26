"""FocusShift: adaptive local focus for efficient vision.

Builds on LSNet's see-large / focus-small idea, but makes the small dynamic
mix spatially movable via a soft focus map (FocusShiftConv). A cheap static
depthwise 3x3 covers low-focus locations. Baseline LSNet remains the fixed-focus
comparison; FocusShift is the proposed method.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite
from timm.models.registry import register_model

from .lsnet import (
    Attention,
    BN_Linear,
    Conv2d_BN,
    FFN,
    LKP,
    Residual,
    RepVGGDW,
)
from .ska import SKA


class FocusPredictor(nn.Module):
    """Soft focus map in (0, 1). Bias init keeps early behavior ≈ full LSConv."""

    def __init__(self, dim: int, init_bias: float = 4.0):
        super().__init__()
        mid = max(dim // 4, 8)
        self.net = nn.Sequential(
            nn.Conv2d(dim, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class FocusShiftConv(nn.Module):
    """See large everywhere; move the small dynamic mix with a focus map.

    At initialization focus≈1 and the static BN scale is 0, so the module
    numerically matches LSConv. Training then learns where to down-weight the
    expensive dynamic path and rely on the static fallback.
    """

    def __init__(self, dim: int, lks: int = 7, sks: int = 3, groups: int = 8,
                 hard_gate_thresh: float = 0.15):
        super().__init__()
        self.lkp = LKP(dim, lks=lks, sks=sks, groups=groups)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)
        self.static = Conv2d_BN(dim, dim, ks=sks, pad=(sks - 1) // 2, groups=dim)
        nn.init.zeros_(self.static.bn.weight)
        nn.init.zeros_(self.static.bn.bias)
        self.focus = FocusPredictor(dim)
        self.hard_gate_thresh = hard_gate_thresh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        focus = self.focus(x)

        if (not self.training) and bool((focus.mean() < self.hard_gate_thresh).item()):
            return self.static(x) + x

        dynamic = self.bn(self.ska(x, self.lkp(x)))
        static = self.static(x)
        # Convex blend; focus=1 → identical to LSConv (+ residual).
        return focus * dynamic + (1.0 - focus) * static + x


class FocusShiftBlock(nn.Module):
    def __init__(self, ed, kd, nh=8, ar=4, resolution=14, stage=-1, depth=-1, adaptive=True):
        super().__init__()
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=resolution))
            elif adaptive:
                self.mixer = FocusShiftConv(ed)
            else:
                from .lsnet import LSConv
                self.mixer = LSConv(ed)
        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))


class FocusShift(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=8,
        in_chans=3,
        num_classes=1000,
        embed_dim=None,
        key_dim=None,
        depth=None,
        num_heads=None,
        distillation=False,
        adaptive=True,
    ):
        super().__init__()
        embed_dim = embed_dim or [64, 128, 256, 384]
        key_dim = key_dim or [16, 16, 16, 16]
        depth = depth or [0, 2, 8, 10]
        num_heads = num_heads or [3, 3, 3, 4]

        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1),
        )

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]

        for i, (ed, kd, dpth, nh, ar) in enumerate(
            zip(embed_dim, key_dim, depth, num_heads, attn_ratio)
        ):
            for d in range(dpth):
                blocks[i].append(
                    FocusShiftBlock(ed, kd, nh, ar, resolution, stage=i, depth=d, adaptive=adaptive)
                )
            if i != len(depth) - 1:
                resolution = (resolution - 1) // 2 + 1
                blocks[i + 1].append(
                    Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1, groups=embed_dim[i])
                )
                blocks[i + 1].append(Conv2d_BN(embed_dim[i], embed_dim[i + 1], ks=1, stride=1, pad=0))

        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


@register_model
def focusshift_t(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    """FocusShift-T: same width as LSNet-T, movable focus."""
    return FocusShift(
        num_classes=num_classes,
        distillation=distillation,
        img_size=224,
        patch_size=8,
        embed_dim=[64, 128, 256, 384],
        depth=[0, 2, 8, 10],
        num_heads=[3, 3, 3, 4],
        adaptive=True,
        **kwargs,
    )


@register_model
def focusshift_micro(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    """Parameter-matched compact FocusShift."""
    return FocusShift(
        num_classes=num_classes,
        distillation=distillation,
        img_size=224,
        patch_size=8,
        embed_dim=[48, 96, 192, 288],
        key_dim=[16, 16, 16, 16],
        depth=[0, 2, 4, 6],
        num_heads=[2, 2, 3, 3],
        adaptive=True,
        **kwargs,
    )


@register_model
def lsnet_micro(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    """Compact plain LSNet for fair size comparison."""
    return FocusShift(
        num_classes=num_classes,
        distillation=distillation,
        img_size=224,
        patch_size=8,
        embed_dim=[48, 96, 192, 288],
        key_dim=[16, 16, 16, 16],
        depth=[0, 2, 4, 6],
        num_heads=[2, 2, 3, 3],
        adaptive=False,
        **kwargs,
    )
