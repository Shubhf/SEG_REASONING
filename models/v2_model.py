"""
HiReMed V2 — Model components with pretrained encoder support.

Supported encoders (via config.encoder_name):
  ResNet family   : resnet18, resnet34, resnet50, resnet101, resnet152
  ConvNeXT family : convnext_tiny, convnext_small, convnext_base, convnext_large

All encoders return 4 multi-scale feature maps (s1, s2, s3, s4) at spatial
resolutions H/4, H/8, H/16, H/32 respectively.
Channel dimensions vary per encoder and are auto-detected at build time.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Encoder registry  (name → channel tuple for s1, s2, s3, s4)
# ---------------------------------------------------------------------------

ENCODER_CHANNELS: Dict[str, tuple] = {
    # ResNet family  (all share the same bottleneck channels)
    "resnet18":  (64,   128,  256,  512),
    "resnet34":  (64,   128,  256,  512),
    "resnet50":  (256,  512,  1024, 2048),
    "resnet101": (256,  512,  1024, 2048),
    "resnet152": (256,  512,  1024, 2048),
    # ConvNeXT family  (verified by forward-pass shape inspection)
    "convnext_tiny":  (96,   192,  384,  768),
    "convnext_small": (96,   192,  384,  768),
    "convnext_base":  (128,  256,  512,  1024),
    "convnext_large": (192,  384,  768,  1536),
}

ENCODER_PARAMS_M: Dict[str, float] = {
    "resnet18":     11.69,
    "resnet34":     21.80,
    "resnet50":     25.56,
    "resnet101":    44.55,
    "resnet152":    60.19,
    "convnext_tiny":   28.59,
    "convnext_small":  50.22,
    "convnext_base":   88.59,
    "convnext_large": 197.77,
}

# Pretrained weight enums from torchvision (note: ConvNeXt uses lowercase 'xt' in class names)
ENCODER_WEIGHTS_MAP: Dict[str, object] = {
    "resnet18":  tv_models.ResNet18_Weights.DEFAULT,
    "resnet34":  tv_models.ResNet34_Weights.DEFAULT,
    "resnet50":  tv_models.ResNet50_Weights.DEFAULT,
    "resnet101": tv_models.ResNet101_Weights.DEFAULT,
    "resnet152": tv_models.ResNet152_Weights.DEFAULT,
    "convnext_tiny":  tv_models.ConvNeXt_Tiny_Weights.DEFAULT,
    "convnext_small": tv_models.ConvNeXt_Small_Weights.DEFAULT,
    "convnext_base":  tv_models.ConvNeXt_Base_Weights.DEFAULT,
    "convnext_large": tv_models.ConvNeXt_Large_Weights.DEFAULT,
}


def get_encoder_channels(name: str) -> tuple:
    """Return (s1_ch, s2_ch, s3_ch, s4_ch) for the named encoder."""
    if name not in ENCODER_CHANNELS:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Choose from: {list(ENCODER_CHANNELS.keys())}"
        )
    return ENCODER_CHANNELS[name]


# ---------------------------------------------------------------------------
# Pretrained Encoders
# ---------------------------------------------------------------------------

class ResNetEncoder(nn.Module):
    """
    Wraps a torchvision ResNet as a 4-stage feature extractor.

    Returns (s1, s2, s3, s4) at spatial sizes:
        s1: H/4×W/4   s2: H/8×W/8   s3: H/16×W/16   s4: H/32×W/32
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        weights = ENCODER_WEIGHTS_MAP[name] if pretrained else None
        self.backbone = getattr(tv_models, name)(weights=weights)

        # Remove the classification head (avgpool + fc)
        del self.backbone.avgpool
        del self.backbone.fc

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)       # → H/4
        s1 = self.backbone.layer1(x)       # H/4
        s2 = self.backbone.layer2(s1)      # H/8
        s3 = self.backbone.layer3(s2)      # H/16
        s4 = self.backbone.layer4(s3)      # H/32
        return s1, s2, s3, s4


class ConvNeXTEncoder(nn.Module):
    """
    Wraps a torchvision ConvNeXT as a 4-stage feature extractor.

    ConvNeXT ``features`` is a flat Sequential with 8 entries:
        [0] stem conv          (stride=4)  → H/4
        [1] stage_1 blocks     (no ds)     → H/4
        [2] downsample to s2   (stride=2)  → H/8
        [3] stage_2 blocks                → H/8
        [4] downsample to s3   (stride=2)  → H/16
        [5] stage_3 blocks                → H/16
        [6] downsample to s4   (stride=2)  → H/32
        [7] stage_4 blocks                → H/32

    Returns (s1, s2, s3, s4) at spatial sizes:
        s1: H/4×W/4   s2: H/8×W/8   s3: H/16×W/16   s4: H/32×W/32
    """

    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        weights = ENCODER_WEIGHTS_MAP[name] if pretrained else None
        self.backbone = getattr(tv_models, name)(weights=weights)

        # Remove classification head
        del self.backbone.classifier

    def forward(self, x: torch.Tensor):
        feats = self.backbone.features
        # [0] stem conv (stride=4)  → H/4
        # [1] stage_1 blocks (no downsample)  → H/4
        # [2] downsample to stage_2 (stride=2)  → H/8
        # [3] stage_2 blocks  → H/8
        # [4] downsample to stage_3 (stride=2)  → H/16
        # [5] stage_3 blocks  → H/16
        # [6] downsample to stage_4 (stride=2)  → H/32
        # [7] stage_4 blocks  → H/32

        x   = feats[0](x)       # stem → H/4
        s1  = feats[1](x)       # stage_1 → H/4
        x   = feats[2](s1)      # downsample → H/8
        s2  = feats[3](x)       # stage_2 → H/8
        x   = feats[4](s2)      # downsample → H/16
        s3  = feats[5](x)       # stage_3 → H/16
        x   = feats[6](s3)      # downsample → H/32
        s4  = feats[7](x)       # stage_4 → H/32
        return s1, s2, s3, s4


def build_encoder(name: str, pretrained: bool = True):
    """Factory: returns the appropriate encoder module."""
    name_lower = name.lower()
    if name_lower.startswith("resnet"):
        return ResNetEncoder(name, pretrained)
    elif name_lower.startswith("convnext"):
        return ConvNeXTEncoder(name, pretrained)
    else:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Choose from: {list(ENCODER_CHANNELS.keys())}"
        )


# ---------------------------------------------------------------------------
# Building blocks (shared with v1)
# ---------------------------------------------------------------------------

def _safe_groups(channels: int, desired: int = 8) -> int:
    g = min(desired, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class ConvGNAct(nn.Module):
    """Conv → GroupNorm → GELU."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(_safe_groups(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Two-layer residual block with optional stride for downsampling."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, 3, stride, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(out_ch), out_ch),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


# ---------------------------------------------------------------------------
# Low-level spatial refinement  (K_inner times per outer step)
# ---------------------------------------------------------------------------

class LowLevelRefinement(nn.Module):
    """
    Updates z using current h and s2 skip features.
    Input:  z [B, dz, Hz, Wz],  h [B, dh],  s2 [B, s2_ch, H2, W2]
    Output: z [B, dz, Hz, Wz]
    """
    def __init__(self, dz: int, dh: int, s2_ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(dz + dh + s2_ch, dz, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dz, dz, 3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(dz), dz),
        )
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor, h: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        hz, wz = z.shape[-2], z.shape[-1]
        h_map  = h.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hz, wz)
        s2_res = F.interpolate(s2, size=(hz, wz), mode="bilinear", align_corners=False)
        cat    = torch.cat([z, h_map, s2_res], dim=1)
        return self.act(self.conv2(self.conv1(cat)) + z)


# ---------------------------------------------------------------------------
# High-level global transition  (once per outer step)
# ---------------------------------------------------------------------------

class HighLevelTransition(nn.Module):
    """
    Updates h using refined z and s3 context.
    Returns (h_new, mu, logvar) — mu/logvar used for KL regularisation.
    Input:  h [B, dh],  z [B, dz, Hz, Wz],  s3 [B, s3_ch, H3, W3]
    Output: h_new [B, dh],  mu [B, dh],  logvar [B, dh]
    """
    def __init__(self, dh: int, dz: int, s3_ch: int):
        super().__init__()
        self.fc1         = nn.Linear(dh + dz + s3_ch, dh)
        self.fc2         = nn.Linear(dh, dh)
        self.mu_head     = nn.Linear(dh, dh)
        self.logvar_head = nn.Linear(dh, dh)

    def forward(self, h: torch.Tensor, z: torch.Tensor, s3: torch.Tensor):
        z_pool  = z.mean(dim=(2, 3))
        s3_pool = s3.mean(dim=(2, 3))
        u      = F.gelu(self.fc1(torch.cat([h, z_pool, s3_pool], dim=1)))
        u      = self.fc2(u)
        mu     = self.mu_head(u)
        logvar = self.logvar_head(u)
        h_new  = u + mu
        return h_new, mu, logvar


# ---------------------------------------------------------------------------
# Hierarchical decoder
# ---------------------------------------------------------------------------

class UpFuse(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = ResBlock(in_ch + skip_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class Decoder(nn.Module):
    """
    z → fuse(s3) → fuse(s2) → fuse(s1) → 1×1 conv → residual logit delta
    """
    def __init__(self, z_ch: int, s1_ch: int, s2_ch: int, s3_ch: int):
        super().__init__()
        self.fuse3  = UpFuse(z_ch,  s3_ch, s3_ch)
        self.fuse2  = UpFuse(s3_ch, s2_ch, s2_ch)
        self.fuse1  = UpFuse(s2_ch, s1_ch, s1_ch)
        self.refine = ResBlock(s1_ch, s1_ch, stride=1)
        self.out    = nn.Conv2d(s1_ch, 1, 1)

    def forward(self, z, s1, s2, s3, out_size) -> torch.Tensor:
        x = self.fuse3(z, s3)
        x = self.fuse2(x, s2)
        x = self.fuse1(x, s1)
        x = self.out(self.refine(x))
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Uncertainty head
# ---------------------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """Predicts spatial uncertainty in [0,1]; trained to match |y_prob - y_true|."""
    def __init__(self, z_ch: int, s1_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(z_ch + s1_ch, s1_ch, 3, 1, 1),
            nn.Conv2d(s1_ch, 1, 1),
        )

    def forward(self, z: torch.Tensor, s1: torch.Tensor, out_size) -> torch.Tensor:
        z_up = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x    = self.block(torch.cat([z_up, s1], dim=1))
        return torch.sigmoid(F.interpolate(x, size=out_size, mode="bilinear", align_corners=False))


# ---------------------------------------------------------------------------
# Halt head — per-sample scalar in [0, 1]
# ---------------------------------------------------------------------------

class HaltHead(nn.Module):
    """
    Per-sample halt score in [0, 1].
    High score → model believes prediction is good enough to stop.
    Supervised to regress the current step's Dice score.

    Inputs:
        z  [B, dz, Hz, Wz]  spatial latent
        u  [B, 1, H, W]     uncertainty map
        h  [B, dh]          global latent
    Output:
        score [B]  in [0, 1]
    """
    def __init__(self, dz: int, dh: int):
        super().__init__()
        self.z_pool = nn.Sequential(
            ConvGNAct(dz, dz // 2, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dz // 2 + dh + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z_feat   = self.z_pool(z).squeeze(-1).squeeze(-1)
        u_scalar = u.mean(dim=(1, 2, 3))
        feat     = torch.cat([z_feat, h, u_scalar.unsqueeze(1)], dim=1)
        return self.mlp(feat).squeeze(1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HiReMed(nn.Module):
    """
    Hierarchical Recursive Medical Segmentation model.

    Args:
        encoder_name     : one of ENCODER_CHANNELS keys
        encoder_pretrained : load ImageNet-1K pretrained weights?
        base_channels     : latent dimensions (dz, dh) — kept for backwards compat
        latent_dim_h      : high-level global latent dim
        latent_dim_z      : low-level spatial latent dim
        max_steps         : upper bound on refinement steps
        min_steps         : minimum steps before halting allowed
        K_inner           : spatial refinement iterations per outer step
        halt_threshold    : inference-time halt trigger

    forward() returns a list of per-step dicts:
        y_logit     [B, 1, H, W]   accumulated residual logits
        y_prob      [B, 1, H, W]   sigmoid(y_logit)
        uncertainty [B, 1, H, W]   uncertainty map in [0,1]
        halt_score  [B]            halt signal in [0,1]
        mu          [B, dh]        latent mean  (KL)
        logvar      [B, dh]        latent log-variance  (KL)
        step        int            0-indexed step number

    force_steps=None  → normal behaviour:
        training  → always runs max_steps (full deep supervision)
        inference → adaptive halting (exits early per-sample)

    force_steps=1     → single-pass mode: runs exactly 1 outer step regardless.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_pretrained: bool = True,
        base_channels: int = 48,       # kept for compat; unused when pretrained
        latent_dim_h: int = 256,
        latent_dim_z: int = 128,
        max_steps: int = 6,
        min_steps: int = 2,
        K_inner: int = 3,
        halt_threshold: float = 0.82,
    ):
        super().__init__()

        # Store hyper-parameters as attributes (replacing cfg reference)
        self.latent_dim_h     = latent_dim_h
        self.latent_dim_z     = latent_dim_z
        self.max_steps        = max_steps
        self.min_steps        = min_steps
        self.K_inner          = K_inner
        self.halt_threshold   = halt_threshold

        # --- Encoder ---
        dh = latent_dim_h
        dz = latent_dim_z
        s1_ch, s2_ch, s3_ch, s4_ch = get_encoder_channels(encoder_name)

        self.encoder_name       = encoder_name
        self.encoder_pretrained = encoder_pretrained
        self.encoder            = build_encoder(encoder_name, encoder_pretrained)

        # Channel dimensions stored for introspection / logging
        self.s1_ch = s1_ch
        self.s2_ch = s2_ch
        self.s3_ch = s3_ch
        self.s4_ch = s4_ch

        # --- Projection from encoder output to latent z ---
        self.z_proj = nn.Conv2d(s4_ch, dz, 1)

        # --- Recursive refinement modules ---
        self.low  = LowLevelRefinement(dz, dh, s2_ch)
        self.high = HighLevelTransition(dh, dz, s3_ch)

        # --- Decoder + heads ---
        self.decoder  = Decoder(dz, s1_ch, s2_ch, s3_ch)
        self.unc_head = UncertaintyHead(dz, s1_ch)
        self.halt_head = HaltHead(dz, dh)

    def forward(
        self,
        x: torch.Tensor,
        force_steps: Optional[int] = None,
    ) -> List[Dict]:
        B, _, H, W = x.shape
        dev   = x.device
        dtype = x.dtype

        s1, s2, s3, s4 = self.encoder(x)

        z       = self.z_proj(s4)
        h       = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
        y_logit = torch.zeros(B, 1, H, W,               device=dev, dtype=dtype)

        outputs: List[Dict] = []
        active = torch.ones(B, dtype=torch.bool, device=dev)

        # Determine how many steps to run
        if force_steps is not None:
            max_s = force_steps
            use_halting = False
        else:
            max_s = self.max_steps
            use_halting = not self.training  # adaptive only at inference

        for step in range(max_s):

            # 1. Inner spatial refinement
            for _ in range(self.K_inner):
                z = self.low(z, h, s2)

            # 2. Global state update
            h, mu, logvar = self.high(h, z, s3)

            # 3. Residual logit accumulation
            delta   = self.decoder(z, s1, s2, s3, (H, W))
            y_logit = y_logit + delta
            y_prob  = torch.sigmoid(y_logit.float()).to(dtype)

            # 4. Uncertainty map
            u_map = self.unc_head(z, s1, (H, W))

            # 5. Halt score
            halt_score = self.halt_head(z, u_map, h)

            outputs.append({
                "y_logit":     y_logit,
                "y_prob":      y_prob,
                "uncertainty": u_map,
                "halt_score":  halt_score,
                "mu":          mu,
                "logvar":      logvar,
                "step":        step,
            })

            # 6. Adaptive early exit (inference, no force_steps override)
            if use_halting:
                if step + 1 >= self.min_steps:
                    active = active & (halt_score < self.halt_threshold)
                    if not active.any():
                        break

        return outputs


# ---------------------------------------------------------------------------
# Convenience helpers (used by train script for param counting / summary)
# ---------------------------------------------------------------------------

def count_parameters_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_all_parameters_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
