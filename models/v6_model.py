"""
HiReMed V6 — Closed-loop recursive refinement with pretrained encoder support.

Supported encoders (via encoder_name):
  ResNet family   : resnet18, resnet34, resnet50, resnet101, resnet152
  ConvNeXT family : convnext_tiny, convnext_small, convnext_base, convnext_large

All encoders return 4 multi-scale feature maps (s1, s2, s3, s4) at spatial
resolutions H/4, H/8, H/16, H/32 respectively, measured from the input size.

ImageNet normalisation (mean/std) is applied *inside* each encoder so the
caller only needs to supply images in [0, 1].  This keeps preprocessing
encoder-specific and ensures the correct statistics are always used regardless
of how the dataloader is configured.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------

# Output channel counts for (s1, s2, s3, s4).
# These are the channel counts AT THE OUTPUT of each stage block, not at the
# downsample layer between stages.  Verified against torchvision source.
ENCODER_CHANNELS: Dict[str, Tuple[int, int, int, int]] = {
    # ResNet-basicblock variants  (layer1..4 output channels)
    "resnet18":  (64,   128,  256,  512),
    "resnet34":  (64,   128,  256,  512),
    # ResNet-bottleneck variants  (layer1..4 output channels)
    "resnet50":  (256,  512,  1024, 2048),
    "resnet101": (256,  512,  1024, 2048),
    "resnet152": (256,  512,  1024, 2048),
    # ConvNeXt (features[2,4,6,8] output channels — stage blocks, not downsamples)
    "convnext_tiny":  (96,   192,  384,  768),
    "convnext_small": (96,   192,  384,  768),
    "convnext_base":  (128,  256,  512,  1024),
    "convnext_large": (192,  384,  768,  1536),
}

# Approximate backbone-only parameter counts (M) — for logging only.
# These are the encoder parameters; total model params will be higher.
ENCODER_PARAMS_M: Dict[str, float] = {
    "resnet18":       11.7,
    "resnet34":       21.8,
    "resnet50":       25.6,
    "resnet101":      44.5,
    "resnet152":      60.2,
    "convnext_tiny":  28.6,
    "convnext_small": 50.2,
    "convnext_base":  88.6,
    "convnext_large": 197.8,
}

# Torchvision DEFAULT weight enums — ImageNet-1K pretrained.
ENCODER_WEIGHTS_MAP: Dict[str, object] = {
    "resnet18":       tv_models.ResNet18_Weights.DEFAULT,
    "resnet34":       tv_models.ResNet34_Weights.DEFAULT,
    "resnet50":       tv_models.ResNet50_Weights.DEFAULT,
    "resnet101":      tv_models.ResNet101_Weights.DEFAULT,
    "resnet152":      tv_models.ResNet152_Weights.DEFAULT,
    "convnext_tiny":  tv_models.ConvNeXt_Tiny_Weights.DEFAULT,
    "convnext_small": tv_models.ConvNeXt_Small_Weights.DEFAULT,
    "convnext_base":  tv_models.ConvNeXt_Base_Weights.DEFAULT,
    "convnext_large": tv_models.ConvNeXt_Large_Weights.DEFAULT,
}

# ImageNet normalisation constants — same for all torchvision pretrained models.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_encoder_channels(name: str) -> Tuple[int, int, int, int]:
    if name not in ENCODER_CHANNELS:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Choose from: {sorted(ENCODER_CHANNELS.keys())}"
        )
    return ENCODER_CHANNELS[name]


# ---------------------------------------------------------------------------
# Pretrained encoders
# ---------------------------------------------------------------------------

class _NormalizeInput(nn.Module):
    """
    Applies per-channel normalisation in-place:
        out = (x - mean) / std
    where x is expected to be in [0, 1].

    Stored as buffers (not parameters) so they are saved in state_dict,
    moved with .to(device/dtype), but not updated by the optimiser.
    """
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        # shape [1, C, 1, 1] for broadcasting over B, H, W
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor(std).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean.to(x.dtype)) / self.std.to(x.dtype)


class ResNetEncoder(nn.Module):
    """
    Torchvision ResNet as a 4-stage feature extractor.

    Input:  RGB image in [0, 1],  any spatial size ≥ 32×32.
    Output: (s1, s2, s3, s4) at H/4, H/8, H/16, H/32.

    Forward path:
        norm → conv1 → bn1 → relu → maxpool  (→ H/4)
        → layer1 → s1  (H/4)
        → layer2 → s2  (H/8)
        → layer3 → s3  (H/16)
        → layer4 → s4  (H/32)

    The avgpool and fc layers are replaced with nn.Identity() so that
    state_dict() and parameters() remain well-behaved.
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.norm = _NormalizeInput(_IMAGENET_MEAN, _IMAGENET_STD)

        weights = ENCODER_WEIGHTS_MAP[name] if pretrained else None
        bb = getattr(tv_models, name)(weights=weights)

        # Replace classifier layers with Identity so they don't consume
        # parameters or appear incorrectly in state_dict.
        bb.avgpool = nn.Identity()
        bb.fc      = nn.Identity()
        self.backbone = bb

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x  = self.norm(x)
        x  = self.backbone.conv1(x)
        x  = self.backbone.bn1(x)
        x  = self.backbone.relu(x)
        x  = self.backbone.maxpool(x)   # H/4
        s1 = self.backbone.layer1(x)    # H/4
        s2 = self.backbone.layer2(s1)   # H/8
        s3 = self.backbone.layer3(s2)   # H/16
        s4 = self.backbone.layer4(s3)   # H/32
        return s1, s2, s3, s4


class ConvNeXTEncoder(nn.Module):
    """
    Torchvision ConvNeXt as a 4-stage feature extractor.

    Input:  RGB image in [0, 1],  any spatial size ≥ 32×32.
    Output: (s1, s2, s3, s4) at H/4, H/8, H/16, H/32.

    Actual torchvision ConvNeXt ``features`` layout (8 entries, indices 0–7).
    The stem at [0] is itself a Sequential([Conv2d, LayerNorm]) — there is NO
    separate LayerNorm entry at [1]:

        [0] Sequential(Conv2d stem stride=4, LayerNorm) → H/4
        [1] Sequential of CNBlocks  (stage 1)           → H/4   ← s1
        [2] Sequential [LayerNorm, Conv2d stride=2]     → H/8   (downsample)
        [3] Sequential of CNBlocks  (stage 2)           → H/8   ← s2
        [4] Sequential [LayerNorm, Conv2d stride=2]     → H/16  (downsample)
        [5] Sequential of CNBlocks  (stage 3)           → H/16  ← s3
        [6] Sequential [LayerNorm, Conv2d stride=2]     → H/32  (downsample)
        [7] Sequential of CNBlocks  (stage 4)           → H/32  ← s4

    The classifier is replaced with nn.Identity().
    """

    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        self.norm = _NormalizeInput(_IMAGENET_MEAN, _IMAGENET_STD)

        weights = ENCODER_WEIGHTS_MAP[name] if pretrained else None
        bb = getattr(tv_models, name)(weights=weights)
        bb.classifier = nn.Identity()
        self.backbone = bb

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x  = self.norm(x)
        f  = self.backbone.features

        x  = f[0](x)    # stem Sequential(Conv2d, LayerNorm) → H/4
        s1 = f[1](x)    # stage 1 blocks  → H/4
        x  = f[2](s1)   # downsample      → H/8
        s2 = f[3](x)    # stage 2 blocks  → H/8
        x  = f[4](s2)   # downsample      → H/16
        s3 = f[5](x)    # stage 3 blocks  → H/16
        x  = f[6](s3)   # downsample      → H/32
        s4 = f[7](x)    # stage 4 blocks  → H/32
        return s1, s2, s3, s4


def build_encoder(name: str, pretrained: bool = True) -> nn.Module:
    """Factory — returns ResNetEncoder or ConvNeXTEncoder."""
    n = name.lower()
    if n.startswith("resnet"):
        return ResNetEncoder(name, pretrained)
    elif n.startswith("convnext"):
        return ConvNeXTEncoder(name, pretrained)
    else:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Choose from: {sorted(ENCODER_CHANNELS.keys())}"
        )


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _safe_groups(channels: int, desired: int = 8) -> int:
    """Largest divisor of `channels` that is ≤ `desired`."""
    g = min(desired, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class ConvGNAct(nn.Module):
    """Conv2d → GroupNorm → GELU."""
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
    Updates z using current h, s2 skip features, and the previous step's y_prob.

    Feeding y_prob_prev closes the refinement loop: the module explicitly sees
    *what was predicted* so far and can compute a correction, rather than
    operating open-loop from z alone.  On step 0, y_prob_prev is zeros.

    Inputs:
        z          [B, dz, Hz, Wz]      spatial latent (at H/32)
        h          [B, dh]              global latent
        s2         [B, s2_ch, H/8, W/8] encoder stage-2 skip
        y_prob_prev [B, 1, H, W]        previous step's sigmoid probability map
    Output: z  [B, dz, Hz, Wz]
    """
    def __init__(self, dz: int, dh: int, s2_ch: int, y_prob_ch: int = 1):
        super().__init__()
        self.conv1 = ConvGNAct(dz + dh + s2_ch + y_prob_ch, dz, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dz, dz, 3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(dz), dz),
        )
        self.act = nn.GELU()

    def forward(
        self,
        z:          torch.Tensor,
        h:          torch.Tensor,
        s2:         torch.Tensor,
        y_prob_prev: torch.Tensor,
    ) -> torch.Tensor:
        hz, wz = z.shape[-2], z.shape[-1]
        h_map  = h.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hz, wz)
        s2_res = F.interpolate(s2, size=(hz, wz), mode="bilinear", align_corners=False)
        yp_res = F.interpolate(y_prob_prev, size=(hz, wz), mode="bilinear", align_corners=False)
        cat    = torch.cat([z, h_map, s2_res, yp_res], dim=1)
        return self.act(self.conv2(self.conv1(cat)) + z)


# ---------------------------------------------------------------------------
# High-level global transition  (once per outer step)
# ---------------------------------------------------------------------------

class HighLevelTransition(nn.Module):
    """
    Updates h using spatially-pooled z and s3 context.

    Returns (h_new, mu, logvar) — mu/logvar are used for KL regularisation.

    Inputs:
        h  [B, dh]
        z  [B, dz, Hz, Wz]
        s3 [B, s3_ch, H/16, W/16]
    Output:
        h_new  [B, dh]
        mu     [B, dh]
        logvar [B, dh]
    """
    def __init__(self, dh: int, dz: int, s3_ch: int):
        super().__init__()
        self.fc1         = nn.Linear(dh + dz + s3_ch, dh)
        self.fc2         = nn.Linear(dh, dh)
        self.mu_head     = nn.Linear(dh, dh)
        self.logvar_head = nn.Linear(dh, dh)

    def forward(
        self, h: torch.Tensor, z: torch.Tensor, s3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """Upsample x to match skip spatial size, concatenate, pass through ResBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = ResBlock(in_ch + skip_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class Decoder(nn.Module):
    """
    Reconstructs a full-resolution residual logit delta.

    Path: z (H/32) → fuse(s3, H/16) → fuse(s2, H/8) → fuse(s1, H/4)
          → refine → 1×1 conv → bilinear upsample to (H, W)

    Output: Δ logit  [B, 1, H, W]
    """
    def __init__(self, z_ch: int, s1_ch: int, s2_ch: int, s3_ch: int):
        super().__init__()
        self.fuse3  = UpFuse(z_ch,  s3_ch, s3_ch)
        self.fuse2  = UpFuse(s3_ch, s2_ch, s2_ch)
        self.fuse1  = UpFuse(s2_ch, s1_ch, s1_ch)
        self.refine = ResBlock(s1_ch, s1_ch, stride=1)
        self.out    = nn.Conv2d(s1_ch, 1, 1)

    def forward(
        self,
        z:       torch.Tensor,
        s1:      torch.Tensor,
        s2:      torch.Tensor,
        s3:      torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        x = self.fuse3(z,  s3)
        x = self.fuse2(x,  s2)
        x = self.fuse1(x,  s1)
        x = self.out(self.refine(x))
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Uncertainty head
# ---------------------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """
    Predicts a spatial uncertainty map in [0, 1].
    Supervised to match |y_prob - y_true| (per-pixel absolute error).

    Inputs:  z [B, dz, H/32, W/32],  s1 [B, s1_ch, H/4, W/4]
    Output:  u_map [B, 1, H, W]
    """
    def __init__(self, z_ch: int, s1_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(z_ch + s1_ch, s1_ch, 3, 1, 1),
            nn.Conv2d(s1_ch, 1, 1),
        )

    def forward(
        self, z: torch.Tensor, s1: torch.Tensor, out_size: Tuple[int, int]
    ) -> torch.Tensor:
        z_up = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x    = self.block(torch.cat([z_up, s1], dim=1))
        return torch.sigmoid(
            F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        )


# ---------------------------------------------------------------------------
# Halt head — per-sample scalar in [0, 1]
# ---------------------------------------------------------------------------

class HaltHead(nn.Module):
    """
    Produces a per-sample halt score in [0, 1].

    High score → model believes current prediction is good enough to stop.
    Supervised to regress the current step's Dice score (see halt_loss in train.py).

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

    def forward(
        self, z: torch.Tensor, u: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        z_feat   = self.z_pool(z).squeeze(-1).squeeze(-1)   # [B, dz//2]
        u_scalar = u.mean(dim=(1, 2, 3))                     # [B]
        feat     = torch.cat([z_feat, h, u_scalar.unsqueeze(1)], dim=1)
        return self.mlp(feat).squeeze(1)                     # [B]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HiReMed(nn.Module):
    """
    Hierarchical Recursive Medical Segmentation model.

    Constructor args:
        encoder_name        one of ENCODER_CHANNELS keys
        encoder_pretrained  load ImageNet-1K weights (default True)
        latent_dim_h        global latent dimension
        latent_dim_z        spatial latent dimension
        max_steps           upper bound on refinement steps
        min_steps           minimum steps before halting is allowed
        K_inner             spatial refinement iterations per outer step
        halt_improvement_threshold  stop when predicted step improvement < this

    forward(x, force_steps) returns a list of per-step dicts, each containing:
        y_logit     [B, 1, H, W]   accumulated residual logits
        y_prob      [B, 1, H, W]   sigmoid(y_logit)
        uncertainty [B, 1, H, W]   uncertainty map in [0, 1]
        halt_score  [B]            halt signal in [0, 1]
        mu          [B, dh]        latent mean  (used for KL loss)
        logvar      [B, dh]        latent log-variance  (used for KL loss)
        step        int            0-indexed step number

    force_steps=None:
        training  → always runs max_steps (full deep supervision)
        inference → adaptive halting per-sample

    force_steps=N:
        runs exactly N steps regardless of training/eval state (e.g. N=1 for
        single-pass evaluation comparison)
    """

    def __init__(
        self,
        encoder_name:                str   = "convnext_tiny",
        encoder_pretrained:          bool  = True,
        latent_dim_h:                int   = 512,
        latent_dim_z:                int   = 256,
        max_steps:                   int   = 6,
        min_steps:                   int   = 2,
        K_inner:                     int   = 3,
        halt_improvement_threshold:  float = 0.005,
    ):
        super().__init__()

        self.latent_dim_h              = latent_dim_h
        self.latent_dim_z              = latent_dim_z
        self.max_steps                 = max_steps
        self.min_steps                 = min_steps
        self.K_inner                   = K_inner
        self.halt_improvement_threshold = halt_improvement_threshold
        self.encoder_name       = encoder_name
        self.encoder_pretrained = encoder_pretrained

        dh = latent_dim_h
        dz = latent_dim_z
        s1_ch, s2_ch, s3_ch, s4_ch = get_encoder_channels(encoder_name)

        # Store for introspection / logging
        self.s1_ch = s1_ch
        self.s2_ch = s2_ch
        self.s3_ch = s3_ch
        self.s4_ch = s4_ch

        # ── Encoder (pretrained, includes input normalisation) ──────────────
        self.encoder = build_encoder(encoder_name, encoder_pretrained)

        # ── Latent initialisation ───────────────────────────────────────────
        self.z_proj = nn.Conv2d(s4_ch, dz, 1)

        # ── Recursive refinement ────────────────────────────────────────────
        self.low  = LowLevelRefinement(dz, dh, s2_ch)
        self.high = HighLevelTransition(dh, dz, s3_ch)

        # ── Decoder + output heads ──────────────────────────────────────────
        self.decoder   = Decoder(dz, s1_ch, s2_ch, s3_ch)
        self.unc_head  = UncertaintyHead(dz, s1_ch)
        self.halt_head = HaltHead(dz, dh)

    def encoder_parameters(self) -> List[nn.Parameter]:
        """Returns only the encoder parameters (for differential LR in optimiser)."""
        return list(self.encoder.parameters())

    def decoder_parameters(self) -> List[nn.Parameter]:
        """
        Returns all non-encoder parameters — z_proj, low, high, decoder,
        unc_head, halt_head.  These are randomly initialised and should use a
        higher learning rate than the pretrained encoder.
        """
        enc_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc_ids]

    def forward(
        self,
        x:           torch.Tensor,
        force_steps: Optional[int] = None,
    ) -> List[Dict]:
        B, _, H, W = x.shape
        dev   = x.device
        dtype = x.dtype

        # Encoder runs once; skip features are reused every refinement step
        s1, s2, s3, s4 = self.encoder(x)

        # Initialise mutable states
        z          = self.z_proj(s4)
        h          = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
        y_logit    = torch.zeros(B, 1, H, W,           device=dev, dtype=dtype)
        y_prob_prev = torch.zeros(B, 1, H, W,           device=dev, dtype=dtype)

        outputs: List[Dict] = []
        # Per-sample active flag — only used during adaptive inference
        active = torch.ones(B, dtype=torch.bool, device=dev)

        if force_steps is not None:
            max_s       = force_steps
            use_halting = False
        else:
            max_s       = self.max_steps
            use_halting = not self.training   # adaptive only at inference

        for step in range(max_s):

            # 1. Inner spatial refinement  (K_inner iterations, closed-loop)
            for _ in range(self.K_inner):
                z = self.low(z, h, s2, y_prob_prev)

            # 2. Global state update  (once per outer step)
            h, mu, logvar = self.high(h, z, s3)

            # 3. Residual logit accumulation
            delta   = self.decoder(z, s1, s2, s3, (H, W))
            y_logit = y_logit + delta
            y_prob  = torch.sigmoid(y_logit.float()).to(dtype)

            # Feed current prediction back into next step's low-level refinement
            y_prob_prev = y_prob.detach()

            # 4. Uncertainty map
            u_map = self.unc_head(z, s1, (H, W))

            # 5. Halt score  [B] in [0, 1]
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

            # 6. Adaptive early exit (inference only, not when force_steps set)
            # halt_score now regresses predicted improvement (0 = still improving,
            # 1 = plateau).  Stop when predicted improvement < threshold.
            if use_halting and step + 1 >= self.min_steps:
                active = active & (halt_score > (1.0 - self.halt_improvement_threshold))
                if not active.any():
                    break

        return outputs


# ---------------------------------------------------------------------------
# Parameter counting helpers
# ---------------------------------------------------------------------------

def count_parameters_m(model: nn.Module) -> float:
    """Trainable parameters only, in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_all_parameters_m(model: nn.Module) -> float:
    """All parameters (trainable + frozen), in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6