"""
v8_daflow_model.py
==================
HiReMed v8 with DAFlow-inspired LowLevelRefinement.

Key change: replace simple concat+conv in LowLevelRefinement with
Deformable Attention Flow (DAFlow) from:
  "Single Stage Virtual Try-on via Deformable Attention Flows" (ECCV 2022)

DAFlow idea adapted for segmentation:
  - Instead of one spatial update, predict K flow fields + K attention weights
  - Each flow field samples z from K different offset locations
  - Weighted sum via softmax gives richer spatial context
  - Uncertainty map u_prev guides which regions need more diverse sampling

Architecture change:
  LowLevelRefinement (concat+conv) → DAFlowRefinement (K-flow deformable attention)

Everything else identical to v7-balanced (run_005).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── Safe GroupNorm ────────────────────────────────────────────────────────────

def _safe_groups(channels: int, num_groups: int = 32) -> int:
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return g


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvGNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.GroupNorm(_safe_groups(out_c), out_c),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = ConvGNAct(in_c, out_c)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.GroupNorm(_safe_groups(out_c), out_c),
        )
        self.skip = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class UpFuse(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.block = ResBlock(in_c + skip_c, out_c)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


# ── DAFlow Refinement (core new module) ───────────────────────────────────────

class DAFlowRefinement(nn.Module):
    """
    Deformable Attention Flow-based Low-Level Refinement.

    Replaces simple concat+conv with K-flow deformable attention:
      1. Encode context from z, s2, y_prob_prev, u_prev
      2. Predict K offset fields (2K channels) + K attention weights (K channels)
      3. Sample z at K offset locations using bilinear interpolation
      4. Weighted sum via softmax attention → refined z

    This gives the model K different "views" of the spatial context
    and learns to combine them based on content and uncertainty.

    K=6 following DAFlow paper (best performance in ablation).
    """
    def __init__(self, dz: int, s2_ch: int, dh: int, K: int = 6, dropout_p: float = 0.1):
        super().__init__()
        self.K  = K
        self.dz = dz

        # Context encoder: takes z + s2 + y_prob_prev + u_prev
        # Aligns everything to z spatial resolution
        in_c = dz + s2_ch + 1 + 1  # z + s2 (downsampled) + y_prob + u_prev
        mid  = max(dz // 2, 64)

        self.context_enc = nn.Sequential(
            ConvGNAct(in_c, mid, 3, 1, 1),
            ConvGNAct(mid,  mid, 3, 1, 1),
        )

        # Flow field predictor: outputs 2K offset channels
        self.flow_head = nn.Conv2d(mid, 2 * K, kernel_size=3, padding=1, bias=True)

        # Attention weight predictor: outputs K channels
        self.attn_head = nn.Conv2d(mid, K, kernel_size=3, padding=1, bias=True)

        # Optional: refine after deformable sampling
        self.refine = nn.Sequential(
            ConvGNAct(dz, dz, 3, 1, 1),
            nn.Dropout2d(dropout_p),
        )

        # Initialize flow heads to small values for stable start
        nn.init.constant_(self.flow_head.weight, 0)
        nn.init.constant_(self.flow_head.bias, 0)
        nn.init.constant_(self.attn_head.bias, 0)

    def forward(
        self,
        z:           torch.Tensor,  # [B, dz, Hz, Wz]
        h:           torch.Tensor,  # [B, dh] — not used (global)
        s2:          torch.Tensor,  # [B, s2_ch, H/8, W/8]
        y_prob_prev: torch.Tensor,  # [B, 1, H, W]
        u_prev:      torch.Tensor,  # [B, 1, H, W]
    ) -> torch.Tensor:

        B, C, Hz, Wz = z.shape

        # Align all inputs to z spatial resolution (Hz, Wz = H/32)
        s2_dn  = F.interpolate(s2,          size=(Hz, Wz), mode='bilinear', align_corners=False)
        yp_dn  = F.interpolate(y_prob_prev, size=(Hz, Wz), mode='bilinear', align_corners=False)
        u_dn   = F.interpolate(u_prev,      size=(Hz, Wz), mode='bilinear', align_corners=False)

        # Encode context
        ctx  = torch.cat([z, s2_dn, yp_dn, u_dn], dim=1)   # [B, in_c, Hz, Wz]
        feat = self.context_enc(ctx)                          # [B, mid, Hz, Wz]

        # Predict K flow fields and K attention weights
        flows = self.flow_head(feat)   # [B, 2K, Hz, Wz]
        attn  = self.attn_head(feat)   # [B, K,  Hz, Wz]
        attn  = F.softmax(attn, dim=1) # normalize across K

        # Build sampling grid for each of K flows
        # Base grid: normalized coords in [-1, 1]
        base_grid = self._make_base_grid(B, Hz, Wz, z.device)  # [B, Hz, Wz, 2]

        # Sample z at K offset locations and combine with attention
        out = torch.zeros_like(z)  # [B, dz, Hz, Wz]

        for k in range(self.K):
            # Extract k-th flow field [B, 2, Hz, Wz] → offsets in [-1,1] scale
            offset = flows[:, 2*k:2*k+2, :, :]              # [B, 2, Hz, Wz]
            offset = torch.tanh(offset) * 0.5               # clamp to [-0.5, 0.5]
            offset = offset.permute(0, 2, 3, 1)             # [B, Hz, Wz, 2]

            # Sample grid = base + offset
            grid_k = base_grid + offset                      # [B, Hz, Wz, 2]
            grid_k = grid_k.clamp(-1, 1)

            # Sample z at this offset grid
            z_k = F.grid_sample(z, grid_k, mode='bilinear',
                                 padding_mode='border', align_corners=True)  # [B, dz, Hz, Wz]

            # Weight by k-th attention map
            attn_k = attn[:, k:k+1, :, :]                   # [B, 1, Hz, Wz]
            out    = out + attn_k * z_k

        # Residual connection + refinement
        out = self.refine(out) + z
        return out

    def _make_base_grid(self, B, H, W, device):
        """Create normalized base sampling grid [-1, 1]."""
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)        # [H, W, 2]
        return grid.unsqueeze(0).expand(B, -1, -1, -1)      # [B, H, W, 2]


# ── Rest of architecture (identical to v7) ────────────────────────────────────

class HighLevelTransition(nn.Module):
    def __init__(self, dh: int, dz: int, s3_ch: int, dropout_p: float = 0.1):
        super().__init__()
        in_dim = dh + dz + s3_ch
        self.fc1     = nn.Linear(in_dim, dh)
        self.drop    = nn.Dropout(dropout_p)
        self.fc2     = nn.Linear(dh, dh)
        self.mu_head = nn.Linear(dh, dh)
        self.lv_head = nn.Linear(dh, dh)

    def forward(self, h, z, s3):
        z_pool  = z.mean(dim=(2, 3))
        s3_pool = s3.mean(dim=(2, 3))
        feat    = torch.cat([h, z_pool, s3_pool], dim=1)
        feat    = F.gelu(self.drop(self.fc1(feat)))
        feat    = F.gelu(self.fc2(feat))
        mu      = self.mu_head(feat)
        logvar  = self.lv_head(feat)
        h_new   = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return h_new, mu, logvar


class Decoder(nn.Module):
    def __init__(self, dz, s1_ch, s2_ch, s3_ch):
        super().__init__()
        out3 = max(s3_ch, dz // 2)
        out2 = max(s2_ch, dz // 4)
        out1 = max(s1_ch, 32)
        self.fuse3  = UpFuse(dz, s3_ch, out3)
        self.fuse2  = UpFuse(out3, s2_ch, out2)
        self.fuse1  = UpFuse(out2, s1_ch, out1)
        self.refine = ResBlock(out1, out1)
        self.out    = nn.Conv2d(out1, 1, 1)

    def forward(self, z, s1, s2, s3, target_size):
        x = self.fuse3(z, s3)
        x = self.fuse2(x, s2)
        x = self.fuse1(x, s1)
        x = self.refine(x)
        x = self.out(x)
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


class UncertaintyHead(nn.Module):
    def __init__(self, dz, s1_ch):
        super().__init__()
        in_c = dz + s1_ch
        self.block = nn.Sequential(
            ConvGNAct(in_c, s1_ch, 3, 1, 1),
            nn.Conv2d(s1_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, s1, target_size):
        z_up = F.interpolate(z, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        feat = torch.cat([z_up, s1], dim=1)
        out  = self.block(feat)
        return F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)


class HaltHead(nn.Module):
    def __init__(self, dz, dh):
        super().__init__()
        self.z_pool = nn.Sequential(
            ConvGNAct(dz, dz // 2, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        in_dim = dz // 2 + dh + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 64),    nn.GELU(),
            nn.Linear(64, 1),      nn.Sigmoid(),
        )

    def forward(self, z, u, h):
        z_feat   = self.z_pool(z).squeeze(-1).squeeze(-1)
        u_scalar = u.mean(dim=(1, 2, 3))
        feat     = torch.cat([z_feat, h, u_scalar.unsqueeze(1)], dim=1)
        return self.mlp(feat).squeeze(1)


# ── Full DAFlow HiReMed ───────────────────────────────────────────────────────

class DAFlowHiReMed(nn.Module):
    """
    HiReMed v8 with DAFlow-based LowLevelRefinement.

    The only change from v7-balanced (run_005):
      LowLevelRefinement → DAFlowRefinement (K=6 deformable attention flows)

    Everything else identical: encoder, high, decoder, unc_head, halt_head.
    """
    def __init__(
        self,
        encoder_name               = 'convnext_tiny',
        encoder_pretrained         = True,
        latent_dim_h               = 640,
        latent_dim_z               = 384,
        K_daflow                   = 6,    # number of flow fields
        max_steps                  = 6,
        min_steps                  = 3,
        K_inner                    = 3,    # inner refinement iterations
        halt_improvement_threshold = 0.1,
        dropout_p                  = 0.1,
        recursive_dropout_p        = 0.2,
    ):
        super().__init__()

        # Import ConvNeXTEncoder from v7_model (same as run_005)
        import sys, os
        from v7_model import ConvNeXTEncoder
        ch = {'s1': 96, 's2': 192, 's3': 384, 's4': 768}

        self.encoder  = ConvNeXTEncoder(encoder_name, encoder_pretrained)
        self.z_proj   = nn.Conv2d(ch['s4'], latent_dim_z, 1, bias=False)

        # ← KEY CHANGE: DAFlowRefinement instead of LowLevelRefinement
        self.low = DAFlowRefinement(
            dz        = latent_dim_z,
            s2_ch     = ch['s2'],
            dh        = latent_dim_h,
            K         = K_daflow,
            dropout_p = dropout_p,
        )

        self.high      = HighLevelTransition(latent_dim_h, latent_dim_z, ch['s3'], dropout_p)
        self.decoder   = Decoder(latent_dim_z, ch['s1'], ch['s2'], ch['s3'])
        self.unc_head  = UncertaintyHead(latent_dim_z, ch['s1'])
        self.halt_head = HaltHead(latent_dim_z, latent_dim_h)

        self.latent_dim_h = latent_dim_h
        self.latent_dim_z = latent_dim_z
        self.K_inner      = K_inner
        self.max_steps    = max_steps
        self.min_steps    = min_steps
        self.halt_improvement_threshold = halt_improvement_threshold
        self.recursive_dropout_p = recursive_dropout_p

    def forward(self, x, force_steps=None):
        B, _, H, W = x.shape
        dtype = x.dtype
        dev   = x.device

        s1, s2, s3, s4 = self.encoder(x)
        z      = self.z_proj(s4)
        h      = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
        mu     = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
        logvar = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)

        y_logit     = torch.zeros(B, 1, H, W, device=dev, dtype=dtype)
        y_prob_prev = torch.zeros(B, 1, H, W, device=dev, dtype=dtype)
        u_prev      = torch.zeros(B, 1, H, W, device=dev, dtype=dtype)

        max_s       = force_steps if force_steps is not None else self.max_steps
        use_halting = force_steps is None
        active      = torch.ones(B, dtype=torch.bool, device=dev)
        outputs     = []

        for step in range(max_s):
            if self.training and self.recursive_dropout_p > 0 and step > 0:
                mask = torch.bernoulli(
                    torch.full((B,1,1,1), 1-self.recursive_dropout_p, device=dev))
                z = z * mask

            # K_inner iterations of DAFlow refinement
            for _ in range(self.K_inner):
                z = self.low(z, h, s2, y_prob_prev, u_prev)

            h, mu, logvar = self.high(h, z, s3)
            delta   = self.decoder(z, s1, s2, s3, (H, W))
            y_logit = y_logit + delta
            y_prob  = torch.sigmoid(y_logit.float()).to(dtype)
            u_map   = self.unc_head(z, s1, (H, W)).clamp(0, 1)
            halt_score = self.halt_head(z, u_map, h).clamp(0, 1)

            y_prob_prev = y_prob.detach()
            u_prev      = u_map.detach()

            outputs.append({
                'y_logit':     y_logit,
                'y_prob':      y_prob.clamp(0, 1),
                'uncertainty': u_map,
                'halt_score':  halt_score,
                'mu':          mu,
                'logvar':      logvar,
            })

            if use_halting and step + 1 >= self.min_steps:
                active = active & (halt_score < (1.0 - self.halt_improvement_threshold))
                if not active.any():
                    break

        return outputs


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    print('Testing DAFlowHiReMed...')
    model = DAFlowHiReMed(
        encoder_pretrained = False,
        latent_dim_h       = 640,
        latent_dim_z       = 384,
        K_daflow           = 6,
    )

    total_M = sum(p.numel() for p in model.parameters()) / 1e6
    enc_M   = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    low_M   = sum(p.numel() for p in model.low.parameters()) / 1e6
    print(f'Total  : {total_M:.2f}M')
    print(f'Encoder: {enc_M:.2f}M  ({100*enc_M/total_M:.1f}%)')
    print(f'DAFlow : {low_M:.2f}M  (was 5.538M in run_005)')
    print()

    x   = torch.rand(2, 3, 256, 256)
    out = model(x, force_steps=3)
    print(f'Forward OK: {len(out)} steps')
    print(f'y_logit: {tuple(out[-1]["y_logit"].shape)}')
    print(f'uncertainty: {tuple(out[-1]["uncertainty"].shape)}')
    print()

    # Compare module sizes
    run005_low_M = 5.538
    print(f'LowLevelRefinement (run_005): {run005_low_M:.3f}M')
    print(f'DAFlowRefinement   (K=6)    : {low_M:.3f}M')
    print(f'Delta              : +{low_M-run005_low_M:.3f}M')
