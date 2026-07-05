"""
depth_sweep.py
==============
Depth experiment — vary number of reasoning blocks and K_inner.
Keep h=640, z=384 (run_005 optimal dims) fixed.

Configs:
  depth_1_k3  : 1 block, K_inner=3  (run_005 baseline)
  depth_2_k3  : 2 blocks, K_inner=3
  depth_3_k3  : 3 blocks, K_inner=3
  depth_1_k6  : 1 block, K_inner=6  (deeper inner loop)
  depth_2_k6  : 2 blocks, K_inner=6

Run: nohup python depth_sweep.py > depth_sweep_log.txt 2>&1 &
"""

import os, sys, time, json
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = '/efs/drsanny/visual_extension/storage/Shb_PROJECTS/SEG_REASONING-main'
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'models'))

# Patch
train_path = os.path.join(REPO_DIR, 'v7_train.py')
text = open(train_path).read()
if 'os.environ["CUDA_VISIBLE_DEVICES"] = "3"' in text:
    text = text.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "3"',
                        'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")')
if 'disable_cudnn:       bool = True' in text:
    text = text.replace('disable_cudnn:       bool = True',
                        'disable_cudnn:       bool = False')
open(train_path, 'w').write(text)
print('Repo patched.')

import v7_train as T

# ── Building blocks (copied from v7_model, self-contained) ───────────────────

def _make_gn(channels, num_groups=32):
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)

class ConvGNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            _make_gn(out_c),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = ConvGNAct(in_c, out_c)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            _make_gn(out_c),
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

# ── Deep HiReMed — stacks N reasoning blocks ─────────────────────────────────

class DeepHiReMed(nn.Module):
    """
    HiReMed with configurable number of reasoning blocks.
    Each block = LowLevelRefinement + HighLevelTransition.
    The decoder runs after ALL blocks at each recursive step.

    n_blocks=1, K_inner=3  →  equivalent to run_005
    n_blocks=2, K_inner=3  →  2x reasoning depth
    """
    def __init__(
        self,
        encoder_name               = 'convnext_tiny',
        encoder_pretrained         = True,
        latent_dim_h               = 640,
        latent_dim_z               = 384,
        n_blocks                   = 1,    # NEW: number of reasoning blocks
        max_steps                  = 6,
        min_steps                  = 3,
        K_inner                    = 3,
        halt_improvement_threshold = 0.1,
        dropout_p                  = 0.1,
        recursive_dropout_p        = 0.2,
    ):
        super().__init__()
        from v7_model import (
            ConvNeXTEncoder, LowLevelRefinement,
            HighLevelTransition, Decoder,
            UncertaintyHead, HaltHead,
        )

        # Encoder
        self.encoder = ConvNeXTEncoder(encoder_name, encoder_pretrained)
        ch = {"s1": 96, "s2": 192, "s3": 384, "s4": 768}  # ConvNeXt-tiny hardcoded

        self.z_proj = nn.Conv2d(ch['s4'], latent_dim_z, 1, bias=False)

        # Stack of n_blocks reasoning blocks (each has its own weights)
        self.low_blocks  = nn.ModuleList([
            LowLevelRefinement(latent_dim_z, latent_dim_h, ch["s2"], dropout_p=dropout_p) for _ in range(n_blocks)
        ])
        self.high_blocks = nn.ModuleList([
            HighLevelTransition(
                latent_dim_h, latent_dim_z, ch['s3'],
                dropout_p=dropout_p
            ) for _ in range(n_blocks)
        ])

        self.decoder   = Decoder(latent_dim_z, ch['s1'], ch['s2'], ch['s3'])
        self.unc_head  = UncertaintyHead(latent_dim_z, ch['s1'])
        self.halt_head = T.HaltHead(latent_dim_z, latent_dim_h) \
            if hasattr(T, 'HaltHead') else self._make_halt_head(latent_dim_z, latent_dim_h)

        self.latent_dim_h  = latent_dim_h
        self.latent_dim_z  = latent_dim_z
        self.n_blocks      = n_blocks
        self.max_steps     = max_steps
        self.min_steps     = min_steps
        self.K_inner       = K_inner
        self.halt_improvement_threshold = halt_improvement_threshold
        self.recursive_dropout_p = recursive_dropout_p

    def _make_halt_head(self, dz, dh):
        """Fallback halt head if not in v7_train."""
        from v7_model import ConvGNAct as CGA
        class HH(nn.Module):
            def __init__(self):
                super().__init__()
                self.z_pool = nn.Sequential(CGA(dz, dz//2, 3,1,1), nn.AdaptiveAvgPool2d(1))
                self.mlp = nn.Sequential(
                    nn.Linear(dz//2+dh+1, 128), nn.GELU(),
                    nn.Linear(128, 64), nn.GELU(),
                    nn.Linear(64, 1), nn.Sigmoid(),
                )
            def forward(self, z, u, h):
                zf = self.z_pool(z).squeeze(-1).squeeze(-1)
                us = u.mean(dim=(1,2,3))
                return self.mlp(torch.cat([zf, h, us.unsqueeze(1)], dim=1)).squeeze(1)
        return HH()

    def forward(self, x, force_steps=None):
        B, _, H, W = x.shape
        dtype = x.dtype
        dev   = x.device

        s1, s2, s3, s4 = self.encoder(x)
        z  = self.z_proj(s4)
        h  = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
        mu = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
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
                    torch.full((B,1,1,1), 1-self.recursive_dropout_p, device=dev)
                )
                z = z * mask

            # ── Run through all n_blocks ──────────────────────────────────
            for b in range(self.n_blocks):
                for _ in range(self.K_inner):
                    z = self.low_blocks[b](z, h, s2, y_prob_prev, u_prev)
                h, mu, logvar = self.high_blocks[b](h, z, s3)

            # ── Decode once after all blocks ──────────────────────────────
            delta      = self.decoder(z, s1, s2, s3, (H, W))
            y_logit    = y_logit + delta
            y_prob     = torch.sigmoid(y_logit.float()).to(dtype)
            u_map      = self.unc_head(z, s1, (H, W))
            halt_score = self.halt_head(z, u_map, h)

            y_prob_prev = y_prob.detach()
            u_prev      = u_map.detach()

            outputs.append({
                'y_logit':     y_logit,
                'y_prob':      y_prob.clamp(0, 1),
                'uncertainty': u_map.clamp(0, 1),
                'halt_score':  halt_score.clamp(0, 1),
                'mu':          mu,
                'logvar':      logvar,
            })

            if use_halting and step + 1 >= self.min_steps:
                active = active & (halt_score < (1.0 - self.halt_improvement_threshold))
                if not active.any():
                    break

        return outputs


# ── Count params ─────────────────────────────────────────────────────────────

def count_M(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ── Train one config ──────────────────────────────────────────────────────────

def train_one(name, n_blocks, K_inner, epochs=200):
    print(f'\n{"="*60}')
    print(f'Config {name}: n_blocks={n_blocks}, K_inner={K_inner}')
    print(f'{"="*60}')

    for m in list(sys.modules):
        if 'v7_train' in m or m.endswith('_model'):
            del sys.modules[m]
    import v7_train as T

    cfg = T.Config(
        dataset_name       = 'kvasir',
        use_test_as_val    = False,
        encoder_name       = 'convnext_tiny',
        encoder_pretrained = True,
        image_size         = 256,
        batch_size         = 16,
        num_workers        = 0,
        epochs             = epochs,
        lr_encoder         = 1e-4,
        lr_decoder         = 1e-3,
        disable_cudnn      = False,
        latent_dim_h       = 640,
        latent_dim_z       = 384,
        K_inner            = K_inner,
    )
    T.configure_runtime(cfg)
    T.set_seed(cfg.seed)

    model = DeepHiReMed(
        encoder_name               = cfg.encoder_name,
        encoder_pretrained         = cfg.encoder_pretrained,
        latent_dim_h               = cfg.latent_dim_h,
        latent_dim_z               = cfg.latent_dim_z,
        n_blocks                   = n_blocks,
        max_steps                  = cfg.max_steps,
        min_steps                  = cfg.min_steps,
        K_inner                    = K_inner,
        halt_improvement_threshold = cfg.halt_improvement_threshold,
        dropout_p                  = cfg.dropout_p,
        recursive_dropout_p        = cfg.recursive_dropout_p,
    ).to(cfg.device)

    total_M = count_M(model)
    enc_M   = count_M(model.encoder)
    rsn_M   = total_M - enc_M
    print(f'Params: {total_M:.2f}M  enc={enc_M:.2f}M ({100*enc_M/total_M:.0f}%)  '
          f'reason={rsn_M:.2f}M ({100*rsn_M/total_M:.0f}%)')
    print(f'  low_blocks : {count_M(model.low_blocks):.3f}M x {n_blocks}')
    print(f'  high_blocks: {count_M(model.high_blocks):.3f}M x {n_blocks}')

    run_dir = Path(REPO_DIR) / 'RESULTS' / f'depth_{name}' / 'run_001'
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = T.build_dataloaders(cfg)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": cfg.lr_encoder},
        {"params": model.z_proj.parameters(),     "lr": cfg.lr_decoder},
        {"params": model.low_blocks.parameters(), "lr": cfg.lr_decoder},
        {"params": model.high_blocks.parameters(),"lr": cfg.lr_decoder},
        {"params": model.decoder.parameters(),    "lr": cfg.lr_decoder},
        {"params": model.unc_head.parameters(),   "lr": cfg.lr_decoder},
        {"params": model.halt_head.parameters(),  "lr": cfg.lr_decoder},
    ], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    logger    = T.setup_logger(str(run_dir))

    best_dice = 0.0
    history   = []

    for epoch in range(1, epochs + 1):
        t0       = time.time()
        train_m  = T.run_train_epoch(model, train_loader, optimizer, cfg, epoch, logger)
        scheduler.step()
        val_both = T.eval_both_modes(model, val_loader, cfg,
                                      label=f'val ep{epoch}',
                                      logger=logger, current_epoch=epoch)
        val_dice  = val_both['adaptive']['dice']
        avg_steps = val_both['adaptive']['avg_steps']
        elapsed   = time.time() - t0

        history.append({
            'epoch': epoch,
            'loss': train_m['loss'],
            'dice_1pass': val_both['single_pass']['dice'],
            'dice_adap': val_dice,
            'avg_steps': avg_steps,
        })

        if epoch % 10 == 0 or epoch <= 5:
            print(f'  Ep {epoch:03d}: loss={train_m["loss"]:.4f}  '
                  f'dice_1p={val_both["single_pass"]["dice"]:.4f}  '
                  f'dice_ad={val_dice:.4f}  '
                  f'steps={avg_steps:.2f}  t={elapsed:.0f}s', flush=True)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), run_dir / 'best.pth')
            print(f'  -> New best: {best_dice:.4f}', flush=True)

    # Final test
    model.load_state_dict(torch.load(run_dir / 'best.pth', map_location=cfg.device))
    model.eval()
    final = T.eval_both_modes(model, test_loader, cfg, label='final test')

    result = {
        'name': name,
        'n_blocks': n_blocks,
        'K_inner': K_inner,
        'total_params_M': total_M,
        'enc_M': enc_M,
        'best_val_dice': best_dice,
        'test_dice_1pass':    final['single_pass']['dice'],
        'test_dice_adaptive': final['adaptive']['dice'],
        'test_iou':           final['adaptive']['iou'],
        'test_precision':     final['adaptive']['precision'],
        'test_recall':        final['adaptive']['recall'],
        'avg_steps':          final['adaptive']['avg_steps'],
    }

    with open(run_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f'DONE {name}: best_val={best_dice:.4f}  '
          f'test_dice={result["test_dice_adaptive"]:.4f}')
    return result


# ── Sweep configs ────────────────────────────────────────────────────────────
# (name, n_blocks, K_inner)
SWEEP = [
    ('1block_k6',  1, 6),   # deeper inner loop
    ('2block_k6',  2, 6),   # both deeper
]

# Print param counts before training
print('\nParam counts for all configs:')
print(f'{"Config":<15} {"n_blocks":>8} {"K_inner":>8} {"Params(M)":>12}')
print('-' * 48)
for name, nb, ki in SWEEP:
    cfg_tmp = T.Config(encoder_pretrained=False, latent_dim_h=640, latent_dim_z=384)
    m = DeepHiReMed(encoder_pretrained=False, latent_dim_h=640, latent_dim_z=384,
                    n_blocks=nb, K_inner=ki)
    print(f'{name:<15} {nb:>8} {ki:>8} {count_M(m):>12.2f}')
print()

# Run sweep
all_results = []
for name, nb, ki in SWEEP:
    result = train_one(name, nb, ki, epochs=200)
    all_results.append(result)
    torch.cuda.empty_cache()

# Final summary table
print()
print('=' * 75)
print('DEPTH SWEEP COMPLETE')
print('=' * 75)
print(f'{"Config":<15} {"blocks":>7} {"K":>4} {"Params":>9} '
      f'{"Dice_1p":>9} {"Dice_ad":>9} {"IoU":>8}')
print('-' * 75)
for r in all_results:
    print(f'{r["name"]:<15} {r["n_blocks"]:>7} {r["K_inner"]:>4} '
          f'{r["total_params_M"]:>8.2f}M '
          f'{r["test_dice_1pass"]:>9.4f} '
          f'{r["test_dice_adaptive"]:>9.4f} '
          f'{r["test_iou"]:>8.4f}')
print('-' * 75)
best = max(all_results, key=lambda x: x['test_dice_adaptive'])
print(f'Best: {best["name"]}  blocks={best["n_blocks"]}  K={best["K_inner"]}  '
      f'Dice={best["test_dice_adaptive"]:.4f}')
print(f'Baseline (run_005): Dice=0.9236')
print(f'Delta vs baseline: {best["test_dice_adaptive"]-0.9236:+.4f}')

with open(Path(REPO_DIR) / 'RESULTS' / 'depth_sweep_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('Results saved to RESULTS/depth_sweep_summary.json')
EOF