"""
train_daflow.py
===============
Train HiReMed v8 with DAFlow-based LowLevelRefinement on Kvasir-SEG.

Key change from run_005:
  LowLevelRefinement (concat+conv, 5.538M)
  → DAFlowRefinement (K=6 deformable attention flows, 2.690M)

Hypothesis: deformable attention sampling gives richer spatial context
than simple concatenation, helping the model focus refinement on
uncertain/boundary regions.

Baseline: run_005 Dice 0.9236

Run: nohup python train_daflow.py > daflow_log.txt 2>&1 &
"""

import os, sys, time, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = '/efs/drsanny/visual_extension/storage/Shb_PROJECTS/SEG_REASONING-main'
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'models'))
sys.path.insert(0, '/efs/drsanny/visual_extension/storage/Shb_PROJECTS')

# Patch v7_train.py
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

# ── DAFlow modules (self-contained) ──────────────────────────────────────────

def _safe_groups(channels, num_groups=32):
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return g

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

class DAFlowRefinement(nn.Module):
    """
    K=6 deformable attention flow refinement.
    Replaces simple concat+conv in LowLevelRefinement.
    """
    def __init__(self, dz, s2_ch, dh, K=6, dropout_p=0.1):
        super().__init__()
        self.K  = K
        self.dz = dz
        in_c = dz + s2_ch + 1 + 1
        mid  = max(dz // 2, 64)
        self.context_enc = nn.Sequential(
            ConvGNAct(in_c, mid, 3, 1, 1),
            ConvGNAct(mid,  mid, 3, 1, 1),
        )
        self.flow_head = nn.Conv2d(mid, 2 * K, 3, 1, 1, bias=True)
        self.attn_head = nn.Conv2d(mid, K,     3, 1, 1, bias=True)
        self.refine    = nn.Sequential(
            ConvGNAct(dz, dz, 3, 1, 1),
            nn.Dropout2d(dropout_p),
        )
        nn.init.constant_(self.flow_head.weight, 0)
        nn.init.constant_(self.flow_head.bias,   0)
        nn.init.constant_(self.attn_head.bias,   0)

    def _base_grid(self, B, H, W, device):
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        g = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        return g.expand(B, -1, -1, -1)

    def forward(self, z, h, s2, y_prob_prev, u_prev):
        B, C, Hz, Wz = z.shape
        s2_dn = F.interpolate(s2,          size=(Hz,Wz), mode='bilinear', align_corners=False)
        yp_dn = F.interpolate(y_prob_prev, size=(Hz,Wz), mode='bilinear', align_corners=False)
        u_dn  = F.interpolate(u_prev,      size=(Hz,Wz), mode='bilinear', align_corners=False)
        ctx   = torch.cat([z, s2_dn, yp_dn, u_dn], dim=1)
        feat  = self.context_enc(ctx)
        flows = self.flow_head(feat)
        attn  = F.softmax(self.attn_head(feat), dim=1)
        base  = self._base_grid(B, Hz, Wz, z.device)
        out   = torch.zeros_like(z)
        for k in range(self.K):
            offset = torch.tanh(flows[:, 2*k:2*k+2]) * 0.5
            grid_k = (base + offset.permute(0,2,3,1)).clamp(-1, 1)
            z_k    = F.grid_sample(z, grid_k, mode='bilinear',
                                   padding_mode='border', align_corners=True)
            out    = out + attn[:, k:k+1] * z_k
        return self.refine(out) + z

class HighLevelTransition(nn.Module):
    def __init__(self, dh, dz, s3_ch, dropout_p=0.1):
        super().__init__()
        self.fc1     = nn.Linear(dh + dz + s3_ch, dh)
        self.drop    = nn.Dropout(dropout_p)
        self.fc2     = nn.Linear(dh, dh)
        self.mu_head = nn.Linear(dh, dh)
        self.lv_head = nn.Linear(dh, dh)

    def forward(self, h, z, s3):
        feat = torch.cat([h, z.mean(dim=(2,3)), s3.mean(dim=(2,3))], dim=1)
        feat = F.gelu(self.drop(self.fc1(feat)))
        feat = F.gelu(self.fc2(feat))
        mu   = self.mu_head(feat)
        lv   = self.lv_head(feat)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv), mu, lv

class Decoder(nn.Module):
    def __init__(self, dz, s1_ch, s2_ch, s3_ch):
        super().__init__()
        out3 = max(s3_ch, dz//2); out2 = max(s2_ch, dz//4); out1 = max(s1_ch, 32)
        self.fuse3  = UpFuse(dz, s3_ch, out3)
        self.fuse2  = UpFuse(out3, s2_ch, out2)
        self.fuse1  = UpFuse(out2, s1_ch, out1)
        self.refine = ResBlock(out1, out1)
        self.out    = nn.Conv2d(out1, 1, 1)
    def forward(self, z, s1, s2, s3, target_size):
        x = self.fuse3(z, s3); x = self.fuse2(x, s2); x = self.fuse1(x, s1)
        return F.interpolate(self.out(self.refine(x)), size=target_size,
                             mode='bilinear', align_corners=False)

class UncertaintyHead(nn.Module):
    def __init__(self, dz, s1_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(dz+s1_ch, s1_ch, 3, 1, 1),
            nn.Conv2d(s1_ch, 1, 1), nn.Sigmoid())
    def forward(self, z, s1, target_size):
        z_up = F.interpolate(z, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        out  = self.block(torch.cat([z_up, s1], dim=1))
        return F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

class HaltHead(nn.Module):
    def __init__(self, dz, dh):
        super().__init__()
        self.z_pool = nn.Sequential(ConvGNAct(dz, dz//2, 3,1,1), nn.AdaptiveAvgPool2d(1))
        self.mlp = nn.Sequential(
            nn.Linear(dz//2+dh+1, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, z, u, h):
        zf = self.z_pool(z).squeeze(-1).squeeze(-1)
        us = u.mean(dim=(1,2,3))
        return self.mlp(torch.cat([zf, h, us.unsqueeze(1)], dim=1)).squeeze(1)

class DAFlowHiReMed(nn.Module):
    def __init__(self, encoder_pretrained=True, latent_dim_h=640, latent_dim_z=384,
                 K_daflow=6, max_steps=6, min_steps=3, K_inner=3,
                 halt_improvement_threshold=0.1, dropout_p=0.1, recursive_dropout_p=0.2):
        super().__init__()
        from v7_model import ConvNeXTEncoder
        ch = {'s1': 96, 's2': 192, 's3': 384, 's4': 768}
        self.encoder   = ConvNeXTEncoder('convnext_tiny', encoder_pretrained)
        self.z_proj    = nn.Conv2d(ch['s4'], latent_dim_z, 1, bias=False)
        self.low       = DAFlowRefinement(latent_dim_z, ch['s2'], latent_dim_h,
                                          K=K_daflow, dropout_p=dropout_p)
        self.high      = HighLevelTransition(latent_dim_h, latent_dim_z, ch['s3'], dropout_p)
        self.decoder   = Decoder(latent_dim_z, ch['s1'], ch['s2'], ch['s3'])
        self.unc_head  = UncertaintyHead(latent_dim_z, ch['s1'])
        self.halt_head = HaltHead(latent_dim_z, latent_dim_h)
        self.latent_dim_h = latent_dim_h
        self.latent_dim_z = latent_dim_z
        self.K_inner   = K_inner
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.halt_improvement_threshold = halt_improvement_threshold
        self.recursive_dropout_p = recursive_dropout_p

    def forward(self, x, force_steps=None):
        B, _, H, W = x.shape
        dtype = x.dtype; dev = x.device
        s1, s2, s3, s4 = self.encoder(x)
        z  = self.z_proj(s4)
        h  = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
        mu = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
        logvar = torch.zeros(B, self.latent_dim_h, device=dev, dtype=dtype)
        y_logit     = torch.zeros(B, 1, H, W, device=dev, dtype=dtype)
        y_prob_prev = torch.zeros(B, 1, H, W, device=dev, dtype=dtype)
        u_prev      = torch.zeros(B, 1, H, W, device=dev, dtype=dtype)
        max_s   = force_steps if force_steps is not None else self.max_steps
        active  = torch.ones(B, dtype=torch.bool, device=dev)
        outputs = []
        for step in range(max_s):
            if self.training and self.recursive_dropout_p > 0 and step > 0:
                mask = torch.bernoulli(
                    torch.full((B,1,1,1), 1-self.recursive_dropout_p, device=dev))
                z = z * mask
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
                'y_logit': y_logit, 'y_prob': y_prob.clamp(0,1),
                'uncertainty': u_map, 'halt_score': halt_score,
                'mu': mu, 'logvar': logvar,
            })
            if force_steps is None and step + 1 >= self.min_steps:
                active = active & (halt_score < (1.0 - self.halt_improvement_threshold))
                if not active.any(): break
        return outputs

# ── Train ─────────────────────────────────────────────────────────────────────

print('='*60)
print('HiReMed v8 — DAFlow Reasoning (K=6)')
print('='*60)

cfg = T.Config(
    dataset_name='kvasir', use_test_as_val=False,
    encoder_name='convnext_tiny', encoder_pretrained=True,
    image_size=256, batch_size=16, num_workers=0,
    epochs=200, lr_encoder=1e-4, lr_decoder=1e-3,
    disable_cudnn=False, latent_dim_h=640, latent_dim_z=384,
)
T.configure_runtime(cfg)
T.set_seed(cfg.seed)

model = DAFlowHiReMed(
    encoder_pretrained         = True,
    latent_dim_h               = 640,
    latent_dim_z               = 384,
    K_daflow                   = 6,
    max_steps                  = cfg.max_steps,
    min_steps                  = cfg.min_steps,
    K_inner                    = cfg.K_inner,
    halt_improvement_threshold = cfg.halt_improvement_threshold,
    dropout_p                  = cfg.dropout_p,
    recursive_dropout_p        = cfg.recursive_dropout_p,
).to(cfg.device)

total_M = sum(p.numel() for p in model.parameters()) / 1e6
enc_M   = sum(p.numel() for p in model.encoder.parameters()) / 1e6
low_M   = sum(p.numel() for p in model.low.parameters()) / 1e6
print(f'Total  : {total_M:.2f}M  (run_005 was 43.22M)')
print(f'Encoder: {enc_M:.2f}M  ({100*enc_M/total_M:.1f}%)')
print(f'Reason : {total_M-enc_M:.2f}M  ({100*(total_M-enc_M)/total_M:.1f}%)')
print(f'DAFlow : {low_M:.3f}M  (run_005 LowLevel was 5.538M)')

run_dir = Path(REPO_DIR) / 'RESULTS' / 'daflow_k6' / 'run_001'
run_dir.mkdir(parents=True, exist_ok=True)

train_loader, val_loader, test_loader = T.build_dataloaders(cfg)

optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(),   'lr': cfg.lr_encoder},
    {'params': model.z_proj.parameters(),    'lr': cfg.lr_decoder},
    {'params': model.low.parameters(),       'lr': cfg.lr_decoder},
    {'params': model.high.parameters(),      'lr': cfg.lr_decoder},
    {'params': model.decoder.parameters(),   'lr': cfg.lr_decoder},
    {'params': model.unc_head.parameters(),  'lr': cfg.lr_decoder},
    {'params': model.halt_head.parameters(), 'lr': cfg.lr_decoder},
], weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
logger    = T.setup_logger(str(run_dir))

best_dice = 0.0
history   = []

for epoch in range(1, cfg.epochs + 1):
    t0       = time.time()
    train_m  = T.run_train_epoch(model, train_loader, optimizer, cfg, epoch, logger)
    scheduler.step()

    if not torch.isfinite(torch.tensor(train_m['loss'])):
        print(f'WARNING: loss={train_m["loss"]} at epoch {epoch} — skipping val')
        continue

    val_both = T.eval_both_modes(model, val_loader, cfg,
                                  label=f'val ep{epoch}',
                                  logger=logger, current_epoch=epoch)
    val_dice  = val_both['adaptive']['dice']
    avg_steps = val_both['adaptive']['avg_steps']
    elapsed   = time.time() - t0

    history.append({'epoch': epoch, 'loss': train_m['loss'],
                    'dice_1pass': val_both['single_pass']['dice'],
                    'dice_adap': val_dice, 'avg_steps': avg_steps})

    if epoch % 10 == 0 or epoch <= 5:
        print(f'Ep {epoch:03d}: loss={train_m["loss"]:.4f}  '
              f'dice_1p={val_both["single_pass"]["dice"]:.4f}  '
              f'dice_ad={val_dice:.4f}  steps={avg_steps:.2f}  t={elapsed:.0f}s', flush=True)

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), run_dir / 'best.pth')
        print(f'  -> New best: {best_dice:.4f}', flush=True)

# Final test
model.load_state_dict(torch.load(run_dir / 'best.pth', map_location=cfg.device))
model.eval()
final = T.eval_both_modes(model, test_loader, cfg, label='final test')

print()
print('='*60)
print('FINAL TEST — DAFlow v8 vs run_005 (V7-balanced)')
print('='*60)
metrics = [
    ('Dice (adaptive)', final['adaptive']['dice'],    0.9236),
    ('Dice (1-pass)',   final['single_pass']['dice'], 0.9221),
    ('IoU',            final['adaptive']['iou'],      0.8699),
    ('Precision',      final['adaptive']['precision'],0.9416),
    ('Recall',         final['adaptive']['recall'],   0.9218),
    ('avg_steps',      final['adaptive']['avg_steps'],3.00),
]
print(f'{"Metric":<20} {"DAFlow":>10} {"run_005":>10} {"Delta":>10}')
print('-'*55)
for name, val, base in metrics:
    delta = val - base
    print(f'{name:<20} {val:>10.4f} {base:>10.4f} {delta:>+10.4f}')

with open(run_dir / 'result.json', 'w') as f:
    json.dump({'model': 'DAFlowHiReMed', 'K_daflow': 6,
               'total_M': total_M, 'enc_M': enc_M,
               'test_dice': final['adaptive']['dice'],
               'test_iou':  final['adaptive']['iou']}, f, indent=2)
print(f'\nDone. Results saved to {run_dir}')
