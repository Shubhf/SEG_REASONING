"""
train_focus_sweep.py
====================
4-way ablation: FOCUS loss x DAFlow reasoner

Config 1: baseline        — run_005 (concat+conv, no FOCUS loss)
Config 2: focus_only      — run_005 arch + FOCUS uncertainty loss
Config 3: daflow_only     — DAFlow no-cross-attn + no FOCUS loss
Config 4: daflow_focus    — DAFlow no-cross-attn + FOCUS uncertainty loss

Baseline: run_005 Dice 0.9236
Run: nohup python train_focus_sweep.py > focus_sweep_log.txt 2>&1 &
"""

import os, sys, time, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = '/efs/drsanny/visual_extension/storage/Shb_PROJECTS/SEG_REASONING-main'
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'models'))

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

# ── Building blocks ──────────────────────────────────────────────────────────
def _sg(ch, g=32):
    g = min(g, ch)
    while ch % g != 0: g -= 1
    return g

class CGA(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=1):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(ic,oc,k,s,p,bias=False),
                               nn.GroupNorm(_sg(oc),oc), nn.GELU())
    def forward(self, x): return self.b(x)

class RB(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.c1=CGA(ic,oc)
        self.c2=nn.Sequential(nn.Conv2d(oc,oc,3,1,1,bias=False),nn.GroupNorm(_sg(oc),oc))
        self.sk=nn.Conv2d(ic,oc,1,bias=False) if ic!=oc else nn.Identity()
        self.act=nn.GELU()
    def forward(self, x): return self.act(self.c2(self.c1(x))+self.sk(x))

class UF(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.b=RB(ic+sc,oc)
    def forward(self, x, sk):
        x=F.interpolate(x,sk.shape[-2:],mode='bilinear',align_corners=False)
        return self.b(torch.cat([x,sk],1))

# ── FOCUS uncertainty loss ───────────────────────────────────────────────────
def get_boundary(mask, kernel_size=5):
    """Extract boundary regions from binary mask using max-pooling dilation."""
    pad = kernel_size // 2
    dilated  = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=pad)
    eroded   = -F.max_pool2d(-mask.float(), kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary

def focus_uncertainty_loss(u_map, y_true, margin=0.3):
    """
    FOCUS-inspired uncertainty attention loss.
    Forces u_map HIGH at boundaries, LOW at easy interior regions.

    Based on: FOCUS (arxiv 2605.31145) — attention map optimization
    to focus model attention on spatially relevant regions.

    Adapted for HiReMed: instead of VLM attention tokens,
    we supervise the uncertainty map to attend to boundary pixels.

    u_map:  [B, 1, H, W] uncertainty predictions in [0,1]
    y_true: [B, 1, H, W] ground truth binary mask
    margin: minimum gap between boundary and interior uncertainty (like FOCUS margin mu)
    """
    # Align y_true to u_map resolution
    y_res = F.interpolate(y_true.float(), u_map.shape[-2:],
                          mode='nearest')

    # Boundary = pixels that are edges of the mask
    boundary = get_boundary(y_res)              # [B, 1, H, W]

    # Interior = confidently correct inside regions
    interior = y_res * (1.0 - boundary)        # [B, 1, H, W]

    # Background interior = easy background regions
    bg_interior = (1.0 - y_res) * (1.0 - get_boundary(1.0 - y_res))

    # Easy regions = interior foreground + interior background
    easy = (interior + bg_interior).clamp(0, 1)

    # FOCUS margin loss:
    # p_hard = mean uncertainty at boundary pixels (should be HIGH)
    # p_easy = mean uncertainty at easy pixels (should be LOW)
    # Loss = max(0, margin - (p_hard - p_easy))^2

    bnd_sum  = boundary.sum(dim=(1,2,3)) + 1e-5
    easy_sum = easy.sum(dim=(1,2,3)) + 1e-5

    p_hard = (u_map * boundary).sum(dim=(1,2,3)) / bnd_sum   # [B]
    p_easy = (u_map * easy).sum(dim=(1,2,3)) / easy_sum      # [B]

    loss = F.relu(margin - (p_hard - p_easy)).pow(2).mean()
    return loss

# ── DAFlow refinement ────────────────────────────────────────────────────────
def _base_grid(B, H, W, dev):
    xs=torch.linspace(-1,1,W,device=dev); ys=torch.linspace(-1,1,H,device=dev)
    gy,gx=torch.meshgrid(ys,xs,indexing='ij')
    return torch.stack([gx,gy],-1).unsqueeze(0).expand(B,-1,-1,-1)

class DAFlowRefinement(nn.Module):
    def __init__(self, dz, s2_ch, dh, K=6, dp=0.1):
        super().__init__()
        self.K=K; mid=max(dz//2,64)
        self.enc=nn.Sequential(CGA(dz+s2_ch+2,mid),CGA(mid,mid))
        self.fh=nn.Conv2d(mid,2*K,3,1,1,bias=True)
        self.ah=nn.Conv2d(mid,K,3,1,1,bias=True)
        self.ref=nn.Sequential(CGA(dz,dz),nn.Dropout2d(dp))
        nn.init.constant_(self.fh.weight,0); nn.init.constant_(self.fh.bias,0)
        nn.init.constant_(self.ah.bias,0)
    def forward(self, z, h, s2, yp, up):
        B,C,Hz,Wz=z.shape
        s2d=F.interpolate(s2,(Hz,Wz),mode='bilinear',align_corners=False)
        ypd=F.interpolate(yp,(Hz,Wz),mode='bilinear',align_corners=False)
        upd=F.interpolate(up,(Hz,Wz),mode='bilinear',align_corners=False)
        feat=self.enc(torch.cat([z,s2d,ypd,upd],1))
        flows=self.fh(feat); attn=F.softmax(self.ah(feat),dim=1)
        base=_base_grid(B,Hz,Wz,z.device); out=torch.zeros_like(z)
        for k in range(self.K):
            off=torch.tanh(flows[:,2*k:2*k+2])*0.5
            grid=(base+off.permute(0,2,3,1)).clamp(-1,1)
            zk=F.grid_sample(z,grid,mode='bilinear',padding_mode='border',align_corners=True)
            out+=attn[:,k:k+1]*zk
        return self.ref(out)+z

# ── Shared architecture modules ──────────────────────────────────────────────
class HLT(nn.Module):
    def __init__(self, dh, dz, s3, dp=0.1):
        super().__init__()
        self.f1=nn.Linear(dh+dz+s3,dh); self.d=nn.Dropout(dp)
        self.f2=nn.Linear(dh,dh); self.mu=nn.Linear(dh,dh); self.lv=nn.Linear(dh,dh)
    def forward(self, h, z, s3):
        f=torch.cat([h,z.mean((2,3)),s3.mean((2,3))],1)
        f=F.gelu(self.d(self.f1(f))); f=F.gelu(self.f2(f))
        mu=self.mu(f); lv=self.lv(f)
        return mu+torch.randn_like(mu)*torch.exp(0.5*lv),mu,lv

class Dec(nn.Module):
    def __init__(self, dz, s1, s2, s3):
        super().__init__()
        o3=max(s3,dz//2); o2=max(s2,dz//4); o1=max(s1,32)
        self.f3=UF(dz,s3,o3); self.f2=UF(o3,s2,o2); self.f1=UF(o2,s1,o1)
        self.r=RB(o1,o1); self.o=nn.Conv2d(o1,1,1)
    def forward(self, z, s1, s2, s3, sz):
        x=self.f3(z,s3); x=self.f2(x,s2); x=self.f1(x,s1)
        return F.interpolate(self.o(self.r(x)),sz,mode='bilinear',align_corners=False)

class UH(nn.Module):
    def __init__(self, dz, s1):
        super().__init__()
        self.b=nn.Sequential(CGA(dz+s1,s1),nn.Conv2d(s1,1,1),nn.Sigmoid())
    def forward(self, z, s1, sz):
        zu=F.interpolate(z,s1.shape[-2:],mode='bilinear',align_corners=False)
        return F.interpolate(self.b(torch.cat([zu,s1],1)),sz,mode='bilinear',align_corners=False)

class HH(nn.Module):
    def __init__(self, dz, dh):
        super().__init__()
        self.p=nn.Sequential(CGA(dz,dz//2),nn.AdaptiveAvgPool2d(1))
        self.m=nn.Sequential(nn.Linear(dz//2+dh+1,128),nn.GELU(),
                             nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, z, u, h):
        zf=self.p(z).squeeze(-1).squeeze(-1); us=u.mean((1,2,3))
        return self.m(torch.cat([zf,h,us.unsqueeze(1)],1)).squeeze(1)

# ── Full model ───────────────────────────────────────────────────────────────
class HiReMed_v(nn.Module):
    def __init__(self, use_daflow=False, dh=640, dz=384,
                 max_steps=6, min_steps=3, Ki=3,
                 ht=0.1, dp=0.1, rdp=0.2):
        super().__init__()
        from v7_model import ConvNeXTEncoder, LowLevelRefinement
        ch={'s1':96,'s2':192,'s3':384,'s4':768}
        self.encoder=ConvNeXTEncoder('convnext_tiny',True)
        self.z_proj=nn.Conv2d(ch['s4'],dz,1,bias=False)
        if use_daflow:
            self.low=DAFlowRefinement(dz,ch['s2'],dh,K=6,dp=dp)
        else:
            self.low=LowLevelRefinement(dz,dh,ch['s2'],dropout_p=dp)
        self.high=HLT(dh,dz,ch['s3'],dp)
        self.decoder=Dec(dz,ch['s1'],ch['s2'],ch['s3'])
        self.unc_head=UH(dz,ch['s1'])
        self.halt_head=HH(dz,dh)
        self.dh=dh; self.dz=dz; self.Ki=Ki
        self.max_steps=max_steps; self.min_steps=min_steps
        self.ht=ht; self.rdp=rdp

    def forward(self, x, force_steps=None):
        B,_,H,W=x.shape; dtype=x.dtype; dev=x.device
        s1,s2,s3,s4=self.encoder(x); z=self.z_proj(s4)
        h=torch.zeros(B,self.dh,device=dev,dtype=dtype)
        mu=torch.zeros(B,self.dh,device=dev,dtype=dtype)
        lv=torch.zeros(B,self.dh,device=dev,dtype=dtype)
        yl=torch.zeros(B,1,H,W,device=dev,dtype=dtype)
        yp=torch.zeros(B,1,H,W,device=dev,dtype=dtype)
        up=torch.zeros(B,1,H,W,device=dev,dtype=dtype)
        ms=force_steps or self.max_steps
        ac=torch.ones(B,dtype=torch.bool,device=dev); outs=[]
        for step in range(ms):
            if self.training and self.rdp>0 and step>0:
                z=z*torch.bernoulli(torch.full((B,1,1,1),1-self.rdp,device=dev))
            for _ in range(self.Ki): z=self.low(z,h,s2,yp,up)
            h,mu,lv=self.high(h,z,s3); delta=self.decoder(z,s1,s2,s3,(H,W))
            yl=yl+delta; yprob=torch.sigmoid(yl.float()).to(dtype)
            umap=self.unc_head(z,s1,(H,W)).clamp(0,1)
            halt=self.halt_head(z,umap,h).clamp(0,1)
            yp=yprob.detach(); up=umap.detach()
            outs.append({'y_logit':yl,'y_prob':yprob.clamp(0,1),
                         'uncertainty':umap,'halt_score':halt,'mu':mu,'logvar':lv})
            if force_steps is None and step+1>=self.min_steps:
                ac=ac&(halt<(1.0-self.ht))
                if not ac.any(): break
        return outs

# ── Custom total loss with optional FOCUS term ───────────────────────────────
def total_loss_with_focus(outputs, masks, cfg, use_focus=False, focus_weight=0.1):
    """
    Standard v7_train total_loss + optional FOCUS uncertainty attention loss.
    focus_weight: weight for FOCUS loss term (0.1 found stable in experiments)
    """
    # Use standard v7 total_loss
    loss, parts = T.total_loss(outputs, masks, cfg)

    if use_focus:
        focus_l = 0.0
        n = len(outputs)
        decay = 0.7
        for i, out in enumerate(outputs):
            w = decay ** (n - 1 - i)
            fl = focus_uncertainty_loss(
                out['uncertainty'],
                masks,
                margin=0.3
            )
            focus_l += w * fl
        focus_l /= n
        loss = loss + focus_weight * focus_l
        parts['focus'] = focus_l.item()
    else:
        parts['focus'] = 0.0

    return loss, parts

# ── Train one config ─────────────────────────────────────────────────────────
def cM(m): return sum(p.numel() for p in m.parameters())/1e6

def train_one(name, use_daflow, use_focus, epochs=200):
    print(f'\n{"="*60}')
    print(f'{name}  daflow={use_daflow}  focus={use_focus}')
    print(f'{"="*60}')

    for m in list(sys.modules):
        if 'v7_train' in m or m.endswith('_model'): del sys.modules[m]
    import v7_train as T

    cfg=T.Config(dataset_name='kvasir',use_test_as_val=False,
                 encoder_name='convnext_tiny',encoder_pretrained=True,
                 image_size=256,batch_size=16,num_workers=0,
                 epochs=epochs,lr_encoder=1e-4,lr_decoder=1e-3,
                 disable_cudnn=False,latent_dim_h=640,latent_dim_z=384)
    T.configure_runtime(cfg); T.set_seed(cfg.seed)

    model=HiReMed_v(use_daflow=use_daflow,dh=640,dz=384,
                    max_steps=cfg.max_steps,min_steps=cfg.min_steps,
                    Ki=cfg.K_inner,ht=cfg.halt_improvement_threshold,
                    dp=cfg.dropout_p,rdp=cfg.recursive_dropout_p).to(cfg.device)

    tot=cM(model); enc=cM(model.encoder); low=cM(model.low)
    print(f'Total:{tot:.2f}M  Enc:{enc:.2f}M({100*enc/tot:.0f}%)  Low:{low:.3f}M')

    run_dir=Path(REPO_DIR)/'RESULTS'/f'focus_{name}'/'run_001'
    run_dir.mkdir(parents=True,exist_ok=True)

    train_loader,val_loader,test_loader=T.build_dataloaders(cfg)
    opt=torch.optim.AdamW([
        {'params':model.encoder.parameters(),'lr':cfg.lr_encoder},
        {'params':model.z_proj.parameters(),'lr':cfg.lr_decoder},
        {'params':model.low.parameters(),'lr':cfg.lr_decoder},
        {'params':model.high.parameters(),'lr':cfg.lr_decoder},
        {'params':model.decoder.parameters(),'lr':cfg.lr_decoder},
        {'params':model.unc_head.parameters(),'lr':cfg.lr_decoder},
        {'params':model.halt_head.parameters(),'lr':cfg.lr_decoder},
    ],weight_decay=cfg.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    logger=T.setup_logger(str(run_dir)); best=0.0; history=[]

    for epoch in range(1,epochs+1):
        # ── Train epoch manually to inject FOCUS loss ──────────────────────
        model.train(); total_l=0.0; total_d=0.0; nb=0
        for batch in train_loader:
            images=batch['image'].to(cfg.device)
            masks =batch['mask'].to(cfg.device)
            opt.zero_grad(set_to_none=True)
            outputs=model(images)
            loss,parts=total_loss_with_focus(outputs,masks,cfg,
                                              use_focus=use_focus,
                                              focus_weight=0.1)
            if not torch.isfinite(loss): continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            opt.step()
            mets=T.compute_metrics(outputs[-1]['y_logit'],masks,cfg.threshold)
            total_l+=loss.item(); total_d+=mets['dice']; nb+=1

        sch.step()
        train_dice = total_d/max(nb,1)
        train_loss = total_l/max(nb,1)

        if not torch.isfinite(torch.tensor(train_loss)):
            print(f'NaN at epoch {epoch}'); continue

        val=T.eval_both_modes(model,val_loader,cfg,
                              label=f'val ep{epoch}',logger=logger,current_epoch=epoch)
        vd=val['adaptive']['dice']
        history.append({'epoch':epoch,'loss':train_loss,'dice_ad':vd})

        if epoch%10==0 or epoch<=5:
            focus_tag=' +FOCUS' if use_focus else ''
            print(f'  Ep{epoch:03d}{focus_tag}: loss={train_loss:.4f}  '
                  f'dice_1p={val["single_pass"]["dice"]:.4f}  '
                  f'dice_ad={vd:.4f}',flush=True)

        if vd>best:
            best=vd; torch.save(model.state_dict(),run_dir/'best.pth')
            print(f'  ->New best:{best:.4f}',flush=True)

    model.load_state_dict(torch.load(run_dir/'best.pth',map_location=cfg.device))
    model.eval()
    final=T.eval_both_modes(model,test_loader,cfg,label='final test')

    result={'name':name,'use_daflow':use_daflow,'use_focus':use_focus,
            'total_M':tot,'low_M':low,
            'test_dice_1pass':final['single_pass']['dice'],
            'test_dice_adaptive':final['adaptive']['dice'],
            'test_iou':final['adaptive']['iou'],
            'test_precision':final['adaptive']['precision'],
            'test_recall':final['adaptive']['recall']}
    with open(run_dir/'result.json','w') as f: json.dump(result,f,indent=2)

    delta=final['adaptive']['dice']-0.9236
    print(f'DONE {name}: dice={final["adaptive"]["dice"]:.4f} (vs run_005: {delta:+.4f})')
    return result

# ── 4-way ablation ───────────────────────────────────────────────────────────
# (name, use_daflow, use_focus)
CONFIGS = [
    ('baseline',     False, False),  # run_005 equivalent
    ('focus_only',   False, True),   # concat+conv + FOCUS loss
    ('daflow_only',  True,  False),  # DAFlow + no FOCUS
    ('daflow_focus', True,  True),   # DAFlow + FOCUS loss
]

all_results=[]
for name,ud,uf in CONFIGS:
    result=train_one(name,ud,uf,epochs=200)
    all_results.append(result); torch.cuda.empty_cache()

# ── Final table ──────────────────────────────────────────────────────────────
print(f'\n{"="*75}')
print('4-WAY ABLATION: DAFlow x FOCUS Loss')
print(f'{"="*75}')
print(f'{"Config":<18}{"DAFlow":>8}{"FOCUS":>7}{"Low(M)":>8}{"Dice_1p":>9}{"Dice_ad":>9}{"IoU":>8}')
print('-'*75)
for r in all_results:
    print(f'{r["name"]:<18}{str(r["use_daflow"]):>8}{str(r["use_focus"]):>7}'
          f'{r["low_M"]:>8.3f}{r["test_dice_1pass"]:>9.4f}'
          f'{r["test_dice_adaptive"]:>9.4f}{r["test_iou"]:>8.4f}')
print('-'*75)
print(f'{"run_005 (ref)":<18}{"False":>8}{"False":>7}{"5.538":>8}{"0.9221":>9}{"0.9236":>9}{"0.8699":>8}')

with open(Path(REPO_DIR)/'RESULTS'/'focus_ablation.json','w') as f:
    json.dump(all_results,f,indent=2)
print('Saved to RESULTS/focus_ablation.json')
