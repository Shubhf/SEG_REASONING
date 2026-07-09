"""
train_daflow_sweep.py
Ablation: DAFlow with and without cross-attention.
Config A: no_cross_attn  — simple context concat (our current impl)
Config B: cross_attn     — cross-attn between u_prev and z (faithful to DAWarp)
Baseline: run_005 Dice 0.9236
Run: nohup python train_daflow_sweep.py > daflow_sweep_log.txt 2>&1 &
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

# ── Building blocks ─────────────────────────────────────────────────────────
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
        self.c1=CGA(ic,oc); self.c2=nn.Sequential(nn.Conv2d(oc,oc,3,1,1,bias=False),nn.GroupNorm(_sg(oc),oc))
        self.sk=nn.Conv2d(ic,oc,1,bias=False) if ic!=oc else nn.Identity(); self.act=nn.GELU()
    def forward(self, x): return self.act(self.c2(self.c1(x))+self.sk(x))

class UF(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.b=RB(ic+sc,oc)
    def forward(self, x, sk):
        x=F.interpolate(x,sk.shape[-2:],mode='bilinear',align_corners=False)
        return self.b(torch.cat([x,sk],1))

# ── DAFlow base sampling logic ───────────────────────────────────────────────
def _base_grid(B, H, W, dev):
    xs=torch.linspace(-1,1,W,device=dev); ys=torch.linspace(-1,1,H,device=dev)
    gy,gx=torch.meshgrid(ys,xs,indexing='ij')
    return torch.stack([gx,gy],-1).unsqueeze(0).expand(B,-1,-1,-1)

def daflow_sample(z, flows, attn_weights, K):
    B,C,Hz,Wz=z.shape; base=_base_grid(B,Hz,Wz,z.device); out=torch.zeros_like(z)
    for k in range(K):
        off=torch.tanh(flows[:,2*k:2*k+2])*0.5
        grid=(base+off.permute(0,2,3,1)).clamp(-1,1)
        zk=F.grid_sample(z,grid,mode='bilinear',padding_mode='border',align_corners=True)
        out+=attn_weights[:,k:k+1]*zk
    return out

# ── Config A: No cross-attention ─────────────────────────────────────────────
class DAFlow_NoCrossAttn(nn.Module):
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
        out=daflow_sample(z,flows,attn,self.K)
        return self.ref(out)+z

# ── Config B: With cross-attention ───────────────────────────────────────────
class CrossAttn2D(nn.Module):
    def __init__(self, dz, heads=8, dp=0.1):
        super().__init__()
        self.heads=heads; self.hd=max(dz//heads,1); dm=self.hd*heads
        self.q=nn.Conv2d(dz,dm,1,bias=False)
        self.k=nn.Conv2d(1,dm,1,bias=False)
        self.v=nn.Conv2d(dz,dm,1,bias=False)
        self.o=nn.Conv2d(dm,dz,1,bias=False)
        self.n=nn.GroupNorm(_sg(dz),dz); self.d=nn.Dropout(dp)
        self.sc=self.hd**-0.5
    def forward(self, z, u):
        B,C,H,W=z.shape; u=F.interpolate(u,(H,W),mode='bilinear',align_corners=False)
        def rs(x): return x.view(B,self.heads,self.hd,H*W).transpose(-1,-2)
        Q,K,V=rs(self.q(z)),rs(self.k(u)),rs(self.v(z))
        a=self.d(F.softmax((Q@K.transpose(-1,-2))*self.sc,dim=-1))
        out=(a@V).transpose(-1,-2).contiguous().view(B,-1,H,W)
        return self.n(self.o(out)+z)

class DAFlow_CrossAttn(nn.Module):
    def __init__(self, dz, s2_ch, dh, K=6, heads=8, dp=0.1):
        super().__init__()
        self.K=K; mid=max(dz//2,64)
        self.ca=CrossAttn2D(dz,heads,dp)
        self.enc=nn.Sequential(CGA(dz+s2_ch+2,mid),CGA(mid,mid))
        self.fh=nn.Conv2d(mid,2*K,3,1,1,bias=True)
        self.ah=nn.Conv2d(mid,K,3,1,1,bias=True)
        self.ref=nn.Sequential(CGA(dz,dz),nn.Dropout2d(dp))
        nn.init.constant_(self.fh.weight,0); nn.init.constant_(self.fh.bias,0)
        nn.init.constant_(self.ah.bias,0)
    def forward(self, z, h, s2, yp, up):
        B,C,Hz,Wz=z.shape
        za=self.ca(z,up)
        s2d=F.interpolate(s2,(Hz,Wz),mode='bilinear',align_corners=False)
        ypd=F.interpolate(yp,(Hz,Wz),mode='bilinear',align_corners=False)
        upd=F.interpolate(up,(Hz,Wz),mode='bilinear',align_corners=False)
        feat=self.enc(torch.cat([za,s2d,ypd,upd],1))
        flows=self.fh(feat); attn=F.softmax(self.ah(feat),dim=1)
        out=daflow_sample(z,flows,attn,self.K)
        return self.ref(out)+z

# ── Rest of architecture ─────────────────────────────────────────────────────
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

class DAFlowHiReMed(nn.Module):
    def __init__(self, use_cross_attn=False, dh=640, dz=384, K=6,
                 max_steps=6, min_steps=3, Ki=3, ht=0.1, dp=0.1, rdp=0.2):
        super().__init__()
        from v7_model import ConvNeXTEncoder
        ch={'s1':96,'s2':192,'s3':384,'s4':768}
        self.encoder=ConvNeXTEncoder('convnext_tiny',True)
        self.z_proj=nn.Conv2d(ch['s4'],dz,1,bias=False)
        self.low=(DAFlow_CrossAttn(dz,ch['s2'],dh,K,dp=dp) if use_cross_attn
                  else DAFlow_NoCrossAttn(dz,ch['s2'],dh,K,dp=dp))
        self.high=HLT(dh,dz,ch['s3'],dp); self.decoder=Dec(dz,ch['s1'],ch['s2'],ch['s3'])
        self.unc_head=UH(dz,ch['s1']); self.halt_head=HH(dz,dh)
        self.dh=dh; self.dz=dz; self.Ki=Ki
        self.max_steps=max_steps; self.min_steps=min_steps; self.ht=ht; self.rdp=rdp

    def forward(self, x, force_steps=None):
        B,_,H,W=x.shape; dtype=x.dtype; dev=x.device
        s1,s2,s3,s4=self.encoder(x); z=self.z_proj(s4)
        h=torch.zeros(B,self.dh,device=dev,dtype=dtype)
        mu=torch.zeros(B,self.dh,device=dev,dtype=dtype)
        lv=torch.zeros(B,self.dh,device=dev,dtype=dtype)
        yl=torch.zeros(B,1,H,W,device=dev,dtype=dtype)
        yp=torch.zeros(B,1,H,W,device=dev,dtype=dtype)
        up=torch.zeros(B,1,H,W,device=dev,dtype=dtype)
        ms=force_steps or self.max_steps; ac=torch.ones(B,dtype=torch.bool,device=dev); outs=[]
        for step in range(ms):
            if self.training and self.rdp>0 and step>0:
                z=z*torch.bernoulli(torch.full((B,1,1,1),1-self.rdp,device=dev))
            for _ in range(self.Ki): z=self.low(z,h,s2,yp,up)
            h,mu,lv=self.high(h,z,s3); delta=self.decoder(z,s1,s2,s3,(H,W)); yl=yl+delta
            yprob=torch.sigmoid(yl.float()).to(dtype); umap=self.unc_head(z,s1,(H,W)).clamp(0,1)
            halt=self.halt_head(z,umap,h).clamp(0,1); yp=yprob.detach(); up=umap.detach()
            outs.append({'y_logit':yl,'y_prob':yprob.clamp(0,1),'uncertainty':umap,
                         'halt_score':halt,'mu':mu,'logvar':lv})
            if force_steps is None and step+1>=self.min_steps:
                ac=ac&(halt<(1.0-self.ht))
                if not ac.any(): break
        return outs

# ── Train function ───────────────────────────────────────────────────────────
def cM(m): return sum(p.numel() for p in m.parameters())/1e6

def train_one(name, use_ca, epochs=200):
    print(f'\n{"="*60}\n{name}  cross_attn={use_ca}\n{"="*60}')
    for m in list(sys.modules):
        if 'v7_train' in m or m.endswith('_model'): del sys.modules[m]
    import v7_train as T
    cfg=T.Config(dataset_name='kvasir',use_test_as_val=False,encoder_name='convnext_tiny',
                 encoder_pretrained=True,image_size=256,batch_size=16,num_workers=0,
                 epochs=epochs,lr_encoder=1e-4,lr_decoder=1e-3,disable_cudnn=False,
                 latent_dim_h=640,latent_dim_z=384)
    T.configure_runtime(cfg); T.set_seed(cfg.seed)
    model=DAFlowHiReMed(use_cross_attn=use_ca,dh=640,dz=384,K=6,
                        max_steps=cfg.max_steps,min_steps=cfg.min_steps,Ki=cfg.K_inner,
                        ht=cfg.halt_improvement_threshold,dp=cfg.dropout_p,
                        rdp=cfg.recursive_dropout_p).to(cfg.device)
    tot=cM(model); enc=cM(model.encoder); low=cM(model.low)
    print(f'Total:{tot:.2f}M  Enc:{enc:.2f}M({100*enc/tot:.0f}%)  DAFlow:{low:.3f}M')
    run_dir=Path(REPO_DIR)/'RESULTS'/f'daflow_{name}'/'run_001'
    run_dir.mkdir(parents=True,exist_ok=True)
    train_loader,val_loader,test_loader=T.build_dataloaders(cfg)
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':cfg.lr_encoder},
                            {'params':model.z_proj.parameters(),'lr':cfg.lr_decoder},
                            {'params':model.low.parameters(),'lr':cfg.lr_decoder},
                            {'params':model.high.parameters(),'lr':cfg.lr_decoder},
                            {'params':model.decoder.parameters(),'lr':cfg.lr_decoder},
                            {'params':model.unc_head.parameters(),'lr':cfg.lr_decoder},
                            {'params':model.halt_head.parameters(),'lr':cfg.lr_decoder}],
                           weight_decay=cfg.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    logger=T.setup_logger(str(run_dir)); best=0.0; history=[]
    for epoch in range(1,epochs+1):
        t0=time.time(); tm=T.run_train_epoch(model,train_loader,opt,cfg,epoch,logger); sch.step()
        if not torch.isfinite(torch.tensor(tm['loss'])): print(f'NaN ep{epoch}'); continue
        val=T.eval_both_modes(model,val_loader,cfg,label=f'val ep{epoch}',logger=logger,current_epoch=epoch)
        vd=val['adaptive']['dice']; elapsed=time.time()-t0
        history.append({'epoch':epoch,'loss':tm['loss'],'dice_1p':val['single_pass']['dice'],'dice_ad':vd})
        if epoch%10==0 or epoch<=5:
            print(f'  Ep{epoch:03d}: loss={tm["loss"]:.4f} dice_1p={val["single_pass"]["dice"]:.4f} dice_ad={vd:.4f} t={elapsed:.0f}s',flush=True)
        if vd>best: best=vd; torch.save(model.state_dict(),run_dir/'best.pth'); print(f'  ->New best:{best:.4f}',flush=True)
    model.load_state_dict(torch.load(run_dir/'best.pth',map_location=cfg.device)); model.eval()
    final=T.eval_both_modes(model,test_loader,cfg,label='final test')
    result={'name':name,'cross_attn':use_ca,'total_M':tot,'low_M':low,
            'test_dice_1pass':final['single_pass']['dice'],'test_dice_adaptive':final['adaptive']['dice'],
            'test_iou':final['adaptive']['iou'],'test_precision':final['adaptive']['precision'],
            'test_recall':final['adaptive']['recall']}
    with open(run_dir/'result.json','w') as f: json.dump(result,f,indent=2)
    delta=final['adaptive']['dice']-0.9236
    print(f'DONE {name}: dice={final["adaptive"]["dice"]:.4f} (vs run_005: {delta:+.4f})')
    return result

# ── Run ablation ─────────────────────────────────────────────────────────────
all_results=[]
for name,use_ca in [('no_cross_attn',False),('cross_attn',True)]:
    result=train_one(name,use_ca,epochs=200)
    all_results.append(result); torch.cuda.empty_cache()

print(f'\n{"="*70}\nDAFlow ABLATION COMPLETE\n{"="*70}')
print(f'{"Config":<22}{"Low(M)":>7}{"Total(M)":>9}{"Dice_1p":>9}{"Dice_ad":>9}{"IoU":>8}')
print('-'*70)
for r in all_results:
    print(f'{r["name"]:<22}{r["low_M"]:>7.3f}{r["total_M"]:>9.2f}{r["test_dice_1pass"]:>9.4f}{r["test_dice_adaptive"]:>9.4f}{r["test_iou"]:>8.4f}')
print('-'*70)
print(f'{"run_005 (baseline)":<22}{"5.538":>7}{"43.22":>9}{"0.9221":>9}{"0.9236":>9}{"0.8699":>8}')
with open(Path(REPO_DIR)/'RESULTS'/'daflow_ablation.json','w') as f: json.dump(all_results,f,indent=2)
print('Saved to RESULTS/daflow_ablation.json')
