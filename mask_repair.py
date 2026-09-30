"""
mask_repair.py
=========================================================
Same cross-attention fusion architecture as
train_mask_repair_synapse_fused_crossattn.py (currently the best-performing
fusion mechanism: near-flat Dice across k=0..5, real HD95 improvement),
with a K-STEP SCHEDULED-SAMPLING UNROLL added to the training loop.

WHY THIS EXISTS: real diagnosis from the mentor group chat -- Aryan:
"ye asey girega nhi first of all either kuch galat kr rahe ho ya tumhara
model dhangse train nhi ho raha" (it shouldn't decline like this -- either
something's wrong, or the model isn't training properly). The k=0->k=5
decline seen in ALL THREE fusion mechanisms (addition, cross-attention,
LoMix-learnable) traces to a real training/inference MISMATCH, not
insufficient epochs (ruled out: more epochs on the same flawed objective
converges harder to the same wrong behavior, doesn't fix it):

  E1 was only ever trained on two extremes -- fully-synthetic-corrupted
  input (learn to fix it) or exact-identity input (learn to leave it
  alone). At real inference, k>=1's input is NEITHER of those -- it's
  E1's OWN previous output: already fairly good, with small REAL errors.
  E1 has never seen that specific situation during training, so it has
  no learned behavior for "mostly right, small errors, refine gently" --
  every pass changes something, and across k=1..5 that compounds into
  decline.

THE FIX: during training, occasionally run repair_model on ITS OWN output
from a previous step (not just the dataset's synthetic corruption/
real-error/identity samples), so it directly learns what its real
inference-time input distribution actually looks like.

ONE REAL SUBTLETY: argmax (needed to turn a step's logits into next-step's
mask input) has NO GRADIENT in PyTorch -- true end-to-end backprop through
the whole chained sequence isn't actually possible. This uses SCHEDULED
SAMPLING instead: each step's input comes from the model's own (detached)
prediction, but each step's OWN forward pass still gets a real,
gradient-tracked loss against the true GT, summed across all K steps into
one optimizer.step() call. Standard, stable technique for exactly this
situation -- not a compromise, the actual correct approach here.

Step 1 uses the full combined_loss (EMCAD's real loss + corruption-focus/
identity extras, since step 1's input has a real, known corruption_mask
from the dataset). Steps 2..K use the base EMCAD loss only (no
corruption-focus/identity terms -- there's no longer a "known corruption
mask" once the input is the model's own prediction, not synthetic
damage) -- a disclosed simplification, not an oversight.

K_UNROLL=3 by default (env var overridable) -- each additional step
roughly multiplies per-batch compute by that much, since it's K real
forward+backward passes through repair_model per training step, not free.

Run:
  K_UNROLL=3 CUDA_VISIBLE_DEVICES=<idx> PYTHONUNBUFFERED=1 \
  nohup python mask_repair.py > logs/mask_repair_crossattn_unrolled.log 2>&1 &
"""

import os, sys, glob, re, json, math, time
import multiprocessing as mp
from scene_graph_core import extract_scene_graph, compare_scene_graphs  # pure numpy/scipy, no torch -- see corruption_health_check() below; one-directional import, scene_graph_core.py never imports from here
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

_PROJ = Path(__file__).parent.resolve()
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

from lib.networks import EMCADNet
from lib.losses import DiceLoss, powerset

if not hasattr(F, 'scaled_dot_product_attention'):
    raise RuntimeError('This torch has no F.scaled_dot_product_attention (needs >= 2.0).')

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    def _sdpa(q, k, v):
        with sdpa_kernel(_SDPA_BACKENDS):
            return F.scaled_dot_product_attention(q, k, v)
except Exception:  # torch < 2.1 -- no backend selection API
    def _sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

NUM_CLASSES = 9
ENCODER = os.environ.get('EMCAD_ENCODER', 'pvt_v2_b2')
IMG_SIZE = int(os.environ.get('IMG_SIZE', '224'))
SEED = int(os.environ.get('SEED', '42'))
EPOCHS = int(os.environ.get('EPOCHS', '50'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '8'))
K_UNROLL = int(os.environ.get('K_UNROLL', '3'))  # scheduled-sampling unroll depth, see module docstring -- Aryan: try 6, 10, 11 as candidate step counts
HEALTH_CHECK_EVERY_N_EPOCHS = int(os.environ.get('HEALTH_CHECK_EVERY_N_EPOCHS', '20'))  # 0 disables the periodic corruption health check entirely
MAX_SEVERITY = float(os.environ.get('MAX_SEVERITY', '1.0'))  # sweepable ceiling on corruption severity -- see severity_for_k
SELF_GEN_FRACTION = float(os.environ.get('SELF_GEN_FRACTION', '0.5'))  # for steps 2..K: fraction of the time to use the model's OWN live prediction instead of fresh k-severity corruption -- addresses exposure bias (train/inference input-distribution mismatch), see run_epoch's docstring
ADAPTIVE_BLEND = os.environ.get('ADAPTIVE_BLEND', '0') == '1'  # False (default) reproduces the exact validated 0.8469 result -- global scalar alpha, identical for every pixel. True switches to per-pixel, confidence-modulated alpha -- see blend_with_current's docstring
MONOTONIC_LOSS_WEIGHT = float(os.environ.get('MONOTONIC_LOSS_WEIGHT', '0.0'))  # OFF by default -- at 0.0 this contributes exactly zero loss and zero gradient, so an unmodified launch command reproduces the EXACT pre-existing loss, byte for byte, matching "everything stays the same, just update the loss function" literally. Set nonzero (e.g. 0.1) to explicitly enforce Loss(pred_1,GT) > Loss(pred_2,GT) > ... > Loss(pred_K,GT), a zero-margin hinge -- see run_epoch's docstring for exactly what this does and does not guarantee, and does NOT change any already-running or already-finished checkpoint's directory tag unless it's actually turned on (see sweep_tag in main()).
LR = float(os.environ.get('LR', '1e-4'))
VAL_HOLDOUT_CASES = int(os.environ.get('VAL_HOLDOUT_CASES', '3'))
VAL_EVERY = int(os.environ.get('VAL_EVERY', '1'))  # validate on the held-out cases every N epochs; best.pth is chosen ONLY from these scores
VAL_K_MAX = int(os.environ.get('VAL_K_MAX', str(max(5, K_UNROLL))))  # validation score = mean Dice over k=1..VAL_K_MAX. Default max(5, K_UNROLL): steps beyond the trained range are always covered, and K=3 keeps the same k=1..5 range as before
EVAL_K_MAX = int(os.environ.get('EVAL_K_MAX', str(max(5, K_UNROLL))))  # last step scored on the test set; same default rule, so K=6 is evaluated through k=6 instead of being cut at 5
CORRUPTION_FOCUS_WEIGHT = float(os.environ.get('CORRUPTION_FOCUS_WEIGHT', '2.0'))
IDENTITY_WEIGHT = float(os.environ.get('IDENTITY_WEIGHT', '0.3'))
IDENTITY_BATCH_FRACTION = float(os.environ.get('IDENTITY_BATCH_FRACTION', '0.15'))
REAL_ERROR_FRACTION = float(os.environ.get('REAL_ERROR_FRACTION', '0.5'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(SEED)
np.random.seed(SEED)

_RAW_ID_MAP = {'spleen': 1, 'right_kidney': 2, 'left_kidney': 3, 'gallbladder': 4,
               'pancreas': 11, 'liver': 6, 'stomach': 7, 'aorta': 8}
_CLASS_ORDER = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder',
                'pancreas', 'liver', 'stomach', 'aorta']
_RAW_TO_9 = np.zeros(14, dtype=np.uint8)
for _seq_id, _name in enumerate(_CLASS_ORDER, start=1):
    _RAW_TO_9[_RAW_ID_MAP[_name]] = _seq_id

_DEFAULT_CKPT = {
    'pvt_v2_b0': 'emcad_pretrained/pvt_v2_b0_emcad_synapse.pth',
    'pvt_v2_b2': 'emcad_pretrained/pvt_v2_b2_emcad_synapse.pth',
}
CKPT_PATH = os.environ.get('EMCAD_CKPT_PATH', '') or _DEFAULT_CKPT.get(ENCODER, '')
if not CKPT_PATH or not Path(CKPT_PATH).exists():
    sys.exit(f'{CKPT_PATH or "(no default for this encoder)"} not found -- run '
              f'download_emcad_synapse_weights.py first (see train_mask_repair_synapse.py '
              f'for the full setup, unchanged here).')


# ── Checkpoint loading (identical to train_mask_repair_synapse.py) ─────────

def load_emcad_checkpoint(model, path: str) -> None:
    raw = torch.load(path, map_location='cpu')
    if isinstance(raw, dict) and 'model' in raw and isinstance(raw['model'], dict):
        state_dict = raw['model']
    elif isinstance(raw, dict) and all(hasattr(v, 'shape') for v in raw.values()):
        state_dict = raw
    else:
        sys.exit(f'Unrecognised checkpoint format at {path}')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f'WARNING: {len(missing)} missing, {len(unexpected)} unexpected keys loading {path}')
    else:
        print(f'Checkpoint loaded with an EXACT key match — good sign.')


STAGE_CHANNELS = [64, 128, 320, 512]  # pvt_v2_b2's real per-stage embed_dims, confirmed earlier this session
DECODER_OUT_CHANNELS = [512, 320, 128, 64]  # DecoderEMCAD's real dec_outs=[d4,d3,d2,d1] channel counts, confirmed from lib/decoders.py's EMCAD class __init__ (channels=[512,320,128,64] default) -- REVERSE order/progression from STAGE_CHANNELS, coarsest-to-finest matching the decoder's own upsampling direction, not the encoder's


class TimestepEmbedding(nn.Module):
    """Standard diffusion-style timestep embedding -- sinusoidal encoding
    at multiple frequencies (the same construction as the original
    Transformer's positional encoding, and every diffusion model's own
    timestep embedding, e.g. DDPM), projected through a small MLP.
    Genuine positional encoding, not a raw scalar -- replaces the earlier
    raw-k/k_max-channel-concat mechanism entirely, per direct request
    ("time step balance... positional encoding information") to actually
    do what the diffusion analogy this project has been drawing on
    literally means, not a simplified stand-in for it."""

    def __init__(self, dim: int = 128):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )

    def sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] float, k/k_max in [0,1]. Scaled up before the sinusoidal
        projection -- left in [0,1] directly, the higher frequencies barely
        vary across the whole range, collapsing much of the encoding's
        expressiveness. 1000x matches common diffusion timestep ranges
        (e.g. DDPM's own 1000 steps), giving the frequencies real room to
        differentiate between k values."""
        t = t * 1000.0
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


class FiLMModulation(nn.Module):
    """Feature-wise linear modulation: feat -> feat * (1 + scale) + shift,
    scale/shift both derived from the timestep embedding via a
    ZERO-INITIALIZED linear projection -- an exact identity at
    initialization (same discipline as CrossScaleAttention's zero-init,
    and the fix for the k-channel bug found earlier: the new signal
    contributes NOTHING until training actually earns a reason to use
    it, rather than injecting noise into an already-working model from
    step one)."""

    def __init__(self, channels: int, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(embed_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(t_embed).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return feat * (1 + scale) + shift


class CrossScaleAttention(nn.Module):
    """beta (query) attends over alpha (key/value) -- E1's current guess
    asks 'where in the real image does this look right or wrong', not the
    reverse (alpha has no reason to attend to a possibly-corrupted mask).
    Full spatial attention: every (h,w) position in beta attends over
    ALL (h,w) positions in alpha, not just the same position -- lets a
    corrupted region pull information from the correct anatomical
    location even if the corruption also shifted apparent position.

    Zero-init output projection + residual (beta + attn_out): identity at
    init, matching this project's own AttentionSite2d convention
    (attention_site2d.py) -- attention contributes ~0 at the start of
    training and has to earn its contribution, rather than immediately
    overwriting beta's own signal before the projections have learned
    anything useful."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, f'channels={channels} not divisible by num_heads={num_heads}'
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.o_proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.o_proj.weight)  # identity-at-init

    def forward(self, beta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        B, C, H, W = beta.shape
        q, k, v = self.q_proj(beta), self.k_proj(alpha), self.v_proj(alpha)

        def to_heads(t):
            return t.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn_out = _sdpa(to_heads(q), to_heads(k), to_heads(v))
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return beta + self.o_proj(attn_out)


def build_models():
    print(f'Building base_model (frozen) — encoder={ENCODER}, activation=relu6')
    base_model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER, pretrain=False, activation='relu6')
    load_emcad_checkpoint(base_model, CKPT_PATH)
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    print('Building repair_model — backbone/conv copied from base (trainable), '
          'decoder/heads SHARED with base (frozen, same objects)')
    repair_model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER, pretrain=False, activation='relu6')
    repair_model.backbone.load_state_dict(base_model.backbone.state_dict())
    repair_model.conv.load_state_dict(base_model.conv.state_dict())  # plain 1-channel adapter again -- k-information
                                                                       # now carried by TimestepEmbedding+FiLM below,
                                                                       # not a raw scalar concatenated at the input.
                                                                       # REPLACES the earlier k-channel mechanism
                                                                       # entirely (see module docstring).
    repair_model.decoder = base_model.decoder
    repair_model.out_head1 = base_model.out_head1
    repair_model.out_head2 = base_model.out_head2
    repair_model.out_head3 = base_model.out_head3
    repair_model.out_head4 = base_model.out_head4

    print('Building TimestepEmbedding + 4 FiLMModulation modules (one per scale, '
          f'channels={STAGE_CHANNELS}) — genuine sinusoidal positional encoding, '
          'injected via zero-init FiLM (identity at init, same discipline as '
          'CrossScaleAttention and the fixed k-channel bug)')
    timestep_embed = TimestepEmbedding(dim=128)
    film_modules = nn.ModuleList([FiLMModulation(c, embed_dim=128) for c in STAGE_CHANNELS])

    print(f'Building 4 more FiLMModulation modules for the DECODER pathway (channels='
          f'{DECODER_OUT_CHANNELS}) — real diffusion U-Nets inject timestep conditioning '
          'throughout BOTH encoder and decoder, not just the encoder (the earlier gap, '
          'directly flagged: "time step tumne decoder mein nhi kiya?"). The decoder itself '
          'stays completely frozen -- these new, trainable FiLM layers modulate the '
          'decoder\'s OUTPUT activations only, never touching its own weights.')
    decoder_film_modules = nn.ModuleList([FiLMModulation(c, embed_dim=128) for c in DECODER_OUT_CHANNELS])

    film_params = (sum(p.numel() for p in timestep_embed.parameters()) +
                    sum(p.numel() for p in film_modules.parameters()) +
                    sum(p.numel() for p in decoder_film_modules.parameters()))
    print(f'timestep_embed + film_modules + decoder_film_modules: {film_params/1e6:.4f}M trainable params')

    trainable = (sum(p.numel() for p in repair_model.backbone.parameters()) +
                 sum(p.numel() for p in repair_model.conv.parameters()) + film_params)
    frozen = sum(p.numel() for p in repair_model.decoder.parameters())
    print(f'repair_model: {trainable/1e6:.3f}M trainable (backbone+conv+timestep), '
          f'{frozen/1e6:.3f}M frozen (decoder+heads, shared with base -- weights untouched, '
          f'only decoder_film_modules touches the decoder\'s OUTPUT)')

    print(f'Building 4 CrossScaleAttention modules (one per scale, channels={STAGE_CHANNELS})')
    attn_modules = nn.ModuleList([CrossScaleAttention(c) for c in STAGE_CHANNELS]).to(DEVICE)
    attn_params = sum(p.numel() for p in attn_modules.parameters())
    print(f'attn_modules: {attn_params/1e6:.4f}M trainable params')

    return (base_model.to(DEVICE), repair_model.to(DEVICE), attn_modules.to(DEVICE),
            timestep_embed.to(DEVICE), film_modules.to(DEVICE), decoder_film_modules.to(DEVICE))


# ── Fused forward: the actual architecture change ───────────────────────────

def fused_forward(base_model: EMCADNet, repair_model: EMCADNet, attn_modules: nn.ModuleList,
                   timestep_embed: TimestepEmbedding, film_modules: nn.ModuleList,
                   decoder_film_modules: nn.ModuleList,
                   image: torch.Tensor, mask_intensity: torch.Tensor, k: int, k_max: int) -> List[torch.Tensor]:
    """image: [B,1,H,W] real image (raw values, matching EMCAD's real
    preprocessing). mask_intensity: [B,1,H,W] corrupted-mask class-index
    encoding. k, k_max: which unroll step this is (1-indexed) and the
    total -- encoded via a genuine sinusoidal timestep embedding
    (TimestepEmbedding) and injected into E1's own backbone features at
    all 4 scales via zero-init FiLM modulation (FiLMModulation), not a
    raw scalar channel concatenated at the input. Returns [p4,p3,p2,p1].

    alpha (from the frozen base encoder, real image) is computed under
    no_grad explicitly -- belt-and-suspenders on top of
    requires_grad=False, since a stray forward pass without no_grad still
    builds an autograd graph even if the leaves don't require grad,
    wasting memory on activations that will never be backpropped through."""
    with torch.no_grad():
        img_in = base_model.conv(image) if image.size(1) == 1 else image
        alpha1, alpha2, alpha3, alpha4 = base_model.backbone(img_in)

    mask_in = repair_model.conv(mask_intensity) if mask_intensity.size(1) == 1 else mask_intensity
    beta1, beta2, beta3, beta4 = repair_model.backbone(mask_in)

    t = torch.full((mask_intensity.shape[0],), k / k_max, device=mask_intensity.device)
    t_embed = timestep_embed(t)
    beta1 = film_modules[0](beta1, t_embed)
    beta2 = film_modules[1](beta2, t_embed)
    beta3 = film_modules[2](beta3, t_embed)
    beta4 = film_modules[3](beta4, t_embed)

    gamma1 = attn_modules[0](beta1, alpha1)
    gamma2 = attn_modules[1](beta2, alpha2)
    gamma3 = attn_modules[2](beta3, alpha3)
    gamma4 = attn_modules[3](beta4, alpha4)

    dec_outs = repair_model.decoder(gamma4, [gamma3, gamma2, gamma1])
    # dec_outs = [d4, d3, d2, d1], channels=[512,320,128,64] (DECODER_OUT_CHANNELS) --
    # SAME t_embed reused, no need to recompute -- injecting timestep
    # conditioning into the decoder pathway too, matching real diffusion
    # U-Nets (conditioning throughout encoder AND decoder). Decoder's own
    # weights untouched -- only its output activations get modulated,
    # via these separate, trainable decoder_film_modules.
    dec_outs = [
        decoder_film_modules[0](dec_outs[0], t_embed),
        decoder_film_modules[1](dec_outs[1], t_embed),
        decoder_film_modules[2](dec_outs[2], t_embed),
        decoder_film_modules[3](dec_outs[3], t_embed),
    ]
    p4 = repair_model.out_head4(dec_outs[0])
    p3 = repair_model.out_head3(dec_outs[1])
    p2 = repair_model.out_head2(dec_outs[2])
    p1 = repair_model.out_head1(dec_outs[3])
    p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
    p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
    p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
    p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')
    return [p4, p3, p2, p1]


# ── Real-error + image precompute (extends the unfused version's approach
#    to also carry the image, since fused_forward needs it) ────────────────

@torch.no_grad()
def precompute_triples(base_model, image_size: int = IMG_SIZE) -> Path:
    """(image, gt, base_pred) triples, built together in ONE pass from the
    raw .npz files -- guaranteed aligned by construction. See module
    docstring for why this can't safely reuse mask_train.npy."""
    all_dir = _PROJ / 'data' / 'synapse_all'
    all_dir.mkdir(parents=True, exist_ok=True)  # REAL BUG FOUND on a genuinely fresh environment: this directory was never explicitly created anywhere in this script -- it only worked on the original server because an unrelated, earlier script (train_v10_synapse_joint9.py) happened to create it first. Any fresh setup that runs only this script hits a FileNotFoundError at the final np.savez write, after already doing all the real (expensive) precompute work -- confirmed directly from a real crash log.
    out_path = all_dir / 'mask_repair_fused_triples_train.npz'
    if out_path.exists():
        print(f'{out_path} already present — skipping precompute.')
        return out_path

    src_dir = _PROJ / 'synapse' / 'train_npz_new'
    if not src_dir.is_dir():
        sys.exit(f'{src_dir} not found — needed to build (image, gt, base_pred) triples.')

    _CASE_RE = re.compile(r'case(\d+)')
    files = sorted(glob.glob(str(src_dir / '*.npz')))
    assert files, f'no npz files in {src_dir}'
    case_ids = [int(_CASE_RE.search(Path(f).stem).group(1)) for f in files]
    unique_cases = sorted(set(case_ids))
    val_cases = set(unique_cases[-VAL_HOLDOUT_CASES:]) if VAL_HOLDOUT_CASES > 0 else set()
    train_cases = set(unique_cases) - val_cases
    print(f'precompute_triples: {len(train_cases)} train cases (VAL_HOLDOUT_CASES={VAL_HOLDOUT_CASES})')

    base_model.eval()
    images, gts, preds = [], [], []
    for f, cid in zip(files, case_ids):
        if cid not in train_cases:
            continue
        d = np.load(f)
        img, lab = d['image'], d['label']
        gt = _RAW_TO_9[lab.astype(np.uint8)]

        x, y = img.shape
        if x != image_size or y != image_size:
            img_r = zoom(img, (image_size / x, image_size / y), order=3)
            gt_r = zoom(gt, (image_size / x, image_size / y), order=0)
        else:
            img_r, gt_r = img, gt

        inp = torch.from_numpy(img_r.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = base_model(inp, mode='test')
        p1 = out[-1] if isinstance(out, (list, tuple)) else out
        pred = p1.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        images.append(img_r.astype(np.float32))
        gts.append(gt_r.astype(np.uint8))
        preds.append(pred)

        if len(images) % 200 == 0:
            print(f'  precompute_triples: {len(images)} slices done...')

    np.savez(out_path, image=np.stack(images), gt=np.stack(gts), base_pred=np.stack(preds))
    print(f'precompute_triples: saved {len(images)} (image, gt, base_pred) triples -> {out_path}')
    return out_path


# ── Corruption (identical to train_mask_repair_synapse.py) ─────────────────

def severity_for_k(k: int, k_max: int, max_severity: float = None, min_severity: float = 0.15) -> float:
    """Aryan's explicit schedule: k=1 is the MOST severe corruption
    (hardest correction task), decreasing at every successive k --
    "K==1 then most difficult, K==2 then mask corruption should be less
    than K==1... and so on." Linear from max_severity at k=1 down to
    min_severity at k=k_max (never exactly 0 -- k=k_max should still be a
    real, if light, refinement task, not a no-op).

    max_severity defaults to the module-level MAX_SEVERITY (env var
    MAX_SEVERITY, default 1.0) rather than being hardcoded -- Aryan's own
    diffusion-schedule analogy: just as a diffusion model's noise ceiling
    is a real, sweepable hyperparameter (his own worked example: "suppose
    you tested at a 30% level... normalize that 0-30% range to 0-1"),
    this project's corruption ceiling should be equally sweepable, not
    fixed at 100%. Run the same training multiple times with
    MAX_SEVERITY=0.3, 0.5, 1.0 to compare, exactly as his example
    describes -- this function alone doesn't run that sweep, but makes it
    a one-env-var change instead of a code edit."""
    if max_severity is None:
        max_severity = MAX_SEVERITY
    if k_max <= 1:
        return max_severity
    frac = (k - 1) / (k_max - 1)
    return max_severity - frac * (max_severity - min_severity)


def corrupt_mask(mask: np.ndarray, num_classes: int = NUM_CLASSES, severity: float = 1.0,
                  p_erode_dilate: float = 0.5, p_remove_blob: float = 0.3,
                  p_add_blob: float = 0.3, min_remaining_fraction: float = 0.25) -> np.ndarray:
    """severity in [0,1] scales EVERY corruption probability and the
    erosion/dilation depth -- see severity_for_k() for how this ties to
    the unroll step k. severity=1.0 reproduces the original (pre-k-aware)
    behavior exactly, EXCEPT for the min_remaining_fraction fix below,
    which applies regardless of severity (a real bug, not a severity
    tuning knob).

    REAL BUG FOUND AND FIXED via scene_graph_analysis.py, on real
    samples: the "remove blob" branch used to zero out the ENTIRE blob
    unconditionally. For a class with a single connected component (the
    common case -- most organs in a given slice), that means COMPLETE
    deletion of a whole, clearly-present organ. Confirmed directly: 3 of
    4 real samples lost at least one whole organ at k=1's severity, one
    sample lost its ONLY visible organ entirely. A real trained
    segmentation model essentially never produces this failure mode --
    it misses boundaries, blurs edges, misclassifies ambiguous pixels,
    but does not typically omit an entire visible organ outright. This
    was very plausibly teaching E1 to solve "restore a completely absent
    organ from nothing" -- a different, coarser, easier task than "refine
    a mostly-correct prediction," which is what real inference actually
    requires. Fixed: erode heavily, but NEVER past min_remaining_fraction
    of the blob's original size -- partial, realistic under-segmentation
    instead of total omission, regardless of how the random draw or
    severity would otherwise have zeroed it completely."""
    p_erode_dilate *= severity
    p_remove_blob *= severity
    p_add_blob *= severity
    max_iterations = max(1, round(3 * severity))  # depth of erosion/dilation also scales down with severity

    from scipy import ndimage
    out = mask.copy()
    H, W = mask.shape
    for c in range(1, num_classes):
        class_mask = (mask == c)
        if not class_mask.any():
            continue
        labeled, n_blobs = ndimage.label(class_mask)
        for blob_id in range(1, n_blobs + 1):
            blob = labeled == blob_id
            r = np.random.random()
            if r < p_remove_blob:
                target_remaining = max(int(blob.sum() * min_remaining_fraction), 1)
                eroded = blob
                for _ in range(50):  # bounded -- erosion converges in far fewer steps in practice, this is just a safety cap against an unexpected infinite loop
                    next_eroded = ndimage.binary_erosion(eroded)
                    if next_eroded.sum() <= target_remaining or not next_eroded.any():
                        break
                    eroded = next_eroded
                out[blob & ~eroded] = 0
            elif r < p_remove_blob + p_erode_dilate:
                iterations = np.random.randint(1, max_iterations + 1)
                if np.random.random() < 0.5:
                    eroded = ndimage.binary_erosion(blob, iterations=iterations)
                    out[blob & ~eroded] = 0
                else:
                    dilated = ndimage.binary_dilation(blob, iterations=iterations)
                    new_pixels = dilated & ~blob & (mask == 0)
                    out[new_pixels] = c
    if np.random.random() < p_add_blob:
        spurious_class = np.random.randint(1, num_classes)
        cy, cx = np.random.randint(H // 4, 3 * H // 4), np.random.randint(W // 4, 3 * W // 4)
        radius = np.random.randint(3, max(4, round(15 * severity)))
        yy, xx = np.ogrid[:H, :W]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        out[blob & (mask == 0)] = spurious_class
    return out.astype(np.uint8)


def corruption_health_check(gt_samples: List[np.ndarray], k_max: int = None, max_severity: float = None,
                             max_deletion_rate: float = 0.0) -> bool:
    """Periodic, model-free health check -- catches a REGRESSION of the
    exact bug found and fixed via scene_graph_analysis.py: corrupt_mask()
    completely deleting a whole organ. Never touches the model, gradients,
    or GPU -- pure numpy/scipy, same corruption function the training loop
    already calls every step, just re-checked structurally here.

    max_deletion_rate=0.0 means ANY complete deletion across the sampled
    slices is treated as a health-check FAILURE and printed loudly -- the
    min_remaining_fraction floor in corrupt_mask() should make this
    structurally impossible now, so a failure here would mean either that
    floor logic broke, or corrupt_mask() was edited since without
    preserving it. Returns True (healthy) / False (regression detected)
    so callers can decide what to do with the result; this function only
    checks and reports, it never raises or halts training itself."""
    if k_max is None:
        k_max = K_UNROLL
    if max_severity is None:
        max_severity = MAX_SEVERITY

    total_deletions, total_checks = 0, 0
    for gt in gt_samples:
        gt_nodes = extract_scene_graph(gt)
        for k in range(1, k_max + 1):
            severity = severity_for_k(k, k_max, max_severity=max_severity)
            corrupted = corrupt_mask(gt.copy(), severity=severity)
            diff = compare_scene_graphs(gt_nodes, extract_scene_graph(corrupted))
            total_deletions += len(diff['vanished'])
            total_checks += 1

    deletion_rate = total_deletions / max(total_checks, 1)
    healthy = deletion_rate <= max_deletion_rate
    status = 'PASS' if healthy else 'FAIL -- REGRESSION DETECTED'
    print(f'[corruption health check] {status}: {total_deletions} whole-organ deletions '
          f'across {total_checks} (sample, k) checks (rate={deletion_rate:.3f}, threshold={max_deletion_rate})')
    if not healthy:
        print('[corruption health check] corrupt_mask() may have regressed to the whole-organ-deletion '
              'bug found via scene_graph_analysis.py -- check min_remaining_fraction is still applied.')
    return healthy


class FusedMaskRepairDataset(Dataset):
    """Draws image+gt+corrupted_input all from the SAME precomputed triple
    set (see precompute_triples) -- unlike the unfused version, there's no
    separate mask_train.npy source anymore, since every sample now needs a
    correctly-paired image too."""

    def __init__(self, triples_path: Path):
        data = np.load(triples_path)
        self.images = data['image']      # (N, H, W) float
        self.gts = data['gt']            # (N, H, W) uint8
        self.base_preds = data['base_pred']  # (N, H, W) uint8
        print(f'FusedMaskRepairDataset: {len(self.gts)} triples loaded')

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        image = self.images[idx]
        gt = self.gts[idx]

        r = np.random.random()
        use_real_error = r < REAL_ERROR_FRACTION
        is_identity = (not use_real_error) and r < REAL_ERROR_FRACTION + IDENTITY_BATCH_FRACTION

        if use_real_error:
            corrupted = self.base_preds[idx]  # perfectly aligned with THIS idx's gt and image
        elif is_identity:
            corrupted = gt.copy()
        else:
            corrupted = corrupt_mask(gt)
        corruption_mask = (corrupted != gt)

        return {
            'image': torch.from_numpy(image).unsqueeze(0).float(),
            'corrupted_input': torch.from_numpy((corrupted.astype(np.float32) / (NUM_CLASSES - 1))).unsqueeze(0),
            'gt_idx': torch.from_numpy(gt.astype(np.int64)),
            'corruption_mask': torch.from_numpy(corruption_mask),
            'is_identity': torch.tensor(is_identity),
        }


# ── Loss (identical composition to train_mask_repair_synapse.py) ───────────

_ce_loss_fn = nn.CrossEntropyLoss()
_dice_loss_fn = DiceLoss(NUM_CLASSES)
W_CE, W_DICE = 0.3, 0.7


def ce_dice_loss(logits, target):
    return W_CE * _ce_loss_fn(logits, target) + W_DICE * _dice_loss_fn(logits, target, softmax=True)


def dice_bce_loss(logits, target_idx, pixel_weight=None):
    B, C, H, W = logits.shape
    target_onehot = F.one_hot(target_idx, num_classes=C).permute(0, 3, 1, 2).float()
    if pixel_weight is not None:
        bce_map = F.binary_cross_entropy_with_logits(logits, target_onehot, reduction='none')
        w = pixel_weight.unsqueeze(1)
        bce = (bce_map * w).sum() / (w.sum() * C + 1e-6)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, target_onehot)
    probs = torch.sigmoid(logits)
    inter = (probs * target_onehot).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = 1.0 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()
    return bce + dice


def combined_loss(preds, gt_idx, corruption_mask, is_identity, supervision='mutation'):
    n = len(preds)
    idxs = list(range(n))
    if supervision == 'mutation':
        subsets = [s for s in powerset(idxs) if s]
    elif supervision == 'deep_supervision':
        subsets = [[i] for i in idxs]
    else:
        subsets = [[idxs[-1]]]

    B, H, W = gt_idx.shape
    is_identity_map = is_identity.view(B, 1, 1).expand(B, H, W).to(gt_idx.device)
    extra_weight = torch.where(
        is_identity_map,
        torch.full((B, H, W), IDENTITY_WEIGHT, device=gt_idx.device),
        corruption_mask.float().to(gt_idx.device) * CORRUPTION_FOCUS_WEIGHT,
    )
    has_extra = extra_weight.any()

    total = torch.tensor(0.0, device=gt_idx.device)
    for s in subsets:
        iout = sum(preds[i] for i in s)
        total = total + ce_dice_loss(iout, gt_idx)
        if has_extra:
            total = total + dice_bce_loss(iout, gt_idx, pixel_weight=extra_weight)
    return total / len(subsets)


@torch.no_grad()
def mean_dice(logits, target_idx):
    pred = logits.argmax(dim=1)
    dices = []
    for c in range(1, logits.shape[1]):
        p_c, t_c = (pred == c).float(), (target_idx == c).float()
        inter, denom = (p_c * t_c).sum(), p_c.sum() + t_c.sum()
        if denom > 0:
            dices.append(((2 * inter + 1e-6) / (denom + 1e-6)).item())
    return float(np.mean(dices)) if dices else float('nan')


def blend_with_current(current_intensity: torch.Tensor, raw_logits: torch.Tensor, alpha: float,
                        num_classes: int = NUM_CLASSES, bonus_scale: float = 3.0,
                        adaptive: bool = False, min_alpha: float = 1e-4) -> torch.Tensor:
    """The mechanism this whole diagnostic arc pointed at: a diffusion-
    style CONTROLLED update, M_{k} = M_{k-1}*(1-alpha) + R_{k-1}*alpha,
    instead of the full overwrite E1 has always done (M_k = R_{k-1},
    full stop, regardless of alpha). Two independent architectural
    upgrades (encoder-only FiLM, then encoder+decoder FiLM) and a real
    corruption-realism fix all left Dice declining monotonically and
    never beating k=0 -- none of them changed this specific thing.

    TWO REAL BUGS FOUND AND FIXED, in sequence, after a real 200-epoch
    run came back with k=1..5 BIT-FOR-BIT IDENTICAL to k=0 (not "no
    improvement" -- the blend literally never fired once, for any pixel,
    in any of the 12 real test cases):

    Attempt 1 (broken): blended in PROBABILITY space (one-hot current +
    softmax(raw_logits), convex combination). Has a hard mathematical
    floor -- a probability can never exceed 1.0, so the current class's
    guaranteed floor mass (1-alpha) beats ANY other class's maximum
    possible mass (alpha*1.0) whenever alpha<0.5, REGARDLESS of how
    confidently the network disagrees. Proven directly with a minimal
    3-class example (100% network confidence still couldn't switch below
    alpha=0.5) before concluding this was the cause.

    Attempt 2 (also broken): tried logit-space blending with the current
    class scaled to match raw_logits' OWN max magnitude at each pixel.
    This just recreates the identical alpha=0.5 floor one level down --
    by construction the anchor equals the network's own best possible
    logit, so the same (1-alpha) vs alpha comparison reappears.

    FIXED: an ADDITIVE bonus at the current class only, sized as
    bonus_scale*(1/alpha - 1) -- grows UNBOUNDED as alpha->0 (guarantees
    exact preservation, no finite raw confidence can ever override it)
    and shrinks to 0 as alpha->1 (guarantees full trust in the new
    prediction). No fixed ceiling anywhere in between -- confirmed
    directly at the ACTUAL broken config (alpha=0.3, the schedule's most
    aggressive value): a realistic confident correction (comparable to
    real network logit magnitudes) now genuinely overrides the prior,
    while a weak/uncertain correction still does not, at every alpha
    tested. bonus_scale=3.0 is a reasonable starting point, not
    calibrated against this specific architecture's actual logit
    distribution. This version achieved a real, first-of-session result:
    k=1..5 all beat k=0 for the first time (0.8469 best, vs 0.8386
    baseline), and stopped the universal monotonic-decline pattern every
    prior mechanism showed.

    adaptive=False (default) reproduces this EXACT validated behavior --
    alpha stays a single global scalar, identical for every pixel. This
    is what produced the 0.8469 result; kept as the default so that
    result stays reproducible.

    adaptive=True makes alpha genuinely PER-PIXEL instead of global:
    modulated by the network's OWN confidence at each location (how
    peaked its softmax distribution is there there), with the schedule's
    own alpha as a ceiling never exceeded -- a pixel where the network is
    very confident about its correction gets close to the full schedule
    alpha; an uncertain pixel gets much less, preserving the current mask
    more heavily there. This is a genuine generalization, not a separate
    implementation -- internally alpha is ALWAYS converted to a per-pixel
    tensor first (constant when adaptive=False), so both modes share one
    code path and cannot silently drift apart.

    min_alpha: numerical floor (not the original design's exact-zero
    special case) -- avoids a literal division by zero, which would
    otherwise produce inf, and 0*inf=nan contaminating the argmax at
    other classes. At min_alpha=1e-4 the resulting bonus (tens of
    thousands, given bonus_scale=3.0) is large enough to guarantee the
    same practical outcome as exact preservation for any realistic
    network logit magnitude -- confirmed by direct comparison against
    the original exact-zero early-return path.

    alpha: reuses severity_for_k's own schedule directly. current_intensity:
    [B,1,H,W] float in [0,1] (class_id/(num_classes-1)). raw_logits:
    [B,num_classes,H,W], the network's raw prediction (R). Returns the
    blended mask in the SAME intensity encoding."""
    current_class = (current_intensity.squeeze(1) * (num_classes - 1)).round().long().clamp(0, num_classes - 1)

    if adaptive:
        raw_probs = F.softmax(raw_logits, dim=1)
        confidence = raw_probs.max(dim=1, keepdim=True).values  # [B,1,H,W], in [1/num_classes, 1.0]
        confidence_normalized = (confidence - 1.0 / num_classes) / (1.0 - 1.0 / num_classes)  # rescale so "totally uncertain" (uniform softmax) -> 0, "fully certain" -> 1
        alpha_map = alpha * confidence_normalized  # per-pixel, NEVER exceeds the schedule's own ceiling -- adaptive can only be more conservative than the fixed schedule, never more aggressive
    else:
        alpha_map = torch.full_like(current_intensity, alpha)  # [B,1,H,W] constant -- makes this branch a genuine special case of the same formula below, not a separately-maintained implementation

    alpha_map = alpha_map.clamp(min=min_alpha, max=1.0)
    bonus = bonus_scale * (1.0 / alpha_map - 1.0)  # [B,1,H,W]
    current_onehot = F.one_hot(current_class, num_classes).permute(0, 3, 1, 2).float()
    adjusted_logits = raw_logits + current_onehot * bonus  # [B,1,H,W] bonus broadcasts against [B,C,H,W] onehot -- network's OWN logit values at every class left untouched, only the current class gets the bonus
    blended_class = adjusted_logits.argmax(dim=1)  # [B,H,W]
    blended_intensity = (blended_class.float() / (num_classes - 1)).unsqueeze(1)
    return blended_intensity


def run_epoch(base_model, repair_model, attn_modules, timestep_embed, film_modules, decoder_film_modules, loader, optimizer, epoch, log_every=10):
    repair_model.backbone.train()
    repair_model.conv.train()
    repair_model.decoder.eval()  # frozen, shared -- BatchNorm must not update running stats
    for h in [repair_model.out_head1, repair_model.out_head2, repair_model.out_head3, repair_model.out_head4]:
        h.eval()
    attn_modules.train()
    timestep_embed.train()
    film_modules.train()
    decoder_film_modules.train()

    total_loss, total_dice, total_k1_dice, total_monotonic, n = 0.0, 0.0, 0.0, 0.0, 0
    for step, batch in enumerate(loader, start=1):
        image = batch['image'].to(DEVICE)
        x = batch['corrupted_input'].to(DEVICE)
        gt_idx = batch['gt_idx'].to(DEVICE)
        corruption_mask = batch['corruption_mask'].to(DEVICE)
        is_identity = batch['is_identity'].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        # Step 1: real dataset sample (synthetic corruption / real-error /
        # identity mix, unchanged from before) -- k=1 encoding concatenated.
        preds = fused_forward(base_model, repair_model, attn_modules, timestep_embed, film_modules, decoder_film_modules, image, x, k=1, k_max=K_UNROLL)
        loss = combined_loss(preds, gt_idx, corruption_mask, is_identity)
        k1_dice = mean_dice(preds[-1].detach(), gt_idx)  # logged separately -- the number test scripts' k=1 actually measures
        preds_final = preds
        current_intensity = x  # running "M_k" for the blend -- what was fed into the most recently completed step, updated after every step below
        step_pred_losses = [ce_dice_loss(preds[-1], gt_idx)] if MONOTONIC_LOSS_WEIGHT > 0 else None  # CLEAN Loss(pred_mask(k), GT) per step -- same formula at every step, deliberately separate from the composite `loss` above (which carries different extra terms at different steps -- corruption-focus/identity at some, nothing at others -- so those composite values are not directly comparable step to step). Skipped entirely when the weight is 0 -- ce_dice_loss is real compute (CE+Dice on preds[-1], not free even though its result would never reach backward()), so the default-off case pays nothing extra, not just "zero gradient contribution".

        # Steps 2..K: MIXED strategy -- addresses exposure bias (the gap
        # flagged directly: training only ever saw clean synthetic
        # corruption, never the model's own real, messier prior output,
        # which is what it actually sees at real inference). With
        # probability SELF_GEN_FRACTION, use the model's OWN live
        # prediction from the previous step as this step's input (real
        # self-generated data, closing the train/inference gap).
        # Otherwise, use Aryan's explicit k-conditioned corruption
        # curriculum (fresh, progressively lighter synthetic damage,
        # k=1 most severe -- see severity_for_k). Both paths still get a
        # real, gradient-tracked loss at every step; self-generated
        # samples use the PLAIN base loss (no corruption-focus extra
        # term -- there's no known ground-truth corruption mask for the
        # model's own real output, confirmed by combined_loss's own
        # has_extra logic reducing correctly to the base loss when
        # corruption_mask is all-zero and is_identity is all-False).
        #
        # COST DISCLOSED: corrupt_mask() runs per-sample on CPU (scipy.ndimage),
        # unlike step 1's corruption which came pre-computed via DataLoader
        # worker parallelism. The synthetic-corruption branch of steps 2..K
        # corrupts live, on the main process, single-threaded -- a real,
        # deliberate performance cost for correctness, not optimized away here.
        gt_np = gt_idx.cpu().numpy()
        no_corruption_mask = torch.zeros_like(corruption_mask)
        no_identity = torch.zeros(gt_idx.shape[0], dtype=torch.bool, device=DEVICE)
        for k in range(2, K_UNROLL + 1):
            severity = severity_for_k(k, K_UNROLL, max_severity=MAX_SEVERITY)  # reused below both as the fresh-corruption severity AND the blend's alpha -- same schedule, same direction (high early, low late), both uses correct
            if np.random.random() < SELF_GEN_FRACTION:
                # BLEND instead of a raw hard overwrite -- the actual
                # mechanism this diagnostic arc was built to test. M_k
                # (what went INTO the previous step) blended with R_{k-1}
                # (what it just predicted), by alpha -- not a full
                # replacement, matching a real diffusion controlled
                # update rather than the full-overwrite E1 has always
                # done. See blend_with_current's docstring.
                intensity_k = blend_with_current(current_intensity, preds_final[-1].detach(), severity, NUM_CLASSES, adaptive=ADAPTIVE_BLEND)
                preds_k = fused_forward(base_model, repair_model, attn_modules, timestep_embed, film_modules, decoder_film_modules, image, intensity_k, k=k, k_max=K_UNROLL)
                loss = loss + combined_loss(preds_k, gt_idx, no_corruption_mask, no_identity)
            else:
                corrupted_np = np.stack([corrupt_mask(g, severity=severity) for g in gt_np])
                corruption_mask_k = torch.from_numpy(corrupted_np != gt_np).to(DEVICE)
                intensity_k = torch.from_numpy(corrupted_np.astype(np.float32) / (NUM_CLASSES - 1)).unsqueeze(1).to(DEVICE)
                preds_k = fused_forward(base_model, repair_model, attn_modules, timestep_embed, film_modules, decoder_film_modules, image, intensity_k, k=k, k_max=K_UNROLL)
                loss = loss + combined_loss(preds_k, gt_idx, corruption_mask_k, no_identity)
            if MONOTONIC_LOSS_WEIGHT > 0:
                step_pred_losses.append(ce_dice_loss(preds_k[-1], gt_idx))  # same clean metric as step 1, appended regardless of which branch ran this step
            preds_final = preds_k
            current_intensity = intensity_k  # update the running M_k regardless of which branch ran -- keeps the next iteration's blend continuity meaningful even if the path switches branches mid-sequence

        # Explicit monotonic-improvement penalty: Loss(pred_1,GT) > Loss(pred_2,GT)
        # > ... > Loss(pred_K,GT). The existing per-step composite loss above
        # never enforced this -- it only minimizes the SUM across steps, which
        # gradient descent can satisfy with a later step regressing slightly as
        # long as the total stays small. This is a zero-margin hinge: for every
        # consecutive pair, penalize step k+1 by however much it exceeds step
        # k's clean loss, zero penalty once k+1 is already <= k. This enforces
        # NON-increasing, not a strict margin -- a tie costs nothing, which in
        # practice never binds since floating-point equality essentially never
        # happens.  DISCLOSED RISK: the hinge only constrains the GAP between
        # consecutive steps, not which side moves -- gradient descent could in
        # principle satisfy it by making an EARLIER step worse rather than a
        # later step better. MONOTONIC_LOSS_WEIGHT is kept modest (0.1 default)
        # for exactly this reason, a nudge alongside the real per-step loss
        # terms above, not a replacement for them; not calibrated against this
        # architecture's actual loss scale, same disclosure as bonus_scale.
        monotonic_penalty = torch.tensor(0.0, device=DEVICE)
        if MONOTONIC_LOSS_WEIGHT > 0:
            for i in range(len(step_pred_losses) - 1):
                monotonic_penalty = monotonic_penalty + F.relu(step_pred_losses[i + 1] - step_pred_losses[i])
            loss = loss + MONOTONIC_LOSS_WEIGHT * monotonic_penalty

        loss = loss / K_UNROLL  # keep the loss scale comparable across different K_UNROLL settings
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            d = mean_dice(preds_final[-1].detach(), gt_idx)  # final (lightest-severity) step's dice, for the headline metric
        total_loss += loss.item(); total_dice += d; total_k1_dice += k1_dice; total_monotonic += monotonic_penalty.item(); n += 1

        if step % log_every == 0 or step == len(loader):
            print(f'Epoch {epoch:03d}  step {step:04d}/{len(loader):04d}  '
                  f'loss={total_loss/n:.4f}  mean_dice(final_step)={total_dice/n:.4f}  '
                  f'mean_dice(step1)={total_k1_dice/n:.4f}  monotonic_penalty={total_monotonic/n:.4f}')

    return total_loss / max(n, 1), total_dice / max(n, 1)



def calculate_metric_percase(pred, gt):
    """Lazy medpy import -- keeps this function safe to call from any
    script (including ones that haven't already checked medpy is
    installed), matching the graceful-skip behavior run_real_eval's own
    call site has always had. Moved to TOP LEVEL (was a nested closure
    inside run_real_eval) so oracle_eval.py and any future script can
    import and reuse it directly, instead of duplicating this logic."""
    from medpy import metric as medpy_metric
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = medpy_metric.binary.dc(pred, gt)
        hd95 = medpy_metric.binary.hd95(pred, gt)
        jaccard = medpy_metric.binary.jc(pred, gt)
    elif pred.sum() > 0 and gt.sum() == 0:
        dice, hd95, jaccard = 1.0, 0.0, 1.0
    else:
        dice, hd95, jaccard = 0.0, 0.0, 0.0
    return dice, hd95, jaccard


def load_raw_test_images(all_dir: Path, patch_size: int = 224):
    """Moved to TOP LEVEL, same reason as calculate_metric_percase above."""
    import glob, h5py
    from scipy.ndimage import zoom
    src_dir = _PROJ / 'synapse' / 'test_vol_h5_new'
    if not src_dir.is_dir():
        print(f'{src_dir} not found -- skipping real eval.')
        return None, None
    test_files = sorted(glob.glob(str(src_dir / '*.h5')))
    if not test_files:
        print(f'No .h5 files in {src_dir} -- skipping real eval.')
        return None, None
    slices, native_sizes = [], []
    for f in test_files:
        with h5py.File(f, 'r') as h:
            vol = h['image'][:]
        for s in range(vol.shape[0]):
            sl = vol[s]
            x, y = sl.shape
            native_sizes.append((x, y))
            if (x, y) != (patch_size, patch_size):
                sl = zoom(sl, (patch_size / x, patch_size / y), order=3)
            slices.append(sl.astype(np.float32))
    return np.stack(slices), np.array(native_sizes)


def upsample_to_native(mask_volume, native_hw):
    """Moved to TOP LEVEL, same reason as calculate_metric_percase above."""
    from scipy.ndimage import zoom
    x, y = native_hw
    _, ph, pw = mask_volume.shape
    if (x, y) == (ph, pw):
        return mask_volume
    return zoom(mask_volume, (1, x / ph, y / pw), order=0)


def load_repair_model_from_checkpoint(base_model: EMCADNet, ckpt_path: Path):
    """Rebuilds the trained repair_model + all its auxiliary modules
    (attn_modules, timestep_embed, film_modules, decoder_film_modules)
    from a saved checkpoint, sharing the frozen decoder/heads with
    base_model exactly as training does. Moved to TOP LEVEL (was inline
    inside run_real_eval) so oracle_eval.py and any future script can
    reuse this exact reconstruction, instead of duplicating it.

    Returns (eval_repair_model, eval_attn_modules, eval_timestep_embed,
    eval_film_modules, eval_decoder_film_modules, k_unroll_trained,
    max_severity_trained, adaptive_blend_trained) -- the last three read
    back from the checkpoint's own disclosed metadata, so callers
    reproduce what this SPECIFIC checkpoint was actually trained with."""
    print(f'\nLoading {ckpt_path}...')
    repair_ckpt = torch.load(ckpt_path, map_location='cpu')
    k_unroll_trained = repair_ckpt.get('k_unroll_trained', K_UNROLL)
    max_severity_trained = repair_ckpt.get('max_severity_trained', MAX_SEVERITY)
    adaptive_blend_trained = repair_ckpt.get('adaptive_blend', False)

    eval_repair_model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER, pretrain=False, activation='relu6')
    eval_repair_model.backbone.load_state_dict(repair_ckpt['backbone'])
    eval_repair_model.conv.load_state_dict(repair_ckpt['conv'])
    eval_repair_model.decoder = base_model.decoder
    eval_repair_model.out_head1 = base_model.out_head1
    eval_repair_model.out_head2 = base_model.out_head2
    eval_repair_model.out_head3 = base_model.out_head3
    eval_repair_model.out_head4 = base_model.out_head4
    eval_repair_model.to(DEVICE).eval()

    eval_attn_modules = nn.ModuleList([CrossScaleAttention(c) for c in STAGE_CHANNELS])
    eval_attn_modules.load_state_dict(repair_ckpt['attn_modules'])
    eval_attn_modules.to(DEVICE).eval()

    eval_timestep_embed = TimestepEmbedding(dim=128)
    eval_timestep_embed.load_state_dict(repair_ckpt['timestep_embed'])
    eval_timestep_embed.to(DEVICE).eval()

    eval_film_modules = nn.ModuleList([FiLMModulation(c, embed_dim=128) for c in STAGE_CHANNELS])
    eval_film_modules.load_state_dict(repair_ckpt['film_modules'])
    eval_film_modules.to(DEVICE).eval()

    eval_decoder_film_modules = nn.ModuleList([FiLMModulation(c, embed_dim=128) for c in DECODER_OUT_CHANNELS])
    eval_decoder_film_modules.load_state_dict(repair_ckpt['decoder_film_modules'])
    eval_decoder_film_modules.to(DEVICE).eval()

    return (eval_repair_model, eval_attn_modules, eval_timestep_embed, eval_film_modules,
            eval_decoder_film_modules, k_unroll_trained, max_severity_trained, adaptive_blend_trained)


def volume_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    """Per-volume Dice, averaged over organs PRESENT in the ground truth.
    Used for validation only (fast, no medpy, 224px). The test report
    keeps its own medpy convention (run_real_eval), so absolute values
    differ slightly between the two -- validation is for choosing an
    epoch and design options, never for the headline number."""
    dices = []
    for c in range(1, num_classes):
        g = (gt == c)
        if g.sum() == 0:
            continue
        p = (pred == c)
        dices.append(2.0 * (p & g).sum() / (p.sum() + g.sum()))
    return float(np.mean(dices)) if dices else float('nan')


def _case_split():
    """Same split precompute_triples uses: the last VAL_HOLDOUT_CASES case
    ids (sorted) are held out from training."""
    src_dir = _PROJ / 'synapse' / 'train_npz_new'
    files = sorted(glob.glob(str(src_dir / '*.npz')))
    assert files, f'no npz files in {src_dir}'
    case_re = re.compile(r'case(\d+)')
    case_ids = [int(case_re.search(Path(f).stem).group(1)) for f in files]
    unique_cases = sorted(set(case_ids))
    val_cases = set(unique_cases[-VAL_HOLDOUT_CASES:]) if VAL_HOLDOUT_CASES > 0 else set()
    train_cases = set(unique_cases) - val_cases
    return files, case_ids, train_cases, val_cases


def load_val_slices(image_size: int = IMG_SIZE):
    """Held-out TRAINING-set cases (never trained on, never part of the
    12-volume test set) as (images [N,224,224] float32, gts [N,224,224]
    uint8, case_ids [N]). Cached. Returns None if VAL_HOLDOUT_CASES=0."""
    files, case_ids, _train_cases, val_cases = _case_split()
    if not val_cases:
        return None
    cache = _PROJ / 'data' / 'synapse_all' / f'mask_repair_val_slices_{VAL_HOLDOUT_CASES}.npz'
    if cache.exists():
        d = np.load(cache)
        return d['image'], d['gt'], d['case_id']
    cache.parent.mkdir(parents=True, exist_ok=True)
    images, gts, cids = [], [], []
    for f, cid in zip(files, case_ids):
        if cid not in val_cases:
            continue
        d = np.load(f)
        img, lab = d['image'], d['label']
        gt = _RAW_TO_9[lab.astype(np.uint8)]
        x, y = img.shape
        if x != image_size or y != image_size:
            img = zoom(img, (image_size / x, image_size / y), order=3)
            gt = zoom(gt, (image_size / x, image_size / y), order=0)
        images.append(img.astype(np.float32)); gts.append(gt.astype(np.uint8)); cids.append(cid)
    np.savez(cache, image=np.stack(images), gt=np.stack(gts), case_id=np.array(cids))
    print(f'load_val_slices: cached {len(images)} slices from cases {sorted(val_cases)} -> {cache}')
    return np.stack(images), np.stack(gts), np.array(cids)


@torch.no_grad()
def sweep_slices(base_model, repair_model, attn_modules, timestep_embed, film_modules, decoder_film_modules,
                 images_224: torch.Tensor, k_max_eval: int, k_unroll: int, max_severity: float,
                 adaptive: bool, batch_size: int = 8):
    """The k-sweep itself, shared by validation and the final test eval so
    the two cannot drift apart. images_224: cpu tensor [N,1,224,224].
    Returns a list of length k_max_eval+1 of integer class maps
    [N,224,224]: entry 0 is the frozen base model, entry k is the mask
    after k repair steps (timestep and blend strength clamped to the
    trained range for k > k_unroll)."""
    preds_224 = []
    for i in range(0, images_224.shape[0], batch_size):
        batch = images_224[i:i + batch_size].to(DEVICE)
        out = base_model(batch, mode='test')
        p1 = out[-1] if isinstance(out, (list, tuple)) else out
        preds_224.append(p1.argmax(dim=1).cpu().numpy())
    current = np.concatenate(preds_224, axis=0)
    outs = [current]
    for k in range(1, k_max_eval + 1):
        k_clamped = min(k, k_unroll)
        alpha = severity_for_k(k_clamped, k_unroll, max_severity=max_severity)
        reps = []
        for i in range(0, current.shape[0], batch_size):
            chunk = current[i:i + batch_size]
            intensity = (chunk.astype(np.float32) / (NUM_CLASSES - 1))[:, None]
            x = torch.from_numpy(intensity).to(DEVICE)
            img = images_224[i:i + batch_size].to(DEVICE)
            preds_k = fused_forward(base_model, repair_model, attn_modules,
                                     timestep_embed, film_modules, decoder_film_modules,
                                     img, x, k=k_clamped, k_max=k_unroll)
            blended_intensity = blend_with_current(x, preds_k[-1].detach(), alpha, NUM_CLASSES, adaptive=adaptive)
            blended_class = (blended_intensity.squeeze(1) * (NUM_CLASSES - 1)).round().long().clamp(0, NUM_CLASSES - 1)
            reps.append(blended_class.cpu().numpy())
        current = np.concatenate(reps, axis=0)
        outs.append(current)
    return outs


@torch.no_grad()
def eval_val(base_model, repair_model, attn_modules, timestep_embed, film_modules, decoder_film_modules,
             val_data, k_max_eval: int = None):
    """Per-case validation Dice at k=0..k_max_eval on the held-out cases,
    using the CURRENT in-memory weights (no checkpoint round trip).
    Returns an array [n_cases, k_max_eval+1]. This is the ONLY signal
    used to pick best.pth and to make design decisions; the test set is
    scored once, at the end."""
    if k_max_eval is None:
        k_max_eval = VAL_K_MAX
    images, gts, case_ids = val_data
    for m in (repair_model, attn_modules, timestep_embed, film_modules, decoder_film_modules):
        m.eval()
    images_t = torch.from_numpy(images).unsqueeze(1)
    cases = sorted(set(case_ids.tolist()))
    scores = np.zeros((len(cases), k_max_eval + 1))
    for ci, cid in enumerate(cases):
        sl = case_ids == cid
        preds = sweep_slices(base_model, repair_model, attn_modules, timestep_embed, film_modules,
                             decoder_film_modules, images_t[sl], k_max_eval, K_UNROLL, MAX_SEVERITY, ADAPTIVE_BLEND)
        for k, p in enumerate(preds):
            scores[ci, k] = volume_dice(p, gts[sl])
    return scores


@torch.no_grad()
def run_real_eval(base_model: EMCADNet, ckpt_path: Path, k_max_eval: Optional[int] = None,
                  results_path: Optional[Path] = None, results_extra: Optional[dict] = None):
    """Real per-case medpy Dice/HD95/Jaccard k-sweep on the actual Synapse
    test set, using the ACTUAL saved best.pth (not whatever's in memory
    at the end of the epoch loop, which may not be the best epoch) --
    same convention as every other train+test-combined script this
    session. Same medpy/native-resolution logic already validated in
    test_mask_repair_synapse_fused.py, ported here rather than
    reimplemented, so training and the real citable result live in one
    script and one log, per the explicit ask.

    k-encoding clamping: this checkpoint was trained with k in
    [1, K_UNROLL]. For k_max_eval > K_UNROLL (sweeping the eval further
    than training went), k gets clamped to K_UNROLL's own encoding value
    -- reusing the "lightest, most-refined" signal the model actually
    learned, not extrapolating to one it never saw (same logic already
    validated in test_mask_repair_synapse_fused.py's repair_step).

    calculate_metric_percase/load_raw_test_images/upsample_to_native/
    load_repair_model_from_checkpoint are now TOP-LEVEL functions (see
    above) so oracle_eval.py and any future script can import and reuse
    them directly, instead of duplicating this logic a second time."""
    if k_max_eval is None:
        k_max_eval = EVAL_K_MAX
    try:
        from medpy import metric as medpy_metric
    except ImportError:
        print('medpy not installed -- skipping real eval. '
              'Run `pip install medpy --break-system-packages` and '
              're-run just the eval separately if needed.')
        return

    all_dir = _PROJ / 'data' / 'synapse_all'
    if not (all_dir / 'mask_test.npy').exists() or not (all_dir / 'case_id_test.npy').exists():
        print(f'{all_dir}/mask_test.npy or case_id_test.npy not found -- skipping real eval.')
        return

    raw_images, native_sizes = load_raw_test_images(all_dir)
    if raw_images is None:
        return
    mask_test = _RAW_TO_9[np.load(all_dir / 'mask_test.npy').astype(np.uint8)]  # raw BTCV ids -> sequential 1-8, same remap used everywhere else in this file (precompute_triples) -- missing this would silently score against the wrong class ids
    case_id_test = np.load(all_dir / 'case_id_test.npy')
    if not (raw_images.shape[0] == mask_test.shape[0] == case_id_test.shape[0]):
        print('Slice count mismatch between raw test images and case_id_test.npy -- skipping real eval.')
        return

    (eval_repair_model, eval_attn_modules, eval_timestep_embed, eval_film_modules, eval_decoder_film_modules,
     k_unroll_trained, max_severity_trained, adaptive_blend_trained) = load_repair_model_from_checkpoint(base_model, ckpt_path)

    images_t = torch.from_numpy(raw_images).unsqueeze(1)
    n_cases = int(case_id_test.max()) + 1
    metric_sum = {k: np.zeros((len(_CLASS_ORDER), 3)) for k in range(k_max_eval + 1)}
    per_case = np.zeros((n_cases, k_max_eval + 1, 3))  # mean over organs of [dice, hd95, jaccard], per case per k -- kept for paired statistics across seeds

    for case_idx in range(n_cases):
        sl = case_id_test == case_idx
        case_images_224 = images_t[sl]  # already 224x224 -- load_raw_test_images resizes internally, native_sizes tracked separately for scoring
        case_gt = mask_test[sl]
        case_native_hw = tuple(native_sizes[sl][0])

        preds_by_k = sweep_slices(base_model, eval_repair_model, eval_attn_modules, eval_timestep_embed,
                                  eval_film_modules, eval_decoder_film_modules, case_images_224,
                                  k_max_eval, k_unroll_trained, max_severity_trained, adaptive_blend_trained)
        for k, pred_224 in enumerate(preds_by_k):
            pred_native = upsample_to_native(pred_224, case_native_hw)
            rows = []
            for c in range(1, NUM_CLASSES):
                dice, hd95, jaccard = calculate_metric_percase(pred_native == c, case_gt == c)
                metric_sum[k][c - 1] += [dice, hd95, jaccard]
                rows.append([dice, hd95, jaccard])
            per_case[case_idx, k] = np.mean(rows, axis=0)
            print(f'case {case_idx}: k={k} dice={per_case[case_idx, k, 0]:.4f}')

    print()
    print('=' * 78)
    print(f'{"k":<4}{"mean_dice":<12}{"mean_hd95":<12}{"mean_jaccard":<14}per-organ dice')
    print('-' * 78)
    for k in range(k_max_eval + 1):
        m = metric_sum[k] / n_cases
        per_organ = '  '.join(f'{name}={m[i,0]:.3f}' for i, name in enumerate(_CLASS_ORDER))
        print(f'{k:<4}{m[:,0].mean():<12.4f}{m[:,1].mean():<12.2f}{m[:,2].mean():<14.4f}{per_organ}')
    print('=' * 78)
    best_k = max(range(k_max_eval + 1), key=lambda k: (metric_sum[k] / n_cases)[:, 0].mean())
    baseline = (metric_sum[0] / n_cases)[:, 0].mean()
    best = (metric_sum[best_k] / n_cases)[:, 0].mean()
    mean_k1 = per_case[:, 1:, :].mean(axis=(0, 1))  # the pre-specified headline: mean over k=1..k_max, not the best k (choosing the best k on test would be selection on test)
    print(f'Mean over k=1..{k_max_eval}: dice={mean_k1[0]:.4f}  hd95={mean_k1[1]:.2f}  jaccard={mean_k1[2]:.4f}  '
          f'(k=0: dice={baseline:.4f}, delta {mean_k1[0]-baseline:+.4f})')
    print(f'Best single k (descriptive only) = {best_k}, mean_dice = {best:.4f} (vs k=0 baseline {baseline:.4f}, delta {best-baseline:+.4f})')
    if best_k == 0:
        print('WARNING: k=0 (no repair) is the best result -- the repair network '
              'is not currently adding value on the real test set.')
    print(f'(checkpoint={ckpt_path}, k_unroll_trained={k_unroll_trained}, '
          f'max_severity_trained={max_severity_trained}, adaptive_blend_trained={adaptive_blend_trained})')

    results = {
        'checkpoint': str(ckpt_path),
        'k_unroll_trained': k_unroll_trained,
        'max_severity_trained': max_severity_trained,
        'adaptive_blend_trained': bool(adaptive_blend_trained),
        'k_max_eval': k_max_eval,
        'per_k': {str(k): {'dice': float((metric_sum[k] / n_cases)[:, 0].mean()),
                            'hd95': float((metric_sum[k] / n_cases)[:, 1].mean()),
                            'jaccard': float((metric_sum[k] / n_cases)[:, 2].mean()),
                            'per_organ_dice': (metric_sum[k] / n_cases)[:, 0].tolist()}
                  for k in range(k_max_eval + 1)},
        'mean_k1_to_kmax': {'dice': float(mean_k1[0]), 'hd95': float(mean_k1[1]), 'jaccard': float(mean_k1[2])},
        'per_case': per_case.tolist(),  # [case][k][dice, hd95, jaccard]
        'class_order': _CLASS_ORDER,
    }
    if results_extra:
        results.update(results_extra)
    if results_path is not None:
        with open(results_path, 'w') as f:
            json.dump(results, f)
        print(f'Test results written to {results_path}')
    return results


def _eval_worker_process(ckpt_dir: str, eval_gpu_idx: str, k_max_eval: int, poll_seconds: int):
    """DEPRECATED, no longer started by main(): this worker scores the TEST
    set every epoch, which is not allowed for design decisions.

    Runs in a SEPARATE, spawned process (not forked -- see main()'s
    comment on why spawn is required for CUDA safety) on its own GPU,
    set via CUDA_VISIBLE_DEVICES INSIDE this function, before any CUDA
    call -- must happen here, in the child, not in the parent, or this
    process would fight the training process for the training GPU
    instead of using its own.

    Polls this run's own epoch_status.json (written every epoch by the
    training loop in the parent process) and evaluates the current
    latest.pth whenever the epoch advances -- same core logic as the
    standalone eval_watcher.py, now inlined so one script launch handles
    both training and concurrent eval, no separate command needed.

    Does NOT queue every historical epoch if it falls behind -- see
    eval_watcher.py's module docstring for why that's deliberate."""
    os.environ['CUDA_VISIBLE_DEVICES'] = eval_gpu_idx
    # Re-import fresh in this process (spawn gives a clean interpreter,
    # not a fork of the parent's already-initialized CUDA state) --
    # build_models() picks up CUDA_VISIBLE_DEVICES as just set above.
    import importlib
    self_module = importlib.import_module(__name__)
    print(f'[eval-worker] Starting on GPU {eval_gpu_idx}, watching {ckpt_dir}')
    base_model, _, _, _, _, _ = self_module.build_models()

    ckpt_dir = Path(ckpt_dir)
    last_evaluated_epoch = -1
    while True:
        status_path = ckpt_dir / 'epoch_status.json'
        if status_path.exists():
            try:
                with open(status_path) as f:
                    status = json.load(f)
                current_epoch = status['epoch']
                if current_epoch > last_evaluated_epoch:
                    latest_ckpt = ckpt_dir / 'latest.pth'
                    if latest_ckpt.exists():
                        print(f'[eval-worker] epoch {current_epoch}/{status["total_epochs"]} '
                              f'(train_dice={status["train_dice"]:.4f}) -- running real eval')
                        try:
                            self_module.run_real_eval(base_model, latest_ckpt, k_max_eval=k_max_eval)
                        except Exception as e:
                            print(f'[eval-worker] !!! eval FAILED at epoch {current_epoch}: {e}')
                        last_evaluated_epoch = current_epoch
            except (json.JSONDecodeError, OSError):
                pass  # status file being written concurrently -- try again next poll
        time.sleep(poll_seconds)


def main():
    # Checkpoint path encodes the sweep parameters -- MAX_SEVERITY and
    # K_UNROLL are both meant to be swept per Aryan's own diffusion-schedule
    # analogy (try 30%/50%/100% severity ceilings, try K=6/10/11 steps).
    # A fixed path here would mean a second sweep value silently overwrites
    # the first run's checkpoint -- exactly the kind of collision that's
    # already cost real time earlier in this project (log files, eval runs).
    # BUG FIXED: this tag originally omitted SELF_GEN_FRACTION -- three
    # simultaneously-launched runs that differed ONLY in that parameter
    # (0.5, 0.0, 1.0, all at MAX_SEVERITY=1.0/K_UNROLL=3) silently wrote
    # to the SAME checkpoint path throughout training, racing against
    # each other. All three of those runs' results were invalid,
    # discovered only because two of them produced bit-for-bit identical
    # final k-sweep tables. Every sweepable hyperparameter must appear
    # here, not just some of them.
    # BUG FIXED AGAIN: ADAPTIVE_BLEND was missing from this tag too --
    # same collision class as the earlier SELF_GEN_FRACTION bug. A real
    # adaptive-blend run collided with the already-validated 0.8469
    # non-adaptive checkpoint (same MAX_SEVERITY/K_UNROLL/SELF_GEN_FRACTION,
    # different ADAPTIVE_BLEND) -- caught via the eval worker immediately
    # reading a stale epoch=200 status file from the OTHER run the moment
    # it started, before this run had even finished its own first epoch.
    # MONOTONIC_LOSS_WEIGHT only enters the tag when it's actually nonzero -- at the
    # default 0.0 the tag is BYTE-IDENTICAL to every checkpoint already produced before
    # this option existed, so already-running or already-finished runs stay matched by
    # run_all.py's own bookkeeping and are never silently retrained. Turning it on
    # deliberately gets its own, distinct directory, same collision-avoidance discipline
    # as SELF_GEN_FRACTION/ADAPTIVE_BLEND/SEED, just conditional so the common case (off)
    # doesn't rename every existing run's directory out from under it.
    _ml_suffix = f'_ml{MONOTONIC_LOSS_WEIGHT:.2f}'.replace('.', 'p') if MONOTONIC_LOSS_WEIGHT != 0 else ''
    sweep_tag = f'sev{MAX_SEVERITY:.2f}_k{K_UNROLL}_sg{SELF_GEN_FRACTION:.2f}_ab{int(ADAPTIVE_BLEND)}{_ml_suffix}_s{SEED}'.replace('.', 'p')
    out_dir = f'RESULTS/mask_repair_synapse_fused_crossattn_unrolled_kaware_{sweep_tag}/checkpoints'
    os.makedirs(out_dir, exist_ok=True)

    print(f'=== Sweep config: MAX_SEVERITY={MAX_SEVERITY}  K_UNROLL={K_UNROLL}  ADAPTIVE_BLEND={ADAPTIVE_BLEND}  MONOTONIC_LOSS_WEIGHT={MONOTONIC_LOSS_WEIGHT} ===')
    print('Severity schedule for this run (k -> corruption severity):')
    for k in range(1, K_UNROLL + 1):
        print(f'  k={k}: severity={severity_for_k(k, K_UNROLL):.3f}')
    print(f'Checkpoints -> {out_dir}')

    # The concurrent eval worker is intentionally NOT started any more. It
    # scored the TEST set after every epoch, which invites picking epochs
    # and design options by looking at test numbers. Model selection now
    # uses held-out validation cases (see eval_val); the test set is
    # scored once, at the very end, on the selected checkpoint.
    if os.environ.get('EVAL_GPU', '').strip():
        print('NOTE: EVAL_GPU is ignored. Per-epoch test evaluation was removed on purpose; '
              'validation on the held-out cases runs inline instead.')

    base_model, repair_model, attn_modules, timestep_embed, film_modules, decoder_film_modules = build_models()

    print('Precomputing (image, gt, base_pred) triples (cached, one-time — '
          'shared with other fused variants if already built)...')
    triples_path = precompute_triples(base_model)

    train_ds = FusedMaskRepairDataset(triples_path)

    _files, _case_ids, _train_cases, _val_cases = _case_split()
    _expected_train = sum(1 for c in _case_ids if c in _train_cases)
    if _expected_train != len(train_ds.gts):
        sys.exit(f'Cached triples hold {len(train_ds.gts)} slices but the current split '
                 f'(VAL_HOLDOUT_CASES={VAL_HOLDOUT_CASES}) implies {_expected_train}. The cache was built with a '
                 f'different split; delete data/synapse_all/mask_repair_fused_triples_train.npz and rerun.')
    val_data = load_val_slices()
    if val_data is None:
        print('WARNING: VAL_HOLDOUT_CASES=0, so there is no validation set. best.pth falls back to TRAIN Dice, '
              'which is a weak signal. Do not make design decisions in this mode.')
    else:
        print(f'Validation: {len(set(val_data[2].tolist()))} held-out cases ({sorted(_val_cases)}), '
              f'{len(val_data[0])} slices, never trained on. Test set is scored once, at the end.')

    # Fixed small sample of real GT masks, drawn once, reused for every
    # periodic health check below -- cheap (no I/O per check), and using
    # the SAME samples every time makes any change in the health-check
    # result attributable to corrupt_mask() itself, not sample variance.
    _health_check_rng = np.random.RandomState(123)
    _health_check_gt_samples = [train_ds.gts[i] for i in
                                 _health_check_rng.choice(len(train_ds.gts), size=min(6, len(train_ds.gts)), replace=False)]
    if HEALTH_CHECK_EVERY_N_EPOCHS > 0:
        print('Running corruption health check once before training starts (catches an obvious regression immediately, not N epochs in):')
        corruption_health_check(_health_check_gt_samples, k_max=K_UNROLL, max_severity=MAX_SEVERITY)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    trainable_params = (list(repair_model.backbone.parameters()) +
                         list(repair_model.conv.parameters()) +
                         list(attn_modules.parameters()) +
                         list(timestep_embed.parameters()) +
                         list(film_modules.parameters()) +
                         list(decoder_film_modules.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)

    best_dice = -1.0          # best TRAIN dice, logged only
    best_select = -1.0        # best selection score (validation, or train dice if no validation set)
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        print('=' * 60)
        loss, dice = run_epoch(base_model, repair_model, attn_modules, timestep_embed, film_modules,
                                decoder_film_modules, train_loader, optimizer, epoch)
        print(f'Epoch {epoch:03d}/{EPOCHS} complete  loss={loss:.4f}  train_mean_dice(p1)={dice:.4f}')

        ckpt = {
            'epoch': epoch,
            'backbone': repair_model.backbone.state_dict(),
            'conv': repair_model.conv.state_dict(),
            'attn_modules': attn_modules.state_dict(),
            'timestep_embed': timestep_embed.state_dict(),
            'film_modules': film_modules.state_dict(),
            'decoder_film_modules': decoder_film_modules.state_dict(),
            'encoder': ENCODER, 'num_classes': NUM_CLASSES,
            'fusion': 'cross_attention',  # disclosed so eval scripts know this checkpoint needs
            'k_unroll_trained': K_UNROLL,  # disclosed -- same architecture as the plain cross-attention checkpoint, different training procedure
            'max_severity_trained': MAX_SEVERITY,  # disclosed -- which point in the sweep produced this specific checkpoint
            'self_gen_fraction_trained': SELF_GEN_FRACTION,  # disclosed -- fraction of steps 2..K that used the model's own live output vs fresh synthetic corruption
            'positional_encoding': True,  # CRITICAL disclosure: this checkpoint carries k via TimestepEmbedding+FiLM (sinusoidal encoding, injected at 4 scales), NOT a raw channel concat -- an eval script expecting the old k_encoding/2ch-conv format will crash or silently mis-load this checkpoint
            'blend_mechanism': True,  # disclosed -- trained with the diffusion-style controlled update (blend_with_current), not a full overwrite at each step. Informational only right now -- run_real_eval already applies the blend unconditionally regardless of this flag, since it's the only mechanism this file supports going forward; kept for clarity when reading old checkpoints later.
            'adaptive_blend': ADAPTIVE_BLEND,  # CRITICAL disclosure -- whether alpha was a global scalar (False, the 0.8469 result) or per-pixel confidence-modulated (True). run_real_eval reads this back from the checkpoint and reproduces the SAME mode the checkpoint was trained with, not whatever ADAPTIVE_BLEND happens to be set to at eval time.
            'monotonic_loss_weight': MONOTONIC_LOSS_WEIGHT,  # disclosed -- 0.0 means this checkpoint was trained with the exact pre-existing loss (no monotonic-decrease penalty); a nonzero value means Loss(pred_k,GT) was explicitly pushed to be non-increasing in k during training. Affects TRAINING only -- run_real_eval and oracle_eval score the checkpoint's actual outputs either way, this key is provenance, not something eval reads back and re-applies.
            'base_checkpoint_path': CKPT_PATH,
        }
        best_dice = max(best_dice, dice)

        # Validation on the held-out cases with the CURRENT weights. This,
        # not train Dice and not the test set, decides best.pth.
        val_score, val_k0 = float('nan'), float('nan')
        if val_data is not None and epoch % VAL_EVERY == 0:
            val_scores = eval_val(base_model, repair_model, attn_modules, timestep_embed, film_modules,
                                  decoder_film_modules, val_data)
            val_k0 = float(val_scores[:, 0].mean())
            val_score = float(val_scores[:, 1:].mean())
            print(f'  VAL (held-out cases, 224px): k0={val_k0:.4f}  ' +
                  '  '.join(f'k{k}={val_scores[:, k].mean():.4f}' for k in range(1, val_scores.shape[1])) +
                  f'  mean(k>=1)={val_score:.4f}')
        ckpt['val_score'] = val_score
        ckpt['seed'] = SEED
        select_score = val_score if val_data is not None else dice
        if np.isfinite(select_score) and select_score > best_select:
            best_select, best_epoch = select_score, epoch
            torch.save(ckpt, f'{out_dir}/best.pth')
            print(f'  New best ({"validation" if val_data is not None else "train dice"} {select_score:.4f}) — checkpoint saved')

        # Per-epoch checkpoint + a small status marker, for the EXTERNAL
        # eval_watcher.py to poll -- separate process, separate GPU,
        # checks this file's 'epoch' field and evaluates 'latest.pth'
        # whenever it advances, rather than eval happening inline here.
        torch.save(ckpt, f'{out_dir}/latest.pth')
        with open(f'{out_dir}/epoch_status.json', 'w') as f:
            json.dump({'epoch': epoch, 'total_epochs': EPOCHS, 'train_dice': dice,
                       'best_train_dice': best_dice, 'val_score': val_score,
                       'best_select_score': best_select, 'best_epoch': best_epoch, 'seed': SEED}, f)

        # Periodic, model-free corruption health check -- catches a
        # regression of the whole-organ-deletion bug found via
        # scene_graph_analysis.py. Cheap (no model, no GPU, ~6 samples x
        # K_UNROLL corrupt_mask() calls), so running it every N epochs
        # costs essentially nothing against a real training epoch.
        if HEALTH_CHECK_EVERY_N_EPOCHS > 0 and epoch % HEALTH_CHECK_EVERY_N_EPOCHS == 0:
            corruption_health_check(_health_check_gt_samples, k_max=K_UNROLL, max_severity=MAX_SEVERITY)

    print('Training complete.')
    print(f'Selected checkpoint: epoch {best_epoch} (selection score {best_select:.4f}, seed {SEED}).')
    if os.environ.get('TEST_AT_END', '1') == '1':
        extra = {'seed': SEED, 'config': {'MAX_SEVERITY': MAX_SEVERITY, 'K_UNROLL': K_UNROLL,
                                          'SELF_GEN_FRACTION': SELF_GEN_FRACTION, 'ADAPTIVE_BLEND': ADAPTIVE_BLEND,
                                          'EPOCHS': EPOCHS, 'LR': LR, 'BATCH_SIZE': BATCH_SIZE}}
        # NOTE: the frozen base model was trained on all 18 Synapse training cases, including the 3
        # held-out validation cases used here, so validation Dice is optimistic for k=0 and a weak
        # signal for choosing an epoch. The FINAL-EPOCH checkpoint needs no selection, so it is the
        # primary result; the validation-selected one is reported alongside as secondary.
        print('Scoring the TEST set on the validation-selected checkpoint (secondary result).')
        run_real_eval(base_model, Path(out_dir) / 'best.pth', results_path=Path(out_dir) / 'test_results.json',
                      results_extra={**extra, 'best_epoch': best_epoch, 'selection_score': best_select,
                                     'selection_signal': 'validation' if val_data is not None else 'train_dice',
                                     'checkpoint_kind': 'best_validation'})
        if os.environ.get('TEST_LAST_EPOCH', '1') == '1':
            print('Scoring the TEST set on the FINAL-EPOCH checkpoint (primary result, no selection involved).')
            run_real_eval(base_model, Path(out_dir) / 'latest.pth', results_path=Path(out_dir) / 'test_results_last.json',
                          results_extra={**extra, 'best_epoch': EPOCHS, 'selection_signal': 'none',
                                         'checkpoint_kind': 'final_epoch'})
    else:
        print('TEST_AT_END=0: test set not scored. Use run_real_eval on best.pth or latest.pth when ready.')

if __name__ == '__main__':
    main()
