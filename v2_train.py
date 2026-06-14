import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import logging
import time
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except Exception:
    HAS_ALBUMENTATIONS = False

try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except Exception:
    HAS_TORCHINFO = False

# Import model components from models/
from models.v2_model import (
    HiReMed,
    ENCODER_CHANNELS,
    ENCODER_PARAMS_M,
    count_parameters_m,
    count_all_parameters_m,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- identity ---
    dataset_name: str = "kvasir"              # local dataset folder under data/
    use_test_as_val: bool = True             # if True, use test split as validation during training
    base_results_dir: str = "RESULTS"

    # --- encoder ---
    # Available encoders (encoder_name, trainable params):
    #   "resnet18"           (14.99M)   — lightweight ResNet, fast training
    #   "resnet34"           (25.10M)   — deeper ResNet-basic block variant
    #   "resnet50"           (61.69M)   — ResNet-bottleneck, strong ImageNet features
    #   "resnet101"          (80.68M)   — deep ResNet-bottleneck
    #   "resnet152"          (96.33M)   — deepest ResNet variant
    #   "convnext_tiny"      (34.67M)   — efficient ConvNeXT, modern architecture
    #   "convnext_small"     (56.31M)   — mid-size ConvNeXT
    #   "convnext_base"      (98.53M)   — strong ConvNeXT, good accuracy/size tradeoff
    #   "convnext_large"    (218.65M)   — largest ConvNeXT, highest capacity
    encoder_name: str = "convnext_tiny"
    encoder_pretrained: bool = True

    # --- data ---
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    # --- training ---
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- precision ---
    use_bf16: bool = False
    disable_cudnn: bool = True
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True

    # --- architecture (latents) ---
    latent_dim_h: int = 256
    latent_dim_z: int = 128

    # --- recursive loop ---
    max_steps: int = 6
    min_steps: int = 2
    K_inner: int = 3

    # --- adaptive halting ---
    halt_threshold: float = 0.82

    # --- loss weights ---
    kl_weight: float = 5e-5
    boundary_weight: float = 0.2
    uncertainty_weight: float = 0.15
    halt_weight: float = 0.3
    deep_supervision_decay: float = 0.7

    # --- evaluation ---
    threshold: float = 0.5
    save_visuals: int = 16
    save_every_epoch: bool = False

    # --- logging ---
    log_step_interval: int = 10
    validate_after_epoch: bool = True

    # --- budget ---
    max_params_millions: float = 200.0


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def configure_runtime(cfg: Config) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.enabled = not cfg.disable_cudnn
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark and (not cfg.disable_cudnn)
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def resolve_run_dir(cfg: Config) -> str:
    """
    RESULTS/<dataset_name>_<encoder>/run_<NNN>
    Auto-increments so each run gets a unique folder.
    """
    base = Path(cfg.base_results_dir) / f"{cfg.dataset_name}_{cfg.encoder_name}"
    base.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
    )
    next_id = (int(existing[-1].name.split("_")[1]) + 1) if existing else 1
    run_dir = str(base / f"run_{next_id:03d}")
    ensure_dir(run_dir)
    return run_dir


def setup_logger(run_dir: str) -> logging.Logger:
    """
    Creates a logger named 'hiremed' that writes to both stdout and train.log.
    Fully notebook-safe: clears all handlers before adding fresh ones.
    """
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    logger = logging.getLogger("hiremed")
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(os.path.join(run_dir, "train.log"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LocalNPYDataset(Dataset):
    """
    Loads images and masks from local .npy files in data/<dataset_name>/.

    Expected files per split:
        data_<split>.npy   → (N, 3, 256, 256), uint8, range [0, 255]
        mask_<split>.npy   → (N, 1, 256, 256), uint8, binary {0, 1}

    Images are normalised to [0, 1].  The encoder (v2+) applies ImageNet
    mean/std internally.
    """

    def __init__(self, data_path: str, mask_path: str, image_size: int, train: bool):
        self.data       = np.load(data_path)
        self.masks      = np.load(mask_path)
        self.image_size = image_size
        self.train      = train

        if HAS_ALBUMENTATIONS:
            if train:
                self.tf = A.Compose([
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.15),
                    A.Rotate(limit=20, p=0.4),
                    A.ColorJitter(
                        brightness=0.12, contrast=0.12,
                        saturation=0.12, hue=0.04, p=0.3,
                    ),
                ])
            else:
                self.tf = A.Compose([A.Resize(image_size, image_size)])
        else:
            self.tf = None

    def __len__(self) -> int:
        return len(self.data)

    def _to_rgb(self, arr) -> np.ndarray:
        if arr.shape[0] == 3 and arr.ndim == 3:
            return arr.transpose(1, 2, 0)
        return arr

    def _mask_to_np(self, mask) -> np.ndarray:
        arr = np.array(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr > 127).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        image = self.data[idx]
        mask  = self.masks[idx]

        image = self._to_rgb(image).astype(np.uint8)
        mask  = self._mask_to_np(mask)

        if self.tf is not None:
            out   = self.tf(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"]
        else:
            image = np.array(
                Image.fromarray(image).resize((self.image_size, self.image_size))
            )
            mask = np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (self.image_size, self.image_size)
                )
            )
            mask = (mask > 127).astype(np.float32)

        image   = image.astype(np.float32) / 255.0
        image_t = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_t  = torch.from_numpy(mask).unsqueeze(0).float()
        return {"image": image_t, "mask": mask_t, "idx": idx}


def build_dataloaders(cfg: Config):
    """
    Build train / val / test DataLoaders from local .npy files.

    If cfg.use_test_as_val is True, the validation loader will use the
    test split (common when val and test share identical samples).
    The test loader always loads from data_test.npy for final evaluation.
    """
    base_dir = Path(__file__).parent / "data" / cfg.dataset_name

    train_data = str(base_dir / "data_train.npy")
    train_mask = str(base_dir / "mask_train.npy")

    if cfg.use_test_as_val:
        val_data   = str(base_dir / "data_test.npy")
        val_mask   = str(base_dir / "mask_test.npy")
    else:
        val_data   = str(base_dir / "data_val.npy")
        val_mask   = str(base_dir / "mask_val.npy")

    test_data = str(base_dir / "data_test.npy")
    test_mask = str(base_dir / "mask_test.npy")

    train_ds = LocalNPYDataset(train_data, train_mask, cfg.image_size, train=True)
    val_ds   = LocalNPYDataset(val_data,   val_mask,   cfg.image_size, train=False)
    test_ds  = LocalNPYDataset(test_data,  test_mask,  cfg.image_size, train=False)

    kw = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * inter + eps) / (union + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return (inter + eps) / (union + eps)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict:
    prob  = torch.sigmoid(logits.float())
    pred  = (prob > threshold).float()
    tgt   = target.float()

    dice = dice_score(pred, tgt).mean().item()
    iou  = iou_score(pred, tgt).mean().item()

    tp = (pred * tgt).sum(dim=(1, 2, 3))
    fp = (pred * (1 - tgt)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * tgt).sum(dim=(1, 2, 3))
    tn = ((1 - pred) * (1 - tgt)).sum(dim=(1, 2, 3))

    precision = ((tp + 1e-6) / (tp + fp + 1e-6)).mean().item()
    recall    = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
    accuracy  = ((tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)).mean().item()

    return {
        "dice": dice, "iou": iou,
        "precision": precision, "recall": recall, "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits_f  = logits.float()
    target_f  = target.float()
    bce       = F.binary_cross_entropy_with_logits(logits_f, target_f)
    soft_dice = 1.0 - dice_score(torch.sigmoid(logits_f), target_f).mean()
    return 0.5 * bce + 0.5 * soft_dice


_SOBEL_X: Optional[torch.Tensor] = None
_SOBEL_Y: Optional[torch.Tensor] = None


def _get_sobel(device, dtype):
    global _SOBEL_X, _SOBEL_Y
    if _SOBEL_X is None or _SOBEL_X.device != device:
        _SOBEL_X = torch.tensor(
            [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
            device=device, dtype=dtype,
        ).view(1, 1, 3, 3)
        _SOBEL_Y = torch.tensor(
            [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
            device=device, dtype=dtype,
        ).view(1, 1, 3, 3)
    return _SOBEL_X.to(device=device, dtype=dtype), _SOBEL_Y.to(device=device, dtype=dtype)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kx, ky = _get_sobel(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return (gx * gx + gy * gy + 1e-6).sqrt().clamp(0, 1)


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits.float())
    return F.binary_cross_entropy(sobel_edges(prob), sobel_edges(target.float()))


def uncertainty_loss(u: torch.Tensor, y_prob: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    error = (y_prob.float().detach() - y_true.float()).abs()
    return F.l1_loss(u.float(), error)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1 + logvar.float() - mu.float().pow(2) - logvar.float().exp())).mean()


def halt_loss(halt_score: torch.Tensor, y_prob: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        q = dice_score((y_prob.float() > 0.5).float(), y_true.float())
    return F.mse_loss(halt_score.float(), q.detach())


def ds_weights(n: int, decay: float) -> List[float]:
    """Exponentially increasing weights, normalised to sum to 1."""
    w = [decay ** (n - 1 - i) for i in range(n)]
    s = sum(w)
    return [x / s for x in w]


def total_loss(outputs: List[Dict], y_true: torch.Tensor, cfg: Config) -> tuple:
    weights = ds_weights(len(outputs), cfg.deep_supervision_decay)
    total   = torch.tensor(0.0, device=y_true.device)
    parts   = {k: 0.0 for k in ["seg", "bnd", "unc", "kl", "halt"]}

    for w, out in zip(weights, outputs):
        l_seg  = seg_loss(out["y_logit"], y_true)
        l_bnd  = boundary_loss(out["y_logit"], y_true)
        l_unc  = uncertainty_loss(out["uncertainty"], out["y_prob"], y_true)
        l_kl   = kl_loss(out["mu"], out["logvar"])
        l_halt = halt_loss(out["halt_score"], out["y_prob"], y_true)

        step_loss = (
            l_seg
            + cfg.boundary_weight    * l_bnd
            + cfg.uncertainty_weight * l_unc
            + cfg.kl_weight          * l_kl
            + cfg.halt_weight        * l_halt
        )
        total          = total + w * step_loss
        parts["seg"]  += w * l_seg.item()
        parts["bnd"]  += w * l_bnd.item()
        parts["unc"]  += w * l_unc.item()
        parts["kl"]   += w * l_kl.item()
        parts["halt"] += w * l_halt.item()

    return total, parts


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = ["dice", "iou", "precision", "recall", "accuracy"]


def _empty_meter() -> Dict:
    return {k: 0.0 for k in METRIC_KEYS}


@torch.no_grad()
def eval_single_pass(
    model: HiReMed,
    loader: DataLoader,
    cfg: Config,
) -> Dict:
    """Runs exactly 1 outer refinement step per sample (force_steps=1)."""
    model.eval()
    meter = _empty_meter()
    n = 0

    for batch in loader:
        images = batch["image"].to(cfg.device, non_blocking=True)
        masks  = batch["mask"].to(cfg.device,  non_blocking=True)

        outputs = model(images, force_steps=1)
        mets    = compute_metrics(outputs[0]["y_logit"], masks, cfg.threshold)

        for k in METRIC_KEYS:
            meter[k] += mets[k]
        n += 1

    for k in meter:
        meter[k] /= max(n, 1)
    return meter


@torch.no_grad()
def eval_adaptive(
    model: HiReMed,
    loader: DataLoader,
    cfg: Config,
) -> Dict:
    """Runs with adaptive halting (min_steps … max_steps)."""
    model.eval()
    meter = _empty_meter()
    meter["avg_steps"] = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(cfg.device, non_blocking=True)
        masks  = batch["mask"].to(cfg.device,  non_blocking=True)

        outputs = model(images, force_steps=None)
        mets    = compute_metrics(outputs[-1]["y_logit"], masks, cfg.threshold)

        for k in METRIC_KEYS:
            meter[k] += mets[k]
        meter["avg_steps"] += len(outputs)
        n += 1

    for k in meter:
        meter[k] /= max(n, 1)
    return meter


def eval_both_modes(
    model: HiReMed,
    loader: DataLoader,
    cfg: Config,
    label: str = "",
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Runs both evaluation modes and prints a side-by-side comparison."""
    sp  = eval_single_pass(loader=loader, model=model, cfg=cfg)
    adp = eval_adaptive(loader=loader,   model=model, cfg=cfg)

    combined = {"single_pass": sp, "adaptive": adp}

    tag = f"  [{label}]" if label else ""
    lines = [
        f"\n{'─'*62}{tag}",
        f"  {'Metric':<14}  {'1-pass':>10}  {'Adaptive':>10}  {'Δ':>8}",
        f"  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*8}",
    ]
    for k in METRIC_KEYS:
        delta = adp[k] - sp[k]
        sign  = "+" if delta >= 0 else ""
        lines.append(f"  {k:<14}  {sp[k]:>10.4f}  {adp[k]:>10.4f}  {sign}{delta:>7.4f}")
    lines.append(f"  {'avg_steps':<14}  {'1':>10}  {adp['avg_steps']:>10.2f}  {'':>8}")
    lines.append(f"{'─'*62}")

    block = "\n".join(lines)
    if logger:
        logger.info(block)
    else:
        print(block)

    return combined


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_train_epoch(
    model: HiReMed,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    epoch: int,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Full training pass for one epoch with step-level logging."""
    meter = {k: 0.0 for k in [
        "loss", "seg", "bnd", "unc", "kl", "halt",
        "dice", "iou", "precision", "recall", "accuracy",
    ]}
    window = {k: 0.0 for k in meter}
    window_n = 0

    model.train()
    n = 0
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(cfg.device, non_blocking=True)
        masks  = batch["mask"].to(cfg.device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs     = model(images)
        loss, parts = total_loss(outputs, masks, cfg)
        mets        = compute_metrics(outputs[-1]["y_logit"], masks, cfg.threshold)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        meter["loss"] += loss.item()
        for k in parts:
            meter[k] += parts[k]
        for k in mets:
            meter[k] += mets[k]
        n += 1

        window["loss"] += loss.item()
        for k in parts:
            window[k] += parts[k]
        for k in mets:
            window[k] += mets[k]
        window_n += 1

        if batch_idx % cfg.log_step_interval == 0 or batch_idx == total_batches:
            w_loss = window["loss"] / max(window_n, 1)
            w_dice = window["dice"] / max(window_n, 1)
            w_iou  = window["iou"]  / max(window_n, 1)
            w_seg  = window["seg"]  / max(window_n, 1)
            w_bnd  = window["bnd"]  / max(window_n, 1)
            w_unc  = window["unc"]  / max(window_n, 1)
            w_kl   = window["kl"]   / max(window_n, 1)
            w_halt = window["halt"] / max(window_n, 1)
            msg = (
                f"Epoch {epoch:03d}  step {batch_idx:04d}/{total_batches:04d}  "
                f"loss={w_loss:.4f}  dice={w_dice:.4f}  iou={w_iou:.4f}  "
                f"[seg={w_seg:.4f}  bnd={w_bnd:.4f}  unc={w_unc:.4f}  "
                f"kl={w_kl:.6f}  halt={w_halt:.4f}]"
            )
            if logger:
                logger.info(msg)
            else:
                print(msg)
            for k in window:
                window[k] = 0.0
            window_n = 0

    for k in meter:
        meter[k] /= max(n, 1)
    return meter


# ---------------------------------------------------------------------------
# Visual output
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_predictions(
    model: HiReMed,
    loader: DataLoader,
    cfg: Config,
    out_dir: str,
    max_samples: int = 16,
    mode: str = "adaptive",
) -> None:
    """Saves composite images: input | ground truth | prediction | uncertainty."""
    ensure_dir(out_dir)
    model.eval()
    saved = 0

    for batch in loader:
        images = batch["image"].to(cfg.device)
        masks  = batch["mask"].to(cfg.device)

        force = 1 if mode == "single_pass" else None
        outputs = model(images, force_steps=force)

        final   = outputs[-1]
        pred    = (final["y_prob"].float() > cfg.threshold).float()
        unc     = final["uncertainty"].float()
        n_steps = len(outputs)

        for i in range(images.size(0)):
            if saved >= max_samples:
                return

            img_np = (images[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt_np  = (masks[i].cpu().repeat(3, 1, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pr_np  = (pred[i].cpu().repeat(3, 1, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            unc_np = (unc[i].cpu().repeat(3, 1, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            canvas = np.concatenate([img_np, gt_np, pr_np, unc_np], axis=1)
            fname  = f"sample_{saved:03d}_{mode}_steps{n_steps}.png"
            Image.fromarray(canvas).save(os.path.join(out_dir, fname))
            saved += 1


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def print_model_summary(
    model: HiReMed,
    cfg: Config,
    save_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Prints a torchinfo summary with per-layer details. Falls back to manual table."""
    dummy = torch.zeros(
        1, 3, cfg.image_size, cfg.image_size,
        device=cfg.device,
    )

    lines: List[str] = []

    if HAS_TORCHINFO:
        stats = torchinfo_summary(
            model,
            input_data=dummy,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            col_width=22,
            depth=4,
            verbose=0,
            row_settings=["var_names"],
        )
        lines.append(str(stats))
    else:
        lines.append("torchinfo not installed — showing per-module param table")
        lines.append(f"{'Module':<55} {'Params':>12}  {'Trainable':>10}")
        lines.append("─" * 80)
        total_p = 0
        for name, mod in model.named_modules():
            own_params = sum(
                p.numel() for p in mod.parameters(recurse=False) if p.requires_grad
            )
            if own_params > 0:
                lines.append(f"  {name:<53} {own_params:>12,}  {'yes':>10}")
                total_p += own_params
        lines.append("─" * 80)
        lines.append(f"  {'Total trainable':<53} {total_p:>12,}")

    lines.append("")
    lines.append(f"Encoder : {model.encoder_name} (pretrained={model.encoder_pretrained})")
    lines.append(f"Encoder channels (s1, s2, s3, s4) : ({model.s1_ch}, {model.s2_ch}, {model.s3_ch}, {model.s4_ch})")
    lines.append("")
    lines.append("Per-submodule parameter breakdown:")
    lines.append(f"  {'Submodule':<30} {'Params (M)':>12}  {'% of total':>10}  {'Trainable':>10}")
    lines.append("  " + "─" * 70)
    total_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for attr in ["encoder", "z_proj", "low", "high", "decoder", "unc_head", "halt_head"]:
        submod = getattr(model, attr, None)
        if submod is None:
            continue
        p = sum(x.numel() for x in submod.parameters() if x.requires_grad)
        lines.append(f"  {attr:<30} {p/1e6:>12.3f}  {100*p/total_all:>9.1f}%  {'yes':>10}")
    lines.append("  " + "─" * 70)
    lines.append(f"  {'TOTAL trainable':<30} {total_all/1e6:>12.3f}  {'100.0':>9}%")

    # Also show frozen params if any
    total_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if total_frozen > 0:
        lines.append(f"  {'TOTAL frozen':<30} {total_frozen/1e6:>12.3f}")

    lines.append("")

    output = "\n".join(lines)
    if logger:
        logger.info(output)
    else:
        print(output)

    if save_path:
        with open(save_path, "w") as f:
            f.write(output)
        msg = f"  Model summary saved → {save_path}"
        if logger:
            logger.info(msg)
        else:
            print(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = Config()
    configure_runtime(cfg)
    set_seed(cfg.seed)

    # Validate encoder choice
    if cfg.encoder_name not in ENCODER_CHANNELS:
        raise ValueError(
            f"Unknown encoder '{cfg.encoder_name}'. "
            f"Choose from: {list(ENCODER_CHANNELS.keys())}"
        )

    run_dir  = resolve_run_dir(cfg)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    vis_dir  = os.path.join(run_dir, "visuals")
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    logger = setup_logger(run_dir)

    save_json(os.path.join(run_dir, "config.json"), asdict(cfg))
    logger.info(f"Run directory : {run_dir}")
    logger.info(f"Config : {json.dumps(asdict(cfg), indent=2)}")

    # --- data ---
    logger.info(f"Loading dataset: {cfg.hf_dataset_name}")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    logger.info(
        f"Dataset loaded — train={len(train_loader.dataset)}  "
        f"val={len(val_loader.dataset)}  test={len(test_loader.dataset)}"
    )

    # --- model ---
    logger.info(f"Building model with encoder: {cfg.encoder_name} (pretrained={cfg.encoder_pretrained})")
    model = HiReMed(
        encoder_name=cfg.encoder_name,
        encoder_pretrained=cfg.encoder_pretrained,
        latent_dim_h=cfg.latent_dim_h,
        latent_dim_z=cfg.latent_dim_z,
        max_steps=cfg.max_steps,
        min_steps=cfg.min_steps,
        K_inner=cfg.K_inner,
        halt_threshold=cfg.halt_threshold,
    ).to(cfg.device)

    params_m = count_parameters_m(model)
    all_params_m = count_all_parameters_m(model)
    logger.info(f"  Encoder          : {cfg.encoder_name} ({ENCODER_PARAMS_M.get(cfg.encoder_name, '?'):.1f}M base)")
    logger.info(f"  Trainable params : {params_m:.2f}M")
    logger.info(f"  Total params     : {all_params_m:.2f}M")
    logger.info(f"  Encoder channels : s1={model.s1_ch}, s2={model.s2_ch}, s3={model.s3_ch}, s4={model.s4_ch}")
    logger.info(f"  cuDNN enabled    : {torch.backends.cudnn.enabled}")
    logger.info(f"  Device           : {cfg.device}")

    summary_path = os.path.join(run_dir, "model_summary.txt")
    print_model_summary(model, cfg, save_path=summary_path, logger=logger)

    if params_m > cfg.max_params_millions:
        raise ValueError(
            f"Model has {params_m:.2f}M trainable params, exceeds budget {cfg.max_params_millions:.2f}M"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_dice = -1.0
    history: List[Dict] = []

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    logger.info(f"Starting training — {cfg.epochs} epochs, log every {cfg.log_step_interval} steps")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        logger.info(f"{'='*62}")
        logger.info(f"Epoch {epoch:03d}/{cfg.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_m = run_train_epoch(
            model, train_loader, optimizer, cfg,
            epoch=epoch, logger=logger,
        )

        if cfg.validate_after_epoch:
            val_both = eval_both_modes(
                model, val_loader, cfg,
                label=f"val epoch {epoch}",
                logger=logger,
            )
        else:
            val_both = eval_both_modes(
                model, val_loader, cfg,
                label=f"val epoch {epoch}",
                logger=logger,
            )

        scheduler.step()
        dt = time.time() - t0

        val_dice_adaptive   = val_both["adaptive"]["dice"]
        val_dice_singlepass = val_both["single_pass"]["dice"]
        val_avg_steps       = val_both["adaptive"]["avg_steps"]

        row = {
            "epoch":   epoch,
            "lr":      optimizer.param_groups[0]["lr"],
            "time_s":  round(dt, 1),
            "train":   train_m,
            "val":     val_both,
        }
        history.append(row)
        save_json(os.path.join(run_dir, "history.json"), history)

        epoch_summary = (
            f"Epoch {epoch:03d}/{cfg.epochs} complete  "
            f"train_loss={train_m['loss']:.4f}  train_dice={train_m['dice']:.4f}  "
            f"val_dice(1-pass)={val_dice_singlepass:.4f}  "
            f"val_dice(adaptive)={val_dice_adaptive:.4f}  "
            f"avg_steps={val_avg_steps:.2f}  "
            f"time={dt:.1f}s"
        )
        logger.info(epoch_summary)

        ckpt = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "config":        asdict(cfg),
        }

        if cfg.save_every_epoch:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

        if val_dice_adaptive > best_val_dice:
            best_val_dice         = val_dice_adaptive
            ckpt["best_val_dice"] = best_val_dice
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            logger.info(f"  ↑ New best  val_dice(adaptive)={best_val_dice:.4f}  checkpoint saved")

    # -----------------------------------------------------------------------
    # Final evaluation on best checkpoint — val and test, both modes
    # -----------------------------------------------------------------------
    logger.info("\nLoading best model for final evaluation...")
    best_ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=cfg.device)
    model.load_state_dict(best_ckpt["model"])

    logger.info("\n=== Final evaluation: validation split ===")
    val_final  = eval_both_modes(model, val_loader, cfg, label="final val", logger=logger)

    logger.info("\n=== Final evaluation: test split ===")
    test_final = eval_both_modes(model, test_loader, cfg, label="final test", logger=logger)

    results = {
        "encoder":                  cfg.encoder_name,
        "encoder_pretrained":       cfg.encoder_pretrained,
        "best_val_dice_adaptive":   best_val_dice,
        "final_val":                val_final,
        "final_test":               test_final,
        "params_millions":          params_m,
        "run_dir":                  run_dir,
    }
    save_json(os.path.join(run_dir, "final_results.json"), results)
    logger.info(f"final_results.json saved → {run_dir}")

    # -----------------------------------------------------------------------
    # Visuals — save both modes side by side
    # -----------------------------------------------------------------------
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        save_predictions(
            model, loader, cfg,
            out_dir=os.path.join(vis_dir, split_name, "adaptive"),
            max_samples=cfg.save_visuals,
            mode="adaptive",
        )
        save_predictions(
            model, loader, cfg,
            out_dir=os.path.join(vis_dir, split_name, "single_pass"),
            max_samples=cfg.save_visuals,
            mode="single_pass",
        )

    logger.info(f"Done. All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
