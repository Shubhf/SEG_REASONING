import torch
import torch.nn.functional as F


def _info_nce(q, p, target, temperature):
    scores = torch.matmul(F.normalize(q, p=2, dim=-1), F.normalize(p, p=2, dim=-1).transpose(0, 1)) / temperature
    return F.cross_entropy(scores.view(q.size(0), -1), target)


def cross_corr_penalty(prev, new, eps=1e-6):
    """Mean squared cross-correlation between columns of `prev` [N, d1] and `new` [N, d2]."""
    prev = prev.float()
    new = new.float()
    prev = (prev - prev.mean(0, keepdim=True)) / (prev.std(0, keepdim=True) + eps)
    new = (new - new.mean(0, keepdim=True)) / (new.std(0, keepdim=True) + eps)
    with torch.amp.autocast('cuda', enabled=False):  # keep the correlation matmul in fp32
        corr = torch.matmul(prev.transpose(0, 1), new) / prev.size(0)  # [d1, d2]
    return corr.pow(2).mean()


def block_extra_losses(q_proj, p_proj, dims, weights, temperature, target, block_coef, orth_lambda):
    """Returns (extra_loss, block_loss_detached, orth_loss_detached). dims descending."""
    zero = q_proj.new_zeros((), dtype=torch.float32)
    l_block, l_orth = zero, zero
    feats = torch.cat([q_proj, p_proj], dim=0) if orth_lambda > 0 else None
    for i in range(len(dims) - 1):
        lo, hi = dims[i + 1], dims[i]
        if block_coef > 0:
            l_block = l_block + weights[i] * _info_nce(
                q_proj[:, lo:hi], p_proj[:, lo:hi], target, temperature
            )
        if orth_lambda > 0:
            l_orth = l_orth + cross_corr_penalty(feats[:, :lo], feats[:, lo:hi])
    extra = block_coef * l_block + orth_lambda * l_orth
    return extra, l_block.detach(), l_orth.detach()
