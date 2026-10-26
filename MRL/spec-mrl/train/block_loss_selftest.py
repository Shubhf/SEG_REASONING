import torch
from .block_loss import block_extra_losses, cross_corr_penalty

torch.manual_seed(0)
N = 4096
prev = torch.randn(N, 8); ind = torch.randn(N, 8); dup = prev.clone()
p_ind, p_dup = cross_corr_penalty(prev, ind).item(), cross_corr_penalty(prev, dup).item()
print(f"orth penalty independent={p_ind:.5f} duplicate={p_dup:.5f} (expect ~0 and ~0.125)")
assert p_ind < 0.01 and abs(p_dup - 0.125) < 0.02

D, B, G = 32, 6, 8
dims, weights = [32, 16, 8, 4, 2], [1.0, 0.3, 0.2, 0.1, 0.1]
q = torch.randn(B, D, requires_grad=True); p = torch.randn(B * G, D, requires_grad=True)
target = torch.arange(B) * G
extra, lb, lo = block_extra_losses(q, p, dims, weights, 0.01, target, 1.0, 0.1)
extra.backward()
print(f"extra={extra.item():.4f} block={lb.item():.4f} orth={lo.item():.4f}")
assert torch.isfinite(extra) and q.grad.abs().sum() > 0 and p.grad.abs().sum() > 0

e0, _, _ = block_extra_losses(torch.randn(B, D), torch.randn(B * G, D), dims, weights, 0.01, target, 0.0, 0.0)
assert e0.item() == 0.0

q3 = torch.randn(B, D, requires_grad=True); p3 = torch.randn(B * G, D, requires_grad=True)
e, _, _ = block_extra_losses(q3, p3, dims, weights, 0.01, target, 1.0, 0.0)
e.backward()
assert q3.grad[:, : dims[-1]].abs().sum() == 0, "base prefix must not get block-loss gradient"
assert q3.grad[:, dims[-1]:].abs().sum() > 0
print("ALL OK")
