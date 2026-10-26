# MRL vs SpecMRL with block loss + orthogonality (Qwen3-0.6B-Base)

Code base: https://github.com/CentreChen/spec-mrl (commit 59db245). Our changes: `spec-mrl/train/block_loss.py`
(new loss), hooks in `train/modeling.py` and `train/modeling_rawMRL.py`, switched on with
`BLOCK_COEF=1.0 ORTH_LAMBDA=0.1` (both 0 = baseline). `loss_changes.patch` has the diff.

15-task MTEB subset (paper Table 14), one seed, 1 GPU. Average score by embedding dimension.

## Batch 384, 100 steps
| dims | MRL | MRL+loss | SpecMRL | SpecMRL+loss |
|---|---|---|---|---|
| 1024 | 59.80 | 60.44 | 60.04 | 60.43 |
| 512 | 58.62 | 59.21 | 59.91 | 59.84 |
| 256 | 56.21 | 57.02 | 59.05 | 59.29 |
| 128 | 52.66 | 53.34 | 57.38 | 57.61 |
| 64 | 46.64 | 47.19 | 54.44 | 54.43 |

## Batch 8, 5000 steps
| dims | MRL | MRL+loss | SpecMRL | SpecMRL+loss |
|---|---|---|---|---|
| 1024 | 57.29 | 58.40 | 56.66 | 57.36 |
| 512 | 56.49 | 57.65 | 55.95 | 57.05 |
| 256 | 54.93 | 55.87 | 55.41 | 56.19 |
| 128 | 51.73 | 52.68 | 54.13 | 54.84 |
| 64 | 46.67 | 47.21 | 51.68 | 52.04 |

Not comparable to the paper's 56-task table (different tasks, far less training). Single seed: noise not measured.
