# Do CNN-era CL Rankings Generalize to ViTs?

Empirical study comparing continual learning method rankings on a ViT-Small trained
from scratch versus ResNet-18 on class-incremental Split-CIFAR-100.

Course project — Neural Networks and Deep Learning, University of Padova, 2025-2026.

---

## Motivation

Over 60 of 81 continual learning (CL) approaches surveyed in arXiv:2501.04897 validate
exclusively on ResNet-18. Vision Transformers differ structurally in ways that directly
affect how task-specific features are stored: global self-attention from block 1, more
entangled cross-layer representations (Park & Kim, ICLR 2022), and curved representation
geometry (arXiv:2210.05742). Whether EWC and Experience Replay rankings transfer from
ResNets to ViTs is an open and untested question.

**Hypothesis**: Applying standard CNN-era CL methods to a ViT trained from scratch will
reveal rank-ordering differences (or confirm transfer) with non-overlapping confidence
intervals across seeds.

---

## Experimental Setup

| Dimension | Choice |
|---|---|
| Architectures | ViT-Small from scratch (conv stem, 6 blocks, 384-dim), ResNet-18 (`weights=None`), Swin-Tiny (2 stages, patch=4, window=4) |
| CL methods | Vanilla fine-tuning, EWC (online, diagonal Fisher), Experience Replay (reservoir, 500) |
| Benchmark | Class-incremental Split-CIFAR-100: 10 tasks x 10 classes, single 100-class head |
| Seeds | 3 (0, 1, 2) — 27 total runs |
| Metrics | Average Accuracy (AA), Backward Transfer (BWT), Forgetting (AF) |
| Mechanistic | Layer-wise linear CKA similarity + L2 weight drift per layer type |

**ViT-Small spec**: conv stem (3->32->384, stride 2+2, 64 tokens from 32x32 input),
dim=384, 6 heads, 6 blocks, MLP ratio 4x. Tokenizer follows CCT (Hassani et al., 2021).

---

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt   # cu132 nightly — RTX 5070 Ti (Blackwell)
```

Requires PyTorch >= 2.7 with CUDA 13.2. Adjust the `--extra-index-url` in
`requirements.txt` for other CUDA versions.

All scripts run from the `code/` subdirectory:

```bash
cd code
```

---

## Usage

### Smoke tests (run first, ~30 seconds, CPU)

```bash
cd code
python scripts/pilot.py --smoke
python scripts/run_cl.py --smoke
```

Both scripts accept `--smoke` to run 1 epoch over 2 batches. Always verify smoke
passes before launching a full run.

### M1 — Joint-training pilot

```bash
cd code && source ../.venv/bin/activate && python scripts/pilot.py 2>&1 | tee ../logs/pilot.log
```

Trains all three architectures on full CIFAR-100 (joint, 200 epochs). Exits with code 1
and a dataset-fallback message if ViT or Swin top-1 accuracy < 55%.

Writes to `results/pilot/`:
- `vit_best.pt`, `resnet_best.pt`, `swin_best.pt` — best checkpoints
- `{arch}_train.csv` — per-epoch `epoch, train_loss, val_acc`

### M2 — CL grid

```bash
cd code && source ../.venv/bin/activate && python scripts/run_cl.py 2>&1 | tee ../logs/run_cl.log
```

Runs all 27 conditions (3 arch x 3 methods x 3 seeds). Each run writes to
`results/runs/{arch}_{method}_s{seed}/`:
- `ckpt_task{0..9}.pt` — per-task checkpoints (required by M3)
- `train_log.csv` — `task, epoch, train_loss, val_acc`
- `metrics.csv` — final accuracy matrix R and AA / BWT / AF

Skip completed runs by default (checks for `metrics.csv` existence). To force a rerun:
```bash
python scripts/run_cl.py --arch vit --method ewc --seed 1 --force
```

### M3a — Feature extraction

```bash
cd code && source ../.venv/bin/activate && python scripts/extract.py 2>&1 | tee ../logs/extract.log
```

Loads task checkpoints, builds (or reuses) `data/probe_set.pt` (500 val samples per
task, seed=42, clean transform — deterministic across all runs), then writes to
`results/features/{run}/`:
- `cka.npz` — `{layer_name: (n_tasks x n_tasks) float32}`
- `drift.npz` — `{task_id: {layer_name: L2_float}}`

Process a single run: `python scripts/extract.py --run vit_ewc_s0`

### M3b — Plotting

```bash
cd code && python scripts/plot.py
```

Reads only CSV and npz files — no model loading, no GPU. Writes all figures to
`results/figures/`: learning curves, AA/BWT/AF comparison, CKA heatmaps, drift charts.

---

## Results Layout

All training artifacts live under `code/results/`:

```
code/results/
  runs/               <- training artifacts (GPU-hours; never delete)
    pilot/
    {arch}_{method}_s{seed}/
  features/           <- extracted features (re-computable from runs/ in minutes)
    {arch}_{method}_s{seed}/
  figures/            <- plots (re-generated from features/ in seconds)
```

The three-layer separation mirrors a physics data-acquisition / reconstruction /
analysis workflow: `runs/` is raw data, `features/` is reduced data, `figures/` is
output. Only `runs/` is irreplaceable.

---

## Source Layout

All source lives under `code/` (paths relative to `code/`):

```
src/
  models/vit.py       ViT-Small: conv stem + 6 transformer blocks + CLS head
  models/resnet.py    ResNet-18, weights=None, fc -> Linear(512, n_classes)
  models/swin.py      Swin-Tiny: 2 stages, patch=4, window=4, adapted for 32x32
  data/cifar100.py    get_joint_loaders() and get_split_loaders() -- original label space
  cl/base.py          CLMethod ABC: before_task / after_task / loss hooks
  cl/vanilla.py       No-op pass-through
  cl/ewc.py           Online EWC, diagonal Fisher, 20% subsample
  cl/er.py            ReservoirBuffer(500), reservoir sampling, replay per step
  analysis/cka.py     linear_cka() + between_task_cka() with forward hooks
  analysis/drift.py   snapshot() / compute_drift() -- L2 per named parameter
  metrics.py          compute_metrics(R) -> AA, BWT, AF
  trainer.py          Joint training loop (pilot)
  cl_trainer.py       Task-iterator CL loop; CLMethod hooks; CSV logging
configs/default.py    Single @dataclass Config; n_classes derived via __post_init__
```

All hyperparameters live in `configs/default.py`. Scripts override via `argparse`.
No YAML files.

---

## Key Constraints

- PyTorch only — no HuggingFace weights, no pretrained torchvision weights (`weights=None`)
- No Mixup (complicates class-IL label assignment with a single shared head)
- Class-IL protocol: no task-ID at inference; labels in original 0-99 space throughout
- Max 6-page report; 3-person team

---

## Fallback

If ViT joint top-1 < 55% on CIFAR-100, switch dataset to Tiny ImageNet (64x64) per
the PRD fallback. Update `INPUT_SIZE = 64` in `src/models/vit.py` and rerun M1.
The research question is unchanged.
