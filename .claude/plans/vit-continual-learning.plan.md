# Plan: ViT vs ResNet-18 under Continual Learning

**Source PRD**: `.claude/prds/vit-continual-learning.prd.md`
**Selected Milestone**: M1 complete; M2 next
**Complexity**: Large

## Summary

Build the full project scaffold, data pipeline, both model architectures, and a joint-training pilot run. The pilot determines whether ViT-Small from scratch hits >= 55% on full CIFAR-100 — a hard gate before any CL experiments begin. This plan covers all four milestones but scopes implementation tasks per milestone so they can be confirmed and executed sequentially.

## Patterns to Mirror

Conventions established by this plan and applied consistently.

| Category | Source | Pattern |
|---|---|---|
| Naming | —new— | `snake_case` modules, `CamelCase` classes, `ALL_CAPS` constants |
| Errors | —new— | raise `ValueError` / `RuntimeError` with explicit messages; no silent fallbacks |
| Logging | —new— | CSV via `csv.DictWriter` per task/epoch; print to stdout with `tqdm` |
| Data access | —new— | All dataset construction through `src/data/cifar100.py`; plain `DataLoader` pairs |
| Tests | —new— | Every script accepts `--smoke` (1 epoch, 2 batches) for fast sanity checks |

---

## Data / Analysis Separation

Training and analysis are split into three independent phases, mirroring a physics
data-acquisition / reconstruction / analysis workflow:

| Phase | Scripts | GPU? | Produces | Safe to delete and redo? |
|---|---|---|---|---|
| Training (data) | `pilot.py`, `run_cl.py` | yes, hours | `results/runs/` | No — re-running costs GPU hours |
| Feature extraction | `scripts/extract.py` | helpful, minutes | `results/features/` | Yes — derived from runs/ |
| Plotting | `scripts/plot.py` | no, seconds | `results/figures/` | Yes — derived from features/ |

Rule: **no analysis or plotting code inside training scripts**. The trainer writes
checkpoints and CSVs; everything else is post-hoc.

---

## Directory Structure

```
project/
├── src/
│   ├── models/
│   │   ├── vit.py          # ViT-Small from scratch (conv stem)
│   │   └── resnet.py       # ResNet-18 via torchvision (no pretrained)
│   ├── data/
│   │   └── cifar100.py     # Joint + Split-CIFAR-100 loaders
│   ├── cl/
│   │   ├── base.py         # CLMethod ABC
│   │   ├── vanilla.py      # No-op pass-through
│   │   ├── ewc.py          # Online EWC, diagonal Fisher, 20% subsample
│   │   └── er.py           # ReservoirBuffer(500), replay per step
│   ├── analysis/
│   │   ├── cka.py          # linear_cka() + between_task_cka()
│   │   └── drift.py        # snapshot() / compute_drift() — L2 per layer
│   ├── metrics.py          # compute_metrics(R) -> AA, BWT, AF
│   ├── trainer.py          # Joint training loop (pilot)
│   └── cl_trainer.py       # Task-iterator CL loop; writes metrics.csv + train_log.csv
├── configs/
│   └── default.py          # @dataclass Config; n_classes derived via __post_init__
├── scripts/
│   ├── pilot.py            # M1: joint train both archs; 55% gate
│   ├── run_cl.py           # M2: 2 arch x 3 methods x 3 seeds grid
│   ├── extract.py          # M3a: checkpoints -> cka.npz + drift.npz per run
│   └── plot.py             # M3b: CSVs + npz -> all figures (no model loading)
├── data/
│   └── probe_set.pt        # Fixed probe set (seed=42); created once by extract.py
├── results/
│   ├── runs/               # Training artifacts — append-only, never delete
│   │   ├── pilot/
│   │   │   ├── vit_best.pt
│   │   │   ├── vit_train.csv        # epoch, train_loss, val_acc
│   │   │   ├── resnet_best.pt
│   │   │   └── resnet_train.csv
│   │   └── {arch}_{method}_s{seed}/
│   │       ├── train_log.csv        # task, epoch, train_loss, val_acc
│   │       ├── metrics.csv          # final R matrix + AA, BWT, AF
│   │       └── ckpt_task{t}.pt      # t in 0..9
│   ├── features/           # Extracted features — re-computable, safe to delete
│   │   └── {arch}_{method}_s{seed}/
│   │       ├── cka.npz              # layer_name -> (n_tasks x n_tasks) float32
│   │       └── drift.npz            # task_id -> {layer_name: L2_float}
│   └── figures/            # Plots — instant re-generation from features/
├── requirements.txt
└── README.md
```

**Config system**: `configs/default.py` is the single source of truth. `n_classes`
is derived from `n_tasks * classes_per_task` in `__post_init__` — never set manually.
Scripts import `Config` and override fields via `argparse`.

---

## Key Decisions

| Question | Decision | Rationale |
|---|---|---|
| Config system | `dataclass` + argparse + `__post_init__` | No extra dep; derived fields stay consistent |
| CL head | Single 100-class linear head from task 1 | Standard class-IL; no task-ID at inference |
| EWC Fisher | 20% subsample of task train set | Stable estimates; ~5x faster |
| ER buffer | 500 total exemplars, reservoir sampling | ~5/class after 10 tasks |
| CKA probe set | `data/probe_set.pt` — 500 samples per task, val split, seed=42 | Deterministic across all re-runs; val_transform only (no augmentation) |
| Training logs | Per-epoch CSV (trainer) + per-(task,epoch) CSV (cl_trainer) | Re-plot learning curves without retraining |
| Extract / plot split | `extract.py` (heavy) + `plot.py` (instant) | Can re-style figures without re-loading checkpoints |
| Drift computation | Post-hoc in `extract.py` from saved ckpts | L2 on weights is pure numpy; no reason to couple it to the training loop |
| Augmentation | RandAugment(n=2,m=9) + Normalize; no Mixup | Standard CIFAR-100 from scratch; Mixup complicates class-IL labels |

---

## Files to Change

### M1 — Infrastructure + Pilot (DONE)

| File | Status | Notes |
|---|---|---|
| `src/__init__.py` | done | |
| `src/models/__init__.py` | done | |
| `src/data/__init__.py` | done | |
| `configs/default.py` | done | `momentum` field added; `n_classes` via `__post_init__` |
| `src/data/cifar100.py` | done | |
| `src/models/vit.py` | done | `N_PATCHES` computed from `INPUT_SIZE // STEM_STRIDE` |
| `src/models/resnet.py` | done | |
| `src/trainer.py` | done | `out_dir` param; per-epoch CSV log; `cfg.momentum` |
| `scripts/pilot.py` | done | `--num_workers`; explicit `out_dir`; gate check |

### M2 — CL Training

| File | Action | Why |
|---|---|---|
| `src/cl/base.py` | CREATE | `CLMethod` ABC: `before_task`, `after_task`, `loss` hooks |
| `src/cl/vanilla.py` | CREATE | Pass-through; no regularization |
| `src/cl/ewc.py` | CREATE | Online EWC; Fisher via 20%-subsample diagonal |
| `src/cl/er.py` | CREATE | Reservoir buffer; replay batch mixed into each training step |
| `src/metrics.py` | CREATE | `compute_metrics(R)` -> AA, BWT, AF |
| `src/cl_trainer.py` | CREATE | Task-iterator loop; CLMethod hooks; writes `train_log.csv` + `metrics.csv`; saves `ckpt_task{t}.pt`; no CKA/drift logic |
| `scripts/run_cl.py` | CREATE | Grid: 2 arch x 3 methods x seeds; writes to `results/runs/{run}/`; `--smoke` |

### M3 — Mechanistic Analysis

| File | Action | Why |
|---|---|---|
| `src/analysis/cka.py` | CREATE | `linear_cka(X, Y)` + `between_task_cka(model, probe_set, layer_names)` |
| `src/analysis/drift.py` | CREATE | `snapshot(model)` + `compute_drift(model, ref)` -> L2 per named param |
| `scripts/extract.py` | CREATE | Load `ckpt_task{t}.pt` for each run; build/reuse `data/probe_set.pt` (seed=42, val split, clean transform); write `results/features/{run}/cka.npz` + `drift.npz` |
| `scripts/plot.py` | CREATE | Read `results/runs/*/metrics.csv`, `train_log.csv`, `results/features/*/cka.npz`, `drift.npz`; write all figures to `results/figures/`; zero model loading |

---

## Tasks

### M1 · Task 1: Config dataclass (DONE)
`configs/default.py` — `@dataclass Config` with all hyperparams. `n_classes`
derived via `__post_init__` from `n_tasks * classes_per_task`. `momentum=0.9` field.

### M1 · Task 2: CIFAR-100 data pipeline (DONE)
`src/data/cifar100.py` — `get_joint_loaders(cfg)` and `get_split_loaders(cfg)`.
Labels stay in original 0-99 space. Val split uses clean transform (no augmentation).

### M1 · Task 3: ResNet-18 (DONE)
`src/models/resnet.py` — `get_resnet18(n_classes=100)`.

### M1 · Task 4: ViT-Small from scratch (DONE)
`src/models/vit.py` — `N_PATCHES = (INPUT_SIZE // STEM_STRIDE) ** 2 = 64`.
Conv stem, 6 transformer blocks, CLS head. `get_vit_small(n_classes=100)`.

### M1 · Task 5: Joint trainer (DONE)
`src/trainer.py` — `fit(model, train_loader, val_loader, cfg, arch_name, out_dir, smoke)`.
Writes `{arch}_train.csv` (epoch, train_loss, val_acc) and `{arch}_best.pt` to `out_dir`.
Uses `cfg.momentum`. Does not write in smoke mode.

### M1 · Task 6: Pilot script (DONE)
`scripts/pilot.py` — `--num_workers` CLI arg. Passes explicit `out_dir` to `fit()`.
Exits 1 if ViT < 55%.

---

### M2 · Task 7: CL method base + vanilla
`src/cl/base.py` — `CLMethod(ABC)`: `before_task(task_id, train_loader, model)`,
`after_task(task_id, train_loader, model)`, `loss(logits, targets, model) -> Tensor`.
`src/cl/vanilla.py` — no-op implementation.

### M2 · Task 8: EWC
`src/cl/ewc.py` — diagonal Fisher on 20% subsample after each task. Penalty:
`lambda/2 * sum_i F_i * (theta - theta*_i)^2`. Accumulated online. `ewc_lambda` from config.

### M2 · Task 9: ER
`src/cl/er.py` — `ReservoirBuffer(max_size=500)` with `update(x, y)` and
`sample(n) -> (x, y)`. Replay batch concatenated with current batch before each forward.

### M2 · Task 10: Metrics + CL trainer
`src/metrics.py` — `compute_metrics(R: np.ndarray) -> dict` where `R[i,j]` is accuracy
on task j after training task i. Returns AA, BWT, AF.

`src/cl_trainer.py` — `run_cl(model, splits, method, cfg, run_dir) -> R`:
- Iterates 10 tasks; calls `method.before_task`, step loop with `method.loss`, `method.after_task`
- Saves `ckpt_task{t}.pt` after each task (needed by extract.py)
- Writes `train_log.csv` (task, epoch, train_loss, val_acc) per epoch
- Writes `metrics.csv` (final R matrix + AA, BWT, AF) after all tasks
- No CKA or drift logic — analysis is fully post-hoc

### M2 · Task 11: Run script
`scripts/run_cl.py` — grid over `arch in {vit, resnet}`, `method in {vanilla, ewc, er}`,
`seed in {0, 1, 2}`. Each run writes to `results/runs/{arch}_{method}_s{seed}/`.
`--smoke` flag. Skips completed runs (check for `metrics.csv` existence).

---

### M3 · Task 12: Linear CKA library
`src/analysis/cka.py` — `linear_cka(X, Y) -> float` (centered kernel alignment).
`between_task_cka(model, probe_tensors, layer_names) -> dict[str, ndarray]` — registers
forward hooks, returns `(n_tasks x n_tasks)` similarity matrix per layer name.

### M3 · Task 13: Weight drift library
`src/analysis/drift.py` — `snapshot(model) -> dict` saves `state_dict` clone.
`compute_drift(snapshots) -> dict[int, dict[str, float]]` — given a list of per-task
snapshots, returns L2 norm of `(theta_t - theta_0)` per named parameter per task.

### M3 · Task 14: Feature extraction script
`scripts/extract.py` — for each run directory in `results/runs/`:
1. Build or load `data/probe_set.pt`: 500 samples per task, sampled from val split
   with val_transform (no augmentation), fixed seed=42, independent of run seed.
2. Load `ckpt_task{t}.pt` for t in 0..9; run `between_task_cka` -> save `cka.npz`.
3. Compute pairwise L2 drift from task-0 checkpoint for all t -> save `drift.npz`.
4. Writes to `results/features/{run}/`.
`--run` flag to process a single run; default processes all.

### M3 · Task 15: Plot script
`scripts/plot.py` — reads only CSVs and npz files; zero model loading:
- Learning curves from `train_log.csv` per run
- AA / BWT / AF comparison table and bar chart from `metrics.csv`
- CKA heatmaps per arch per method from `cka.npz`
- Drift-per-layer-type bar charts from `drift.npz`
Saves all figures to `results/figures/`.

---

## Validation

```bash
# M1 smoke tests (all passing)
python -c "from configs.default import Config; c=Config(); assert c.n_classes==100"
python -c "from configs.default import Config; c=Config(n_tasks=5,classes_per_task=10); assert c.n_classes==50"
python -c "from src.models.vit import get_vit_small, N_PATCHES; import torch; print(get_vit_small()(torch.randn(2,3,32,32)).shape, N_PATCHES)"
python -c "from src.models.resnet import get_resnet18; import torch; print(get_resnet18()(torch.randn(2,3,32,32)).shape)"
python scripts/pilot.py --smoke

# M1 full pilot (gate check) — run outside session
# source .venv/bin/activate && python scripts/pilot.py 2>&1 | tee logs/pilot.log

# M2 smoke
python scripts/run_cl.py --smoke

# M3 (after M2 complete) — run outside session
# source .venv/bin/activate && python scripts/extract.py 2>&1 | tee logs/extract.log
# python scripts/plot.py
```

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| ViT < 55% on CIFAR-100 joint | Medium | Switch to Tiny ImageNet per PRD fallback; update INPUT_SIZE=64 in vit.py |
| Conv stem token count mismatch after dataset switch | Low | N_PATCHES computed from INPUT_SIZE and STEM_STRIDE — update INPUT_SIZE, retest |
| EWC Fisher OOM | Low | Diagonal only; ViT-Small ~11M params; 20% subsample limits activation mem |
| Class-IL collapse | Low-Med | ER with 500 exemplars anchors at least one non-collapsed condition |
| CKA probe-set mem | Low | 500 x 65 x 384 x 4 bytes ~= 50MB per task — fits comfortably |
| Stale features/ after retraining a run | Low | extract.py overwrites features/ for the specified run; document in CLAUDE.md |

---

## Acceptance

- [x] M1: Both models train on full CIFAR-100; smoke tests pass; per-epoch CSV written
- [ ] M2: All 18 runs complete; `results/runs/` has `metrics.csv` + `train_log.csv` + 10 ckpts per run
- [ ] M3: `results/features/` has `cka.npz` + `drift.npz` per run; `results/figures/` populated
- [ ] Smoke tests pass for all scripts
- [ ] No HuggingFace weights; no pretrained torchvision weights; PyTorch only
