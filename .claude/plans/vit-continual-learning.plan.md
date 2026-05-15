# Plan: ViT vs ResNet-18 under Continual Learning

**Source PRD**: `.claude/prds/vit-continual-learning.prd.md`
**Selected Milestone**: M1 — Infrastructure + Pilot
**Complexity**: Large

## Summary

Build the full project scaffold, data pipeline, both model architectures, and a joint-training pilot run. The pilot determines whether ViT-Small from scratch hits ≥ 55% on full CIFAR-100 — a hard gate before any CL experiments begin. This plan covers all four milestones but scopes implementation tasks per milestone so they can be confirmed and executed sequentially.

## Patterns to Mirror

No existing code in this repository. Conventions below are established by this plan and must be applied consistently.

| Category | Source | Pattern |
|---|---|---|
| Naming | —new— | `snake_case` modules, `CamelCase` classes, `ALL_CAPS` constants |
| Errors | —new— | raise `ValueError` / `RuntimeError` with explicit messages; no silent fallbacks |
| Logging | —new— | CSV via `csv.DictWriter` appended per task/epoch; print to stdout with `tqdm` |
| Data access | —new— | All dataset construction goes through `src/data/cifar100.py`; loaders returned as plain `DataLoader` pairs |
| Tests | —new— | `scripts/` smoke-test scripts (`python scripts/pilot.py --smoke`) run a 1-epoch single-batch sanity check |

---

## Directory Structure

```
project/
├── src/
│   ├── models/
│   │   ├── vit.py          # ViT-Small from scratch (conv stem)
│   │   └── resnet.py       # ResNet-18 via torchvision (no pretrained)
│   ├── data/
│   │   └── cifar100.py     # Download + Split-CIFAR-100 class-IL task splits
│   ├── cl/
│   │   ├── base.py         # Abstract CLMethod interface
│   │   ├── vanilla.py      # Vanilla fine-tuning
│   │   ├── ewc.py          # EWC with 20%-subsample Fisher
│   │   └── er.py           # Experience Replay, reservoir buffer (500)
│   ├── analysis/
│   │   ├── cka.py          # Linear CKA on fixed probe set
│   │   └── drift.py        # L2 weight drift per named layer
│   ├── metrics.py          # AA, BWT, AF from per-task accuracy matrix
│   ├── trainer.py          # Joint training loop (pilot use)
│   └── cl_trainer.py       # CL task-iterator loop (single head, class-IL)
├── configs/
│   └── default.py          # Flat dataclass config (no YAML dep)
├── scripts/
│   ├── pilot.py            # M1: joint train both archs, print accuracy
│   ├── run_cl.py           # M2: all 18–30 CL runs, log to results/
│   └── analyze.py          # M3: CKA matrices + drift plots
├── results/                # Auto-created; one subdir per run
├── requirements.txt
└── README.md
```

**Config system**: a single `configs/default.py` with a `@dataclass` holding all hyperparameters. Scripts import it and override fields via `argparse`. No YAML dependency.

---

## Key Decisions (resolved open questions)

| Question | Decision | Rationale |
|---|---|---|
| Config system | `dataclass` + argparse | No extra dep; IDE-friendly; reproducible |
| CL head | Single 100-class linear head from task 1 | Standard class-IL protocol; no task-ID at inference |
| EWC Fisher | 20% subsample of task train set | Stable estimates; ~5× faster; validated in ablation |
| ER buffer | 500 total exemplars, reservoir sampling | ~5/class after 10 tasks; more discriminative than 200 |
| CKA probe set | 500 samples from each task's val split | Fits in GPU mem for ViT-Small; computed post-task |
| Logging | CSV append per run + matplotlib post-hoc | No tensorboard dep; plain files shareable with team |
| Augmentation | RandAugment(n=2,m=9) + Normalize only | Standard for CIFAR-100 from-scratch; no Mixup (complicates class-IL labels) |

---

## Files to Change

### M1 — Infrastructure + Pilot

| File | Action | Why |
|---|---|---|
| `src/__init__.py` | CREATE | Package root |
| `src/models/__init__.py` | CREATE | Package |
| `src/data/__init__.py` | CREATE | Package |
| `configs/default.py` | CREATE | All hyperparams as dataclass |
| `src/data/cifar100.py` | CREATE | Download, split into 10 tasks, return loaders |
| `src/models/vit.py` | CREATE | ViT-Small: conv stem → patch tokens → 6 transformer blocks → CLS → head |
| `src/models/resnet.py` | CREATE | ResNet-18 torchvision, `pretrained=False`, replace fc for 100 classes |
| `src/trainer.py` | CREATE | `train_epoch`, `eval_epoch`, `fit` (joint, no CL logic) |
| `scripts/pilot.py` | CREATE | Train both archs on full CIFAR-100; print top-1; exit non-zero if ViT < 55% |

### M2 — CL Training

| File | Action | Why |
|---|---|---|
| `src/cl/base.py` | CREATE | `CLMethod` ABC: `before_task`, `after_task`, `loss` hooks |
| `src/cl/vanilla.py` | CREATE | Pass-through; no regularization |
| `src/cl/ewc.py` | CREATE | Online EWC; Fisher via 20%-subsample diagonal |
| `src/cl/er.py` | CREATE | Reservoir buffer; replay batch mixed into each training step |
| `src/metrics.py` | CREATE | `compute_metrics(acc_matrix)` → AA, BWT, AF |
| `src/cl_trainer.py` | CREATE | Task-iterator loop; calls `CLMethod` hooks; logs per-task accuracy matrix |
| `scripts/run_cl.py` | CREATE | Grid: 2 arch × 3 methods × seeds; saves `results/{arch}_{method}_s{seed}/` |

### M3 — Mechanistic Analysis

| File | Action | Why |
|---|---|---|
| `src/analysis/cka.py` | CREATE | `linear_cka(X, Y)` + `between_task_cka(model, probe_sets, layers)` |
| `src/analysis/drift.py` | CREATE | `weight_drift(model, ref_state_dict)` → dict of L2 norms per layer name |
| `scripts/analyze.py` | CREATE | Load saved checkpoints; produce CKA matrices + drift plots per arch |

---

## Tasks

### M1 · Task 1: Config dataclass
- **Action**: Create `configs/default.py` with `@dataclass Config` holding: `arch`, `method`, `seed`, `n_tasks=10`, `classes_per_task=10`, `batch_size=128`, `lr=0.1`, `epochs_per_task=50` (CL) / `epochs=200` (pilot joint), `wd=5e-4`, `ewc_lambda=1000`, `er_buffer_size=500`, `fisher_subsample=0.2`, `device="cuda"`, `data_root="./data"`, `results_root="./results"`.
- **Mirror**: no prior pattern; establish this as the single source of truth
- **Validate**: `python -c "from configs.default import Config; print(Config())"`

### M1 · Task 2: CIFAR-100 data pipeline
- **Action**: `src/data/cifar100.py` — `get_joint_loaders(cfg)` returns `(train_loader, val_loader)` over full CIFAR-100. `get_split_loaders(cfg)` returns a list of 10 `(train_loader, val_loader)` pairs, each covering exactly 10 classes, with class indices remapped to 0–99 globally (no per-task remapping — single head sees original label space).
- **Mirror**: no prior pattern
- **Validate**: `python -c "from src.data.cifar100 import get_split_loaders; from configs.default import Config; splits = get_split_loaders(Config()); assert len(splits)==10; print([len(s[0].dataset) for s in splits])"`

### M1 · Task 3: ResNet-18
- **Action**: `src/models/resnet.py` — wrap `torchvision.models.resnet18(weights=None)`, replace `model.fc` with `nn.Linear(512, n_classes)`. Expose `get_resnet18(n_classes=100)`.
- **Validate**: `python -c "from src.models.resnet import get_resnet18; import torch; m=get_resnet18(); print(m(torch.randn(2,3,32,32)).shape)"` — expect `torch.Size([2, 100])`

### M1 · Task 4: ViT-Small from scratch
- **Action**: `src/models/vit.py` — implement:
  - **Conv stem**: two `Conv2d` layers (3→32→dim, stride 2+2 = effective 4×4 patches) producing `(B, N, dim)` token sequence. `dim=384`.
  - **CLS token** + **learned positional embeddings**.
  - **6 TransformerBlock** layers: `LayerNorm → MultiheadAttention(6 heads) → residual → LayerNorm → MLP(dim→4*dim→dim) → residual`.
  - **Head**: `LayerNorm → Linear(384, n_classes)` on CLS token.
  - Expose `get_vit_small(n_classes=100)`.
- **Validate**: `python -c "from src.models.vit import get_vit_small; import torch; m=get_vit_small(); print(m(torch.randn(2,3,32,32)).shape)"` — expect `torch.Size([2, 100])`; also print param count (target ≈ 5–8M).

### M1 · Task 5: Joint trainer
- **Action**: `src/trainer.py` — `train_epoch(model, loader, optimizer, device)` and `eval_epoch(model, loader, device) → float`. Both use cross-entropy. `fit(model, train_loader, val_loader, cfg) → list[float]` runs `cfg.epochs` epochs with cosine LR schedule, saves best checkpoint.
- **Validate**: smoke-test with `--smoke` flag (1 epoch, 2 batches).

### M1 · Task 6: Pilot script
- **Action**: `scripts/pilot.py` — train ViT-Small and ResNet-18 on full CIFAR-100 joint, print final val accuracy for each. Exit with code 1 and message if ViT < 55%.
- **Validate**: `python scripts/pilot.py --smoke` completes without error; full run confirms gate.

---

### M2 · Task 7: CL method base + vanilla
- **Action**: `src/cl/base.py` defines `CLMethod(ABC)` with hooks `before_task(task_id, train_loader)`, `after_task(task_id, train_loader, model)`, `loss(logits, targets, model) → Tensor`. `src/cl/vanilla.py` is a no-op implementation.

### M2 · Task 8: EWC
- **Action**: `src/cl/ewc.py` — after each task, subsample 20% of task train data, compute diagonal Fisher `F_i` for each parameter. Penalty: `λ/2 · Σ_i F_i · (θ - θ*_i)²`. Accumulate across tasks (online EWC). `ewc_lambda` from config.

### M2 · Task 9: ER
- **Action**: `src/cl/er.py` — `ReservoirBuffer(max_size=500)` with `update(x, y)` and `sample(n) → (x, y)`. Each training step: concatenate buffer replay batch with current batch before forward pass. Buffer updated after each task.

### M2 · Task 10: Metrics + CL trainer
- **Action**: `src/metrics.py` — `compute_metrics(R: np.ndarray) → dict` where `R[i,j]` is accuracy on task j after training task i. Returns AA, BWT, AF per standard definitions. `src/cl_trainer.py` — `run_cl(model, splits, method, cfg) → R` iterates tasks, calls hooks, logs `R` to `results/{arch}_{method}_s{seed}/metrics.csv`.

### M2 · Task 11: Run script
- **Action**: `scripts/run_cl.py` — grid over `arch ∈ {vit, resnet}`, `method ∈ {vanilla, ewc, er}`, `seed ∈ {0,1,2}`. Saves checkpoints per task for later analysis. `--smoke` flag for fast validation.

---

### M3 · Task 12: Linear CKA
- **Action**: `src/analysis/cka.py` — `linear_cka(X, Y) → float` (centered kernel alignment). `between_task_cka(model, probe_sets, layer_names) → np.ndarray` hooks activations via `register_forward_hook`, returns `(n_tasks × n_tasks)` similarity matrix per layer.

### M3 · Task 13: Weight drift
- **Action**: `src/analysis/drift.py` — `snapshot(model) → dict` saves `state_dict` clone. `compute_drift(model, ref) → dict[str, float]` returns L2 norm of `(param - ref_param)` per named parameter. Called after each task in CL trainer.

### M3 · Task 14: Analysis script
- **Action**: `scripts/analyze.py` — loads checkpoints from `results/`; produces: (a) CKA heatmaps per arch per method, (b) drift-per-layer-type bar charts. Saves figures to `results/figures/`.

---

## Validation

```bash
# M1 smoke tests (run before full pilot)
python -c "from configs.default import Config; print(Config())"
python -c "from src.data.cifar100 import get_split_loaders; from configs.default import Config; splits=get_split_loaders(Config()); assert len(splits)==10"
python -c "from src.models.vit import get_vit_small; import torch; print(get_vit_small()(torch.randn(2,3,32,32)).shape)"
python -c "from src.models.resnet import get_resnet18; import torch; print(get_resnet18()(torch.randn(2,3,32,32)).shape)"
python scripts/pilot.py --smoke

# M1 full pilot (gate check)
python scripts/pilot.py  # ViT must reach ≥ 55% top-1 on CIFAR-100

# M2 smoke
python scripts/run_cl.py --smoke

# M3 (after M2 runs complete)
python scripts/analyze.py
```

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| ViT < 55% on CIFAR-100 joint | Medium | Switch dataset to Tiny ImageNet per PRD fallback; don't change research question |
| Conv stem produces too few tokens for 6 blocks | Low | Verify spatial resolution: 32×32 → 8×8 (64 tokens) with stride-2+2 stem — adequate |
| EWC Fisher OOM on full parameter set | Low | Diagonal only; ViT-Small ≈ 6M params; 20% subsample limits activation mem |
| Class-IL collapse (all methods near 0% after task 1) | Low-Med | ER with 500 exemplars should prevent full collapse; collapse itself is a reportable finding |
| CKA probe-set mem with full val split | Low | 500 samples × ViT activations ≈ 500 × 64 × 384 = ~47M floats = ~180MB — fits |

---

## Acceptance

- [ ] M1: Both models train on full CIFAR-100; ViT top-1 ≥ 55%
- [ ] M2: All 18 runs (2×3×3) complete; `results/` has `metrics.csv` per run with AA/BWT/AF
- [ ] M3: CKA matrices and drift plots exist in `results/figures/`
- [ ] Smoke tests pass for all three milestone scripts
- [ ] No HuggingFace weights; no pretrained torchvision weights; PyTorch only
