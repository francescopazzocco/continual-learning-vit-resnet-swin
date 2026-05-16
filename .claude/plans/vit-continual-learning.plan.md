# Plan: ViT vs ResNet-18 under Continual Learning

**Source PRD**: `.claude/prds/vit-continual-learning.prd.md`
**Selected Milestone**: M1b — Swin-Tiny pilot (next pending)
**Complexity**: Medium

## Summary

Add a Swin-Tiny architecture adapted for 32×32 inputs (2 stages, patch=4, window=4) and
extend the pilot script so Swin trains on full CIFAR-100 and must pass the ≥55% accuracy
gate before M2 begins. This milestone also resolves two open questions in the PRD: the
Swin-32 architecture spec and the layer-naming taxonomy needed by M3.

## Patterns to Mirror

| Category | Source | Pattern |
|---|---|---|
| Naming | `src/models/vit.py:127` | `get_swin_tiny(n_classes)` factory mirrors `get_vit_small` |
| Module layout | `src/models/vit.py:14` | `ALL_CAPS` constants at top; one class per responsibility |
| Weight init | `src/models/vit.py:104` | `trunc_normal_(std=0.02)` for Linear/embed; ones/zeros for norm |
| Errors | `scripts/pilot.py:103` | `print("[FAIL] ...")` + `sys.exit(1)` for gate failure |
| Logging | `scripts/pilot.py:79` | `print(f"  [arch] params: {n:.2f}M")` before training |
| Smoke | `scripts/pilot.py:76` | `--smoke` passes through to `fit()`; results dict accumulates |

---

## Data / Analysis Separation

No change to the three-phase rule: Swin checkpoints land in `results/pilot/` the same
way ViT and ResNet checkpoints do. No analysis or plotting in training scripts.

---

## Architecture Spec (resolves PRD open question)

| Hyperparameter | Value | Rationale |
|---|---|---|
| `PATCH_SIZE` | 4 | 32/4 = 8 → 64 tokens, matches ViT's token count |
| `WINDOW_SIZE` | 4 | stage 0: four 4×4 windows on 8×8 grid (true windowing); stage 1: one 4×4 window on 4×4 grid (degrades to global, acceptable) |
| `EMBED_DIM` | 96 | standard Swin-Tiny base width |
| `DEPTHS` | [2, 6] | mirrors Swin-Tiny heavy-stage ratio ([2,2,6,2] → two-stage [2,6]) |
| `NUM_HEADS` | [3, 6] | 96/32=3 and 192/32=6 head dims |
| `MLP_RATIO` | 4.0 | consistent with ViT and standard Swin |
| `DROP_PATH_RATE` | 0.1 | standard stochastic depth; 8 total blocks |
| Final dim | 192 | one PatchMerging doubles 96→192 |

**Layer taxonomy for CKA/drift (resolves second PRD open question):**

```
SwinTiny32
  patch_embed          Conv2d(3,96,k=4,s=4) + LN  — locality: stem
  stages.0.0           SwinBlock W-MSA  at 8x8     — windowed attention
  stages.0.1           SwinBlock SW-MSA at 8x8     — windowed attention (shifted)
  patch_merging        4*96 -> 192 Linear           — spatial compression
  stages.1.{0..5}      SwinBlocks at 4x4            — effectively global attention
  norm                 LayerNorm(192)
  head                 Linear(192, n_classes)
```

---

## Files to Change

### M1b — Swin-Tiny Pilot

| File | Action | Why |
|---|---|---|
| `src/models/swin.py` | CREATE | Swin-Tiny from scratch for 32×32; `get_swin_tiny(n_classes)` |
| `scripts/pilot.py` | UPDATE | Add `swin` to --arch choices; accuracy gate applies to swin too |

### M2 — CL Training (updated for 3-arch grid)

| File | Action | Why |
|---|---|---|
| `src/cl/base.py` | CREATE | `CLMethod` ABC |
| `src/cl/vanilla.py` | CREATE | pass-through |
| `src/cl/ewc.py` | CREATE | Online EWC; diagonal Fisher; 20% subsample |
| `src/cl/er.py` | CREATE | `ReservoirBuffer(500)`; replay per step |
| `src/metrics.py` | CREATE | `compute_metrics(R)` → AA, BWT, AF |
| `src/cl_trainer.py` | CREATE | task-iterator CL loop; writes CSVs + ckpts |
| `scripts/run_cl.py` | CREATE | **3** arch × 3 methods × 3 seeds = 27 runs; `--smoke` |

### M3 — Mechanistic Analysis

| File | Action | Why |
|---|---|---|
| `src/analysis/cka.py` | CREATE | `linear_cka()` + `between_task_cka()` |
| `src/analysis/drift.py` | CREATE | `snapshot()` + `compute_drift()` |
| `scripts/extract.py` | CREATE | checkpoints → `cka.npz` + `drift.npz` per run |
| `scripts/plot.py` | CREATE | CSVs + npz → figures; zero model loading |

---

## Tasks

### M1 — Tasks 1–6 (DONE)

All M1 tasks complete. ViT 63.70%, ResNet 63.86%; accuracy gate passed 2026-05-16.

---

### M1b · Task 1: Create `src/models/swin.py`

- **Action**: Implement from scratch. Module structure:
  - Module-level constants: `EMBED_DIM=96`, `DEPTHS=[2,6]`, `NUM_HEADS=[3,6]`,
    `WINDOW_SIZE=4`, `PATCH_SIZE=4`, `MLP_RATIO=4.0`, `DROP_PATH_RATE=0.1`,
    `INPUT_SIZE=32`
  - `window_partition(x, ws)` / `window_reverse(windows, ws, H, W)` — reshape helpers;
    no learnable state
  - `WindowAttention(dim, num_heads, window_size)` — `nn.Linear` Q/K/V and proj;
    learnable relative position bias `nn.Parameter` of shape
    `(2*ws-1, 2*ws-1, num_heads)`; indexed via pre-computed coordinate table;
    scaled dot-product attention with optional additive mask
  - `StochasticDepth(p)` — Bernoulli drop of entire residual path during training
  - `SwinBlock(dim, num_heads, window_size, shift, mlp_ratio, drop_path_rate)` —
    pre-LN; W-MSA when `shift=False`, SW-MSA (cyclic shift by `ws//2`) when
    `shift=True`; compute attention mask for SW-MSA on forward (cache by H,W);
    MLP is two Linear + GELU; stochastic depth on both residuals
  - `PatchMerging(dim)` — concatenate 2×2 spatial neighbors →
    `LayerNorm(4*dim)` → `Linear(4*dim, 2*dim, bias=False)`
  - `PatchEmbed` — `Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE,
    stride=PATCH_SIZE, bias=False)` + `LayerNorm(EMBED_DIM)`
  - `SwinTiny32(n_classes)`:
    - `self.patch_embed`: PatchEmbed
    - `self.stages`: `nn.ModuleList` of two `nn.ModuleList`s built from `DEPTHS`
      and `NUM_HEADS`; blocks alternate `shift=False/True`
    - `self.patch_merging`: PatchMerging(EMBED_DIM) applied after `stages[0]`
    - `self.norm`: `LayerNorm(EMBED_DIM * 2)`
    - `self.head`: `Linear(EMBED_DIM * 2, n_classes)`
    - `_init_weights()`: `trunc_normal_(std=0.02)` for Linear and position bias;
      ones/zeros for LayerNorm; zeros for head bias
    - `forward`: patch_embed → (B,8,8,96) → stage0 → merge → (B,4,4,192) →
      stage1 → norm → mean over spatial → head
  - `get_swin_tiny(n_classes=100) -> SwinTiny32`
- **Mirror**: `src/models/vit.py` — constant block, class layout, `_init_weights`,
  factory function signature
- **Validate**:
  ```bash
  python -c "
  from src.models.swin import get_swin_tiny
  import torch
  m = get_swin_tiny()
  x = torch.randn(2, 3, 32, 32)
  out = m(x)
  assert out.shape == (2, 100), out.shape
  params = sum(p.numel() for p in m.parameters()) / 1e6
  print(f'[OK] output {out.shape}, {params:.2f}M params')
  "
  ```
  Expected: output `torch.Size([2, 100])`, param count 2–4M.

---

### M1b · Task 2: Update `scripts/pilot.py`

- **Action**:
  - Add `"swin"` to the `--arch` choices list; extend the `"both"` sentinel to
    `"all"` (or keep `"both"` and add a separate `"swin"` branch — match the
    existing pattern: `if args.arch in ("swin", "all"):`)
  - Add `from src.models.swin import get_swin_tiny` import
  - Add a swin block mirroring the resnet block (lines 84–90): build model,
    print params, call `fit`, store `results["swin"]`
  - Extend the gate-check block: the gate currently only applies to `"vit"`;
    add a parallel check for `"swin"` with the same `VIT_ACCURACY_GATE` threshold
    and the same `[FAIL]`/`sys.exit(1)` path
  - Do not change `fit()`, `get_joint_loaders`, or the CSV/checkpoint logic
- **Mirror**: `scripts/pilot.py:84–111` — resnet branch + gate-check structure
- **Validate**:
  ```bash
  python scripts/pilot.py --smoke --arch swin
  # Expect: [SMOKE] swin: XX.XX% with no crash
  python scripts/pilot.py --smoke --arch all
  # Expect: three [SMOKE] lines, no crash
  ```

---

### M2 · Task 3: CL method base + vanilla (unchanged from original plan)

`src/cl/base.py` — `CLMethod(ABC)`: `before_task`, `after_task`, `loss` hooks.
`src/cl/vanilla.py` — no-op.

### M2 · Task 4: EWC (unchanged)

`src/cl/ewc.py` — diagonal Fisher on 20% subsample; online accumulation.

### M2 · Task 5: ER (unchanged)

`src/cl/er.py` — `ReservoirBuffer(max_size=500)`; replay batch concatenated before
each forward.

### M2 · Task 6: Metrics + CL trainer (unchanged)

`src/metrics.py` — `compute_metrics(R)` → AA, BWT, AF.
`src/cl_trainer.py` — task-iterator loop; CLMethod hooks; CSVs + ckpts.

### M2 · Task 7: Run script (updated — 3 archs)

`scripts/run_cl.py` — grid over `arch in {vit, resnet, swin}`,
`method in {vanilla, ewc, er}`, `seed in {0, 1, 2}` = **27 runs**.
Add `get_swin_tiny` import and `"swin"` branch in `_build_model`.
`--arch swin` flag must work. Skip completed runs (`metrics.csv` check).

---

### M3 · Task 8: Linear CKA library (unchanged)

`src/analysis/cka.py`

### M3 · Task 9: Weight drift library (unchanged)

`src/analysis/drift.py`

### M3 · Task 10: Feature extraction script (unchanged except 3-arch scope)

`scripts/extract.py` — processes all 27 run directories.

### M3 · Task 11: Plot script (unchanged except 3-arch scope)

`scripts/plot.py` — CKA heatmaps and drift charts for all three architectures.

---

## Validation

```bash
# M1b smoke (run now, in session — fast)
python scripts/pilot.py --smoke --arch swin
python scripts/pilot.py --smoke --arch all

# M1b full pilot (run outside session — ~same wall time as ViT/ResNet)
source .venv/bin/activate && python scripts/pilot.py --arch swin 2>&1 | tee logs/swin_pilot.log

# M2 smoke (after M1b impl)
python scripts/run_cl.py --smoke --arch swin
python scripts/run_cl.py --smoke

# M2 full grid (overnight, outside session)
source .venv/bin/activate && python scripts/run_cl.py 2>&1 | tee logs/m2_grid.log

# M3 (after M2 complete, outside session)
source .venv/bin/activate && python scripts/extract.py 2>&1 | tee logs/extract.log
python scripts/plot.py
```

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Swin fails ≥55% gate (insufficient capacity at embed_dim=96) | Medium | First try increasing DEPTHS to [2,6] (already planned); if still failing, raise EMBED_DIM to 128 and re-run pilot |
| `window_size == H` in stage 1 produces incorrect attention mask | Low | Smoke test catches shape errors; add `assert` in `window_partition` that `H % ws == 0` |
| `window_partition` reshape/permute off-by-one | Low | Validate with unit assertion: `window_reverse(window_partition(x, ws), ws, H, W) == x` |
| Cyclic shift mask for stage 1 (4×4 map, 1 window) is non-zero | Low | With 1 window covering the full map, all tokens share the same window region; mask should be all-zero; verify manually in smoke |
| Stochastic depth drops too aggressively during smoke (1 epoch) | Very low | Drop path acts as identity at eval; training with p=0.1 and 8 blocks is mild |
| run_cl.py: 27 runs vs 18 — compute budget | Low | PRD risk table already accepts this; RTX 5070 Ti handles it overnight |

---

## Acceptance

- [x] M1: Both ViT and ResNet train on full CIFAR-100; smoke tests pass; accuracy gate passed
- [x] M1b: Swin-Tiny smoke test passes; full pilot reaches ≥55% on CIFAR-100; `swin_best.pt` written to `results/pilot/` — 57.34% (2026-05-16)
- [ ] M2: All 27 runs complete; each `results/runs/{run}/` has `metrics.csv` + `train_log.csv` + 10 ckpts
- [ ] M3: `results/features/` has `cka.npz` + `drift.npz` per run; `results/figures/` populated
- [ ] Smoke tests pass for all scripts with `--arch all` / `--arch swin`
- [ ] No HuggingFace weights; no pretrained torchvision weights; PyTorch only
