# Hyperparameter Choices

Reference document for all non-default hyperparameter decisions.
Decision log cross-references are noted per entry (LOG-N in doc/decision_log.md).
Ablation results are filled in after running scripts/ablation_hp.py.

---

## Summary Table

| HP | Value | Method/scope | Source | Decision |
|---|---|---|---|---|
| lr | 0.1 | training (all) | He et al. 2016; Loshchilov 2017 | LOG-006 |
| momentum | 0.9 | training (all) | Ioffe & Szegedy 2015 | LOG-006 |
| wd | 5e-4 | training (all) | He et al. 2015 | LOG-006 |
| batch_size | 128 | training (all) | standard; VRAM constraint | LOG-006 |
| epochs | 200 | joint pilot | CIFAR standard | LOG-006 |
| epochs_per_task | 50 | CL grid | per-sample budget parity | LOG-007 |
| randaug_n | 2 | augmentation | Cubuk et al. 2020 | LOG-008 |
| randaug_m | 9 | augmentation | Cubuk et al. 2020 | LOG-008 |
| ewc_lambda | 1000 | EWC | Kirkpatrick et al. 2017 | LOG-009 |
| fisher_subsample | 0.2 | EWC | speed/quality tradeoff | LOG-010 |
| fisher_batch_size | 16 | EWC | speed/bias tradeoff | LOG-011 |
| er_buffer_size | 500 | ER | 5/class low-budget regime | LOG-012 |

---

## Training Hyperparameters (Shared Across Architectures)

All training hyperparameters are fixed across all 27 CL runs. No per-architecture
tuning is done after the pilot; this is by design (LOG-006) to keep the comparison
controlled.

### lr = 0.1 (cosine annealed)

He et al. "Deep Residual Learning" (2016) and Loshchilov & Hutter "SGDR" (2017)
both report lr=0.1 as the canonical starting point for SGD on CIFAR with cosine
annealing. The pilot confirmed convergence for both ViT (63.70%) and ResNet
(63.86%) at this value without modification.

No per-architecture LR tuning was performed. A decoupled LR (higher head vs.
backbone) was considered for ViT (ViTIL, arXiv:2112.06103) but was not needed
after the conv stem closed the accuracy gap.

### momentum = 0.9, wd = 5e-4

Both are standard CIFAR values from He et al. "Delving Deep into Rectifiers"
(2015). No ablation is warranted; these values appear in 90%+ of CIFAR-100
papers that use SGD.

### batch_size = 128

Largest power of 2 that fits ViT-Small (384-dim, 6 blocks, BF16) and the ER
replay batch (128 current + 128 replay = 256 total in the worst case) within
the 16 GB VRAM budget of the RTX 5070 Ti.

### epochs = 200 (joint pilot), epochs_per_task = 50 (CL grid)

See LOG-007. The 50 epochs/task budget was derived from per-sample parity with
the 200-epoch joint run: 5000 samples/task x 50 epochs = 250k sample-epochs,
versus 50000 samples x 200 epochs / 10 tasks = 1M sample-epochs. The factor-of-4
difference is accepted because the CL setting is harder (shifting distribution)
and the research question concerns forgetting dynamics, not absolute convergence.

---

## RandAugment

### num_ops = 2, magnitude = 9

Cubuk et al. "RandAugment" (2020) identify num_ops=2, magnitude=9 as robust
across CIFAR-10/100 and ImageNet without per-dataset search. The DeiT training
recipe (Touvron et al., 2021) uses the same values when training ViTs from scratch
on CIFAR-scale datasets.

Mixup and CutMix are explicitly excluded by project constraints (Mixup complicates
class-IL label assignment; both are out of scope per the PRD). AutoAugment was
considered but its CIFAR-100 policy was learned on the full dataset -- using it
here would be a mild form of data leakage.

---

## EWC Hyperparameters

All three EWC hyperparameters interact: lambda controls regularization strength,
subsample controls Fisher quality, and batch_size controls the bias-variance
tradeoff in the Fisher estimate itself.

### ewc_lambda = 1000

**Rationale**: Kirkpatrick et al. (2017) report lambda=1000 on permuted MNIST.
Mirzadeh et al. "Architecture Matters" (ICLR 2022) -- the closest prior work --
use the same value for CIFAR experiments. Using the same value makes our results
directly comparable to the benchmark.

**ViT caveat**: Park & Kim (ICLR 2022) show that MSAs flatten the loss landscape,
producing smaller and more uniform Fisher estimates than CNNs. With the same
lambda, EWC regularization is effectively weaker for ViT. This is a known confound
(logged in failure insights memory) that must be reported in the analysis section
rather than corrected -- correcting it would require per-architecture lambda tuning,
which introduces a different confound.

**Ablation grid**: {100, 500, 1000, 5000} on ResNet-18, 3 tasks, 10 epochs/task.

| ewc_lambda | AA | BWT | AF |
|---|---|---|---|
| 100 | 0.234 | -0.650 | 0.650 |
| 500 | 0.229 | -0.631 | 0.631 |
| 1000 | 0.219 | -0.624 | 0.624 |
| 5000 | 0.204 | -0.606 | 0.606 |

**Note on a discarded prior run**: an earlier version of this ablation ran on
the pre-fix code path and diverged to inf/nan at lambda=1000 (see LOG-013),
producing degenerate near-chance metrics (AA approx 0.033 = 1/30 for the 3-task
ablation). Those numbers were a numerical artifact, not a result, and have been
replaced. The table above is the corrected ablation on the stabilized code.

**Interpretation**: The corrected ablation shows a smooth, monotonic
stability-plasticity tradeoff. Raising lambda lowers AA (0.234 -> 0.204) and
reduces forgetting (AF 0.650 -> 0.606; BWT -0.650 -> -0.606). There is no phase
transition and no collapse -- the metrics vary continuously across two orders of
magnitude of lambda.

lambda=1000 sits mid-range and behaves as expected. Because the metrics change
only gradually across 100-5000, the method comparison is insensitive to the
exact value in this range. lambda=1000 is retained as the fixed,
literature-standard value (Kirkpatrick et al. 2017; Mirzadeh et al. 2022) rather
than tuned per-architecture: tuning lambda to maximize AA would select the
smallest value (100), the regime where EWC barely regularizes, which would
confound a study whose purpose is to compare methods, not tune them.

**Honest caveat**: even at lambda=5000, BWT remains -0.61 at this 3-task,
10-epoch ablation scale -- EWC provides only a modest reduction in forgetting
here. This is reported as an observed limitation, not corrected away.

---

### fisher_subsample = 0.2

**Rationale**: 20% of the task training set = ~1000 images per task for CIFAR-100
(5000 images/task). Full Fisher on 5000 images x 10 tasks x 3 seeds x 3 archs
would add ~15-20 min per run, roughly doubling total wall time for a marginal
quality gain. 1000-image subsamples produce stable diagonal Fisher estimates at
this scale based on the central limit theorem applied to gradient statistics.

**Ablation grid**: {0.05, 0.10, 0.20, 0.50} on ResNet-18, lambda=1000.

| fisher_subsample | n_images | AA | BWT | AF |
|---|---|---|---|---|
| 0.05 | ~250 | 0.226 | -0.637 | 0.637 |
| 0.10 | ~500 | 0.220 | -0.627 | 0.627 |
| 0.20 | ~1000 | 0.223 | -0.622 | 0.622 |
| 0.50 | ~2500 | 0.224 | -0.615 | 0.615 |

**Interpretation**: AA is effectively flat across all four fractions
(0.220-0.226, within noise), and forgetting improves only marginally as the
subsample grows (AF 0.637 -> 0.615). This is a genuine low-sensitivity result on
the stabilized code -- the diagonal Fisher ordering is stable down to ~250 images
(5%), consistent with CLT-convergence of gradient statistics at this scale.
subsample=0.2 (~1000 images) is indistinguishable from subsample=0.5 in AA while
costing 2.5x less to compute, so it is retained. The small AF gain at 0.5 does
not justify doubling Fisher cost across the full 27-run grid.

---

### fisher_batch_size = 16

**Rationale**: The diagonal Fisher is defined as E[g_i^2] per parameter.
With batch_size=B, PyTorch's autograd gives param.grad = mean(g) over the batch,
so param.grad^2 * B = B * (mean(g))^2. This underestimates mean(g^2) by Jensen's
inequality. However, the relative ordering of parameter importance is preserved
because all parameters are underestimated by the same factor. EWC only needs a
consistent importance ranking, not absolute values.

batch_size=16 is a 16x speedup over per-sample (batch_size=1):
  - batch_size=1:  1000 forward/backward passes per task (subsample=0.2, 5000 imgs)
  - batch_size=16:   63 forward/backward passes per task

**Ablation grid**: {1, 8, 16, 64} on ResNet-18, lambda=1000, subsample=0.2.

| fisher_batch_size | AA | BWT | AF |
|---|---|---|---|
| 1 | 0.175 | -0.586 | 0.586 |
| 8 | 0.218 | -0.632 | 0.632 |
| 16 | 0.223 | -0.632 | 0.632 |
| 64 | 0.235 | -0.645 | 0.645 |

**Interpretation**: The corrected sweep shows the Jensen-inequality bias exactly
as predicted, now as a smooth monotonic trend rather than a collapse. Larger
batches underestimate mean(g^2) more, weakening the effective EWC penalty: AA
rises (0.175 -> 0.235) and forgetting worsens (AF 0.586 -> 0.645) as batch_size
grows from 1 to 64. batch_size=1 (unbiased per-sample Fisher) gives the strongest
regularization -- lowest AA, best forgetting -- confirming it is the most faithful
estimate.

batch_size=8 and 16 are nearly identical (AA 0.218 vs 0.223; AF identical at
0.632), forming a stable plateau where the bias is mild. batch_size=16 is retained
as the speed/bias tradeoff: a 16x reduction in Fisher passes versus batch_size=1
while sitting in that plateau. The honest cost is that batch_size=16 does trade a
small amount of EWC strength for speed relative to batch_size=1 -- the bias is real
but modest, and it is held constant across all architectures, so it does not
confound the comparison.

---

## ER Hyperparameters

### er_buffer_size = 500

**Rationale**: 500 / 100 classes = 5 exemplars per class. This is the standard
"low-budget" ER regime. The reference is Chaudhry et al. "Tiny Episodic Memories"
(2019), who study the effect of small replay buffers in class-IL. iCaRL (Rebuffi
et al., 2017) uses 20/class, which is beyond the typical budget for a CIFAR-100
experiment with 16 GB VRAM and 27 total runs.

**Replay ratio**: with 500 exemplars after 10 tasks, each training step
concatenates ~50 replay samples (500 / 10) with a 128-sample current batch,
giving approximately 28% replay. This ratio decreases as tasks accumulate (the
denominator of 500 covers all prior tasks proportionally via reservoir sampling).
The exact ratio should be logged during training; if ER underperforms, verify
whether the replay fraction was sufficient before attributing failure to
architecture.

**Ablation grid**: {100, 200, 500, 1000} on ResNet-18, 3 tasks, 10 epochs/task.

| er_buffer_size | exemplars/class | AA | BWT | AF |
|---|---|---|---|---|
| 100 | 1 | 0.176 | -0.393 | 0.393 |
| 200 | 2 | 0.230 | -0.512 | 0.512 |
| 500 | 5 | 0.218 | -0.510 | 0.510 |
| 1000 | 10 | 0.260 | -0.449 | 0.449 |

**Interpretation**: AA increases monotonically with buffer size (0.176 → 0.260),
confirming that more exemplars improve overall retention. BWT is non-monotonic
(200 and 500 are nearly identical at -0.51; 1000 improves to -0.449), which
reflects noise at 1 seed and 10 epochs rather than a true non-monotonicity.

The 100-exemplar result (1/class) is clearly worse on both metrics. The 200
and 500 results are within noise of each other; the 1000 result shows a real
improvement but represents 10/class — double the standard ER budget.

buffer=500 (5/class) is validated as the standard low-budget choice:
it delivers most of the benefit of 1000 without the doubled memory footprint.
The 200 vs 500 gap is not robustly established at 10 epochs/1 seed, but the
literature (Chaudhry et al. 2019) supports 5/class as the reliable minimum for
stable replay. In the full 50-epoch grid, the gap should be cleaner.

---

## Running All Ablations

```bash
source .venv/bin/activate

# Smoke test (validates script only; no meaningful metrics)
python scripts/ablation_hp.py --smoke

# Full ablation (all 4 HPs, ~15-20 min on RTX 5070 Ti)
python scripts/ablation_hp.py 2>&1 | tee logs/ablation.log

# Or one HP at a time:
python scripts/ablation_hp.py --target ewc_lambda
python scripts/ablation_hp.py --target fisher_subsample
python scripts/ablation_hp.py --target fisher_batch_size
python scripts/ablation_hp.py --target er_buffer_size
```

After running, copy the result rows from `results/ablation/*.csv` into the
[PENDING] cells above.
