# Decision Log

> Architecture Decision Record — chronological log of choices made and why.
> Append only: new decisions go at the end. Do not edit past entries.
> Each entry: what was chosen, why, and what the other direction would have cost.
> Open questions and upcoming work live in ROADMAP.md.
> Technical specs live in ARCHITECTURE.md.

---
Marked [ASK-N] if the decision was surfaced before acting, [LOG-N] if documented and continued.

---

## M1b — Swin architecture for 32x32

### [LOG-001] Number of stages: 2
Standard Swin-Tiny has 4 stages. With 32x32 input and patch size 4, the feature map
starts at 8x8. After one downsampling it's 4x4. After two it would be 2x2 — too small
to do anything meaningful with. So 2 stages is the only reasonable option.
Other direction: not available; this is a hard constraint from the input size.

### [LOG-002] Patch size: 4
Produces an 8x8 = 64 token sequence, the same as ViT-Small's conv stem. Keeps token
counts comparable between the two transformer architectures.
Other direction: patch size 2 -> 16x16 = 256 tokens, much heavier; patch size 8 ->
4x4 = 16 tokens, too few for meaningful windowed attention in stage 0.

### [LOG-003] Window size: 4
With an 8x8 feature map, window size 4 gives 4 real windows — actual windowed
attention in stage 0, which is the structural difference we want to study. After
the merge (4x4 map), window size 4 covers the whole map, so stage 1 degrades to
global attention. This locality gradient (windowed -> global) is the mechanistic
point of interest.
Other direction: window size 2 -> 16 windows of 2x2, very local; window size 8 ->
would need padding, no windowing at all in stage 0.

### [ASK-001 — RESOLVED] Embedding width: 96 -> 192
Chose 96 initially (standard Swin-Tiny base, ~3M params). Pilot run returned 50.58%
— below the 55% gate, and the capacity gap vs ViT-Small (~11M) and ResNet-18 (~11M)
would have weakened the research claim regardless.
Raised to embed_dim=192, NUM_HEADS=[6,12] -> ~12M params. All three architectures now
comparable in size; differences in forgetting can be attributed to architecture, not
model capacity.

### [LOG-004] Depths: [2, 6]
Mirrors Swin-Tiny's original depth imbalance ([2, 2, 6, 2] across 4 stages). The
heavy stage (6 blocks) is placed where the feature map is smallest and attention is
effectively global — more computation where the representation is most abstract.
Other direction: [2, 2] (symmetric) — simpler but loses the depth asymmetry that
characterizes Swin-Tiny's design.

### [LOG-005] Layer naming for CKA/drift analysis
Module structure chosen to give predictable names for the analysis scripts:
  patch_embed, stages.0.{0,1}, patch_merging, stages.1.{0..5}, norm, head
This resolves the PRD open question on Swin layer taxonomy.
Other direction: use torchvision's SwinTransformer class (naming: features.0,
features.1.0, etc.) — reliable implementation but naming is harder to control and
may change between torchvision versions.

---

## Shared training hyperparameters

### [LOG-006] Standard CIFAR-100 training configuration
lr=0.1, momentum=0.9, wd=5e-4, batch_size=128, epochs=200 (joint pilot).
These are the canonical SGD settings for CIFAR from He et al. "Deep Residual
Learning" (2016), carried forward by virtually all CIFAR-100 benchmarks.
lr=0.1 with cosine annealing is also the default in Loshchilov & Hutter (2017).
wd=5e-4 is the value reported in He et al. "Delving Deep into Rectifiers" (2015).
Batch 128 fits ViT-Small (384-dim, 6 blocks) in 16 GB VRAM with BF16 AMP.
Other direction: lr=0.01 is safe but slower to converge; batch 256 would require
gradient accumulation or memory budget trade-off with the CL buffer.

### [LOG-007] epochs_per_task = 50
Total CL training budget: 500 epochs (10 tasks x 50 epochs), versus 200 epochs
for joint training. The 2.5x reduction is deliberate: each task's training set
is 1/10 the size of the joint set, so 50 epochs on 5000 samples approximates
the per-sample exposure of 200 epochs on 50000 samples. Fewer epochs also
reduces the risk of intra-task overfitting that would inflate within-task accuracy
and artificially suppress BWT.
Other direction: 100 epochs/task (1000 total) would better match joint exposure
but doubles wall time for the 27-run grid; 20 epochs risks underfitting per task.

### [LOG-008] RandAugment: num_ops=2, magnitude=9
Cubuk et al. "RandAugment" (2020) report num_ops=2, magnitude=9 as the best
single configuration across CIFAR-10/100 and ImageNet — no per-dataset search
needed. This matches the configuration used by DeiT (Touvron et al., 2021) for
training ViTs from scratch on CIFAR, which is the closest prior work.
Other direction: stronger augmentation (m=15, CutMix) could push ViT accuracy
higher but risks label-assignment complications for the class-IL protocol
(CutMix is explicitly excluded per project constraints; Mixup is banned).

---

## EWC hyperparameters

### [LOG-009] ewc_lambda = 1000
Kirkpatrick et al. (2017) report lambda=1000 on permuted MNIST; this value has
become the de-facto EWC baseline and is used in Mirzadeh et al. "Architecture
Matters" (ICLR 2022) for CIFAR experiments -- the closest prior work to ours.
Using the same value makes our results directly comparable.
Caveat: ViT Fisher magnitudes are likely orders of magnitude smaller than ResNet
(Park & Kim, ICLR 2022 -- MSAs flatten the loss landscape), so lambda=1000 may
be effectively weaker for ViT. This is a known confound documented in the failure
insights memory; it should be reported rather than corrected.
Other direction: lambda=5000 provides stronger protection but risks plasticity
loss on the first few tasks; lambda=100 is too weak to regularize effectively.
Ablation: see results/ablation/ewc_lambda.csv (run scripts/ablation_hp.py).

### [LOG-010] fisher_subsample = 0.2
20% of the task training set (approx. 1000 images per task for CIFAR-100 tasks
of 5000 images). Full Fisher on 5000 images per task x 10 tasks x 3 seeds x
2-3 architectures is prohibitive within the overnight compute budget.
1000-image subsamples are standard in EWC ablations at this dataset scale.
Other direction: 5% (250 images) likely yields noisy importance estimates;
50% (2500) halves Fisher speed for marginal quality gain.
Ablation: see results/ablation/fisher_subsample.csv.

### [LOG-011] fisher_batch_size = 16
Per-sample Fisher (batch_size=1) is mathematically correct: each sample's
gradient is squared individually giving mean(g_i^2). With batch_size=B,
param.grad holds the mean gradient over the batch, so param.grad^2 * B
approximates B * (mean(g))^2. By Jensen's inequality, (mean(g))^2 <= mean(g^2),
so larger batches systematically underestimate Fisher. However, relative
importance ordering across parameters is preserved -- EWC only needs this
ordering to be consistent, not the absolute scale.
batch_size=16 is a 16x speedup over per-sample (63 vs 1000 backward passes
per task). batch_size=1 was not chosen because Fisher estimation already accounts
for approximately 20-30% of per-task wall time at subsample=0.2.
Other direction: batch_size=1 for exact mean(g^2); batch_size=64 for 64x speedup
but greater underestimation. Ablation confirms the choice is not sensitive.
Ablation: see results/ablation/fisher_batch_size.csv.

---

## ER hyperparameters

### [LOG-012] er_buffer_size = 500
500 exemplars = 5 per class for 100 classes. The "5 per class" regime is the
standard low-budget ER protocol: iCaRL (Rebuffi et al., 2017) uses 20/class,
but 5/class is used in low-memory CL benchmarks (e.g., Chaudhry et al., 2019,
Tiny-Episodic Memory experiments). At 500 exemplars, each training step
concatenates ~50 replay samples (500/10 tasks) with a 128-sample current batch,
giving approximately 28% replay ratio -- enough to anchor earlier task features.
200 exemplars (2/class) risks insufficient coverage per class; 1000 (10/class)
doubles replay storage and may trivially dominate the result, reducing the
discriminability of the architecture comparison.
Other direction: 200 exemplars would stress-test ER's lower bound; ablation
confirms the 500 vs 200 gap in BWT before locking the grid value.

---

## EWC ablation correction

### [LOG-013] ewc_lambda ablation invalidated and re-run; lambda=1000 retained
The ablation originally cited under LOG-009 (and tabulated in
doc/hyperparameter_choices.md) was found to be invalid. On the pre-fix code path
the EWC penalty diverged: at lambda=1000 the task-1 train loss went to inf then
nan on the first step (results/ablation/ewc_lambda/1000_0/train_log.csv), so
every high-lambda cell collapsed to near-chance metrics (AA approx 0.033 = 1/30
for the 3-task ablation). The original writeup rationalized this as a "short-run
artifact" that would resolve at 50 epochs -- an interpretation that directly
contradicts the raw train_log (nan weights do not recover with more epochs).
That interpretation was wrong and has been removed.

After the gradient-stability fix (grad clip; commit b806d40), the ewc_lambda
sweep was re-run on the stabilized code. The corrected results are monotonic and
non-degenerate: AA decreases and forgetting decreases as lambda rises
(lambda=100: AA 0.234, AF 0.650; lambda=5000: AA 0.204, AF 0.606) -- a smooth
stability-plasticity tradeoff with no phase transition.

Decision: lambda=1000 is retained, now on honest grounds. It is the fixed
literature-standard value (Kirkpatrick 2017; Mirzadeh 2022), and the corrected
ablation confirms it is non-pathological and that the comparison is insensitive
to lambda across 100-5000. No retraining is required: the M2 grid runs already
used lambda=1000 and show no divergence.

Note: the fisher_subsample and fisher_batch_size sweeps in
hyperparameter_choices.md ran under the same pre-fix regime (EWC at lambda=1000)
and are likewise invalid; both are marked [INVALID -- PENDING RE-RUN] and must be
regenerated on the stabilized code before being cited.

Other direction: tuning lambda to maximize AA would pick lambda=100, the weakest
regularization regime -- rejected because it confounds a method-comparison study
and effectively removes the EWC effect being measured.

### [LOG-014] fisher_subsample and fisher_batch_size re-run; both choices stand
Following LOG-013, the two Fisher sweeps flagged as invalid were re-run on the
stabilized code. Both are now non-degenerate (no nan; sane AA around 0.18-0.23).

fisher_subsample: AA is flat across 0.05-0.50 (0.220-0.226) with only a marginal
AF improvement at larger fractions -- a genuine low-sensitivity result. 0.2 is
retained: indistinguishable from 0.5 in AA at 2.5x lower cost.

fisher_batch_size: the Jensen-inequality bias appears as predicted, as a smooth
monotonic trend -- larger batch underestimates the Fisher more, weakening
effective EWC (AA 0.175 -> 0.235, AF 0.586 -> 0.645 from B=1 to B=64). B=8 and
B=16 form a stable plateau. 16 is retained as the speed/bias tradeoff (16x fewer
passes than B=1); the modest bias is held constant across architectures and does
not confound the comparison.

No retraining of the grid is implied: these sweeps only inform Fisher-estimation
settings, which the M2 grid already used (subsample=0.2, batch_size=16).
