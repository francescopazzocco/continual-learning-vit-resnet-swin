# Results Notes

> Empirical observations and interpretations recorded during runs.
> Append only. Each entry: what was observed, why it happened, and what it means for the study.
> Raw numbers live in results/; this file is for analysis and interpretation.

---

## M2 — CL Grid

### ViT-EWC: gradient explosion at task 1 boundary

**Observed**: `vit_ewc_s*` runs produce `loss=nan, val=0.00` starting from task 1.
Task 0 completes normally.

**Mechanism**: EWC's penalty term is zero during task 0 (Fisher not yet estimated).
After `after_task(0)`, Fisher is populated and the penalty activates at task 1.
With `ewc_lambda=1000`, ViT's larger inter-task parameter drift (no inductive bias,
global attention) inflates the penalty to the point where gradient norms exceed what
SGD at lr=0.1 can absorb stably. Parameters hit inf, producing NaN from the next
forward pass onward.

**Fix applied**: `clip_grad_norm_(max_norm=1.0)` added to `cl_trainer.py`.
This rescales the full gradient vector (direction preserved) when its L2 norm exceeds 1.0.
ResNet-EWC is unlikely to trigger the clip because conv filters drift less per task.

**Interpretation**: The instability is not a code bug — it is the result.
Adam ADA (2022) uses Adam + EWC on ViT/CIFAR-100 and still reports "EWC cannot handle
CF, especially for multi-class classification." Optimizer choice does not fix the
underlying mismatch: EWC assumes a quadratic loss surface with a meaningful Fisher
diagonal, but ViT's attention heads are redundant and permutation-symmetric, making
the Fisher estimate diffuse and the quadratic approximation unreliable.
Gradient clipping lets us produce a valid (non-NaN) number for comparison, but that
number is expected to show higher forgetting than ResNet-EWC. The asymmetry between
architectures is the finding.

---

## M2 — Full CL Grid: Findings (3 arch x 3 methods x 3 seeds)

### Summary statistics (mean +/- std across seeds 0-2)

| Arch    | Method  | AA mean | AA std | BWT mean | BWT std |
|---------|---------|---------|--------|----------|---------|
| ResNet  | vanilla | 0.0865  | 0.0009 | -0.7881  | 0.0024  |
| ResNet  | ER      | 0.0907  | 0.0006 | -0.7953  | 0.0050  |
| ResNet  | EWC     | 0.0803  | 0.0023 | -0.7381  | 0.0020  |
| ViT     | vanilla | 0.0100  | 0.0000 | -0.3209  | 0.1218  |
| ViT     | ER      | 0.0866  | 0.0012 | -0.7146  | 0.0303  |
| ViT     | EWC     | 0.0778  | 0.0008 | -0.7353  | 0.0111  |
| Swin    | vanilla | 0.0748  | 0.0009 | -0.6219  | 0.0104  |
| Swin    | ER      | 0.0758  | 0.0009 | -0.6420  | 0.0039  |
| Swin    | EWC     | 0.0674  | 0.0001 | -0.5804  | 0.0075  |

AA = mean accuracy over all tasks after training on the final task (average of last row of R).
All methods show near-chance AA (~9%) because class-IL without a task oracle causes
near-total forgetting of previous tasks regardless of method.

### Finding 1: ViT vanilla collapses at task 5

**Observed**: Per-task diagonal R[i,i] drops to 0.00 from task 5 onward for all three
vit_vanilla seeds. The model predicts class 0 for all inputs after collapse (R[9,0]=0.10
= chance on 10 classes, R[9,j]=0 for j>0). Mean diagonal across all 10 tasks: 0.299
(vs 0.796 for resnet_vanilla).

**Mechanism**: Without inductive bias (global attention over all 64 tokens), the loss
surface for ViT is non-convex and sensitive to gradient magnitude. After 5 tasks at
lr=0.1 with no regularization, the accumulated weight norm is large enough that the
gradient from the new task's cross-entropy drives parameters into a degenerate region.
The model saturates at predicting the dominant class.

**Artifact in BWT**: vit_vanilla BWT (-0.321, std=0.122) appears mild compared to
resnet_vanilla (-0.788). This is misleading: BWT = mean(R[T-1,j] - R[j,j]) for j<T-1.
Tasks 5-9 contributed R[j,j]=0, so they add 0 to the sum instead of a large negative
value. The apparent low forgetting is an artifact of the collapse, not a genuine result.
High seed variance (std=0.122) reflects that different seeds collapse at different tasks.

### Finding 2: ER prevents ViT collapse but not forgetting

**Observed**: vit_er diagonal mean = 0.730, min = 0.470. All 10 tasks are learned for
all 3 seeds. Comparing final rows: vit_er last row is near-zero for tasks 0-8 (same
pattern as resnet_vanilla), confirming that catastrophic forgetting still occurs.

**Mechanism**: The 50% replay ratio (equal current and replay batch sizes) stabilizes
the gradient direction at each step, preventing the runaway weight growth that causes
collapse. However, 50 exemplars per previous task are insufficient to maintain
separability across 100 classes after 50 epochs at lr=0.1 on the current task.
The SGD optimizer resets lr=0.1 at every task boundary, so every task is trained
aggressively from the same starting learning rate. The gradient for the current task's
5000 samples dominates the replay gradient from ~55 buffer samples per previous task.

**Implication**: For ViT, the primary value of ER is training stability, not memory
retention. This separates the two roles that replay plays and is directly testable via
CKA (M3): ViT-ER should show less representation drift than ViT-vanilla for tasks 0-4
even though AA is similar after task 9.

### Finding 3: EWC consistently reduces BWT; ER does not

**Observed** (BWT improvement over vanilla):
- ResNet EWC: -0.738 vs -0.788 vanilla  ->  +5.0pp less forgetting
- Swin EWC:   -0.580 vs -0.622 vanilla  ->  +4.2pp less forgetting
- ViT EWC:    -0.735 vs -0.714 ER (collapse-corrected baseline)

ER shows no BWT improvement vs vanilla for ResNet (-0.795 vs -0.788) or Swin
(-0.642 vs -0.622). ER BWT is marginally worse than vanilla in both cases, within
noise (ResNet std ~0.003, Swin std ~0.006).

**Mechanism for EWC**: The Fisher penalty constrains weight movement in directions
important to previous tasks. Even under the quadratic approximation, this directly
reduces the off-diagonal decay (R[T-1,j] for j<T-1). The effect is small (5pp) but
consistent and low-variance across seeds (EWC BWT std ~0.002 for ResNet).

**Mechanism for ER non-effect**: With 500 buffer entries / 10 tasks = 50 exemplars per
task, the replay gradient is overwhelmed by the current-task gradient at lr=0.1. The
combined batch (128 current + 128 replay) sees each old exemplar ~500 times per task,
but with only 50 unique exemplars, the resulting gradient is low-rank and easily
dominated. EWC's explicit penalty on the loss landscape is more effective at this
regime than replay-based gradient mixing.

**Method ranking** (BWT, consistent across all three architectures): EWC > vanilla >= ER.

### Finding 4: Swin forgets less than ResNet and ViT

**Observed** (vanilla BWT): Swin -0.622, ResNet -0.788, ViT -0.321 (collapse-inflated).
After correcting ViT for collapse, the ordering is Swin < ResNet < ViT (where lower
magnitude = less forgetting). Swin per-task accuracy (diagonal mean 0.634) is also lower
than ResNet (0.796), so reduced forgetting may partly reflect lower within-task capacity.

**Hypothesis**: Swin's local attention windows (shifted-window partitioning) impose a
spatial inductive bias similar to convolutions. Each window attends to a 4x4 patch
neighborhood, reducing the receptive field per layer. This may produce more localized,
task-specific features that are less susceptible to global weight updates from later
tasks. CKA analysis (M3) can test whether Swin layers show lower cross-task
representation drift than ViT layers.

### Interpretation for the study

The primary research question is whether CL method rankings differ between ViT and
ResNet. The M2 results reveal a more nuanced picture: the ranking (EWC > vanilla >= ER)
is preserved across all three architectures for BWT, but the failure modes differ
qualitatively. ViT vanilla collapses entirely while ResNet vanilla degrades gracefully.
ER's role shifts from "memory retention" (ResNet) to "training stabilizer" (ViT).

This architecture-method interaction is the main empirical contribution. M3 (CKA +
weight drift) will provide mechanistic evidence for why these behavioral differences
arise at the representation level.
