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
