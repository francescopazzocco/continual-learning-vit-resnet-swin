# ViT vs. ResNet under Continual Learning: Do CNN-era Method Rankings Generalize?

## Problem

A decade of continual learning (CL) research has validated methods — EWC, Experience Replay, and their variants — almost exclusively on CNN backbones (ResNet-18 in 62/81 online CL approaches surveyed by arXiv:2501.04897; ResNet-18 is similarly the de-facto backbone across broader CL benchmarks following Van de Ven et al. 2022). Vision Transformers are now the default backbone in industry and state-of-the-art vision systems, yet the implicit assumption that CL findings transfer across architectures is untested. ViTs differ structurally from CNNs in ways that directly affect how and where task-specific features are stored: global self-attention from the first block, qualitatively different feature reuse across layers (Park & Kim, ICLR 2022), and curved representation geometry (arXiv:2210.05742). If method rankings shift when the backbone changes, practitioners deploying CL on ViT-based systems are operating on invalidated assumptions.

## Evidence

- arXiv:2501.04897 (2025): systematic survey of 81 online CL approaches — ResNet-18 appears in 62/81; no comparable ViT-from-scratch study exists. Scope is online (single-pass) CL; ResNet-18 dominance is consistent with the broader CL literature.
- Van de Ven, Tuytelaars & Tolias (Nature MI, 2022; arXiv:1904.07734): the canonical CL scenario taxonomy; benchmark protocols (Split-CIFAR-100 class-IL) widely adopted by the community with ResNet-18 backbones.
- Park & Kim (ICLR 2022): MSAs act as low-pass filters, flatten loss landscapes, and produce qualitatively different cross-layer feature reuse vs. CNNs — structural reasons to expect different forgetting dynamics.
- arXiv:2210.05742: ViTs have locally curved representation spaces vs. CNNs' approximately linear input-output relationship.
- Existing ViT-CL papers (DyTox CVPR'22, LVT CVPR'22, L2P, DualPrompt, CODA-Prompt) either propose new ViT-specific architectures (DyTox, LVT) or rely on pretrained backbones (L2P, DualPrompt, CODA-Prompt) — none test whether standard CNN-era CL methods (EWC, ER) transfer to a ViT trained from scratch.
- ViTIL (arXiv:2112.06103, 2021): naively replacing CNN with ViT in class-incremental learning degrades performance — motivation for systematic study.

## Users

- **Primary**: Course instructors evaluating empirical originality and methodological rigor (grading context). Secondary audience: CL researchers who would encounter this at a workshop venue (e.g., CLVision @ CVPR).
- **Not for**: Practitioners needing a deployable CL system; users requiring pretrained or production-grade ViT models.

## Hypothesis

We believe **applying standard CNN-era CL methods (EWC, Experience Replay) to a ViT trained from scratch** will **reveal whether method rankings generalize across architectures** for **the CL research community**.

We'll know we're right when **AA / BWT / AF results show rank ordering differences (e.g., ER > EWC on ResNet but EWC ≥ ER on ViT) across 3–5 seeds with non-overlapping confidence intervals — or confirm that rankings transfer, which is itself a positive result**.

## Success Metrics

| Metric | Target | How measured |
|---|---|---|
| Reproducible ranking difference or equivalence | Non-overlapping CIs across 3–5 seeds | Per-condition AA / BWT / AF on Split-CIFAR-100 (class-IL) |
| Mechanistic evidence | Layer-wise CKA + weight-drift analysis covers all transformer block types | CKA similarity matrices + L2 weight drift per layer after each task |
| Baseline viability | ViT joint-training accuracy ≥ 55% on full CIFAR-100 | Pilot joint-training run before CL experiments |
| Report quality | Accepted as a 6-page IEEE/NeurIPS-style report | Instructor grading rubric |

## Scope

**MVP** — Two architectures (ViT-Small from scratch with conv stem, ResNet-18) × three conditions (vanilla fine-tuning, EWC, ER) × 3 seeds on class-incremental Split-CIFAR-100 (10 tasks × 10 classes), with layer-wise CKA similarity and weight-drift analysis as the mechanistic contribution.

**Out of scope**

- Pretrained ViT backbones (DINO, MAE, ImageNet-pretrained) — changes the research question from architecture to pretraining
- Prompt-based CL methods (L2P, DualPrompt, CODA-Prompt) — require pretrained weights by design
- Architectural expansion methods (DyTox, LVT, PNN) — modify model structure per task, orthogonal comparison
- MAS, SI, LwF, PathInt — regularization variants; EWC is the family representative
- Class-IL with task-ID at inference (task-incremental) — masks architectural differences; class-IL is primary protocol
- Self-supervised forgetting analysis (DINO/MAE attention maps under CL) — stretch goal only
- Hyperparameter search beyond pilot tuning — fixed HP across architectures after pilot

**Fallback**: If ViT joint-training accuracy is below 55% on CIFAR-100, switch dataset to Tiny ImageNet (64×64) without changing the research question.

## Delivery Milestones

| # | Milestone | Outcome | Status | Plan |
|---|---|---|---|---|
| 1 | Infrastructure + pilot | ViT-Small and ResNet-18 train on full CIFAR-100 (joint); confirm ViT ≥ 55% accuracy; data pipeline for Split-CIFAR-100 class-IL verified | **complete** (2026-05-16): ViT 63.70%, ResNet 63.86% — gate passed | `.claude/plans/vit-continual-learning.plan.md` |
| 2 | CL training runs | All 18–30 runs complete (2 arch × 3 conditions × 3–5 seeds × 10 tasks); AA / BWT / AF logged per task per seed | pending | — |
| 3 | Mechanistic analysis | CKA similarity matrices and L2 weight-drift plots produced per layer type (attention, MLP, LayerNorm, stem) after each task for both architectures | pending | — |
| 4 | Report | 6-page report with: intro/related work, experimental setup, quantitative results table, CKA/drift figures, discussion of ranking generalization, conclusion | pending | — |

## Open Questions

- [x] **Pilot accuracy gate**: ViT-Small reached 63.70% and ResNet-18 reached 63.86% on joint CIFAR-100 (2026-05-16). Gate passed; proceeding with CIFAR-100.
- [ ] **EWC Fisher subset size**: Computing full Fisher on 5,000 images per task is feasible but slow — validate that a 20% subsample gives stable importance estimates before full runs.
- [ ] **ER buffer size**: 200 vs. 500 exemplars per class-IL — 500 is more powerful but may dominate results; pick one before runs and justify in the report.
- [ ] **CKA implementation**: Between-task CKA requires storing layer activations — confirm memory budget on the 5070 Ti for ViT-Small with a full task's validation set.
- [ ] **Instructor approval**: Self-proposed topic requires approval before implementation begins.
- [ ] **Workshop submission**: If results are strong, CLVision @ CVPR is the natural venue — requires a different format than the course report; decide after seeing results.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ViT fails to reach viable accuracy from scratch on CIFAR-100 | Medium | High — invalidates forgetting analysis | Joint-training pilot gating Milestone 1; fallback to Tiny ImageNet |
| All methods collapse equally in class-IL (near-zero AA) | Low-Medium | Medium — reduces discriminability; ER should survive with buffer | ER with 500 exemplars anchors at least one non-collapsed condition; report collapse as a finding |
| Compute budget: 30 runs × multi-hour training | Low | Medium — 5070 Ti is sufficient; 32×32 images keep run times short | Estimate: ~25–30 total GPU-hours at 32×32; run overnight |
| Method rankings don't differ (null result) | Medium | Low — null result is publishable if well-powered and mechanistic analysis is present | Hypothesis is two-sided; confirmation of transfer is also a valid finding |
| Conv stem complicates "pure ViT" narrative | Low | Low — well-precedented (CCT, CvT); framed explicitly as tokenizer, not backbone | One-sentence methods footnote; cite CCT |

---
*Status: DRAFT — requirements only. Implementation planning pending via /plan.*
