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
- Mirzadeh et al. "Architecture Matters in Continual Learning" (arXiv:2202.00275, ICLR 2022): compares CNNs, ResNets, WideResNets, and ViTs on Split-CIFAR-100 and ImageNet-1K from scratch. Closest prior work, but differs from this study on three axes: (1) **protocol** — they use task-incremental (multi-head, task-ID known at inference), this study uses class-incremental (single head, no task-ID at inference), a strictly harder setting with different forgetting dynamics; (2) **no mechanistic analysis** — no CKA, no weight drift; (3) **no Swin Transformer**. Their key finding (simple CNNs outperform ResNets and ViTs in forgetting) was obtained under task-IL and cannot be assumed to hold under class-IL. This study is positioned as a class-IL extension with mechanistic depth and a third architecture. Must be cited and explicitly differentiated in the report's related work section.

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

**MVP** — Three architectures (ViT-Small from scratch with conv stem, ResNet-18, Swin-Tiny adapted for 32×32) × three conditions (vanilla fine-tuning, EWC, ER) × 3 seeds on class-incremental Split-CIFAR-100 (10 tasks × 10 classes) = 27 runs. Layer-wise CKA similarity and weight-drift analysis as the mechanistic contribution, now spanning a locality gradient: ResNet (fully local conv) → Swin (windowed attention) → ViT (global attention). Swin adds the critical middle point needed to test whether locality is causally responsible for forgetting resistance.

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
| 1 | Infrastructure + pilot | ViT-Small and ResNet-18 train on full CIFAR-100 (joint); confirm ViT ≥ 55% accuracy; data pipeline for Split-CIFAR-100 class-IL verified | **complete** (2026-05-16): ViT 63.70%, ResNet 63.86% — gate passed | `doc/plans/vit-continual-learning.plan.md` |
| 1b | Swin-Tiny pilot | Swin-Tiny (adapted for 32×32) trains on full CIFAR-100 (joint); confirm accuracy ≥ 55%; architecture added to data pipeline | **complete** (2026-05-16): Swin 57.34% — gate passed | `doc/plans/vit-continual-learning.plan.md` |
| 2 | CL training runs | All 27 runs complete (3 arch × 3 conditions × 3 seeds × 10 tasks); AA / BWT / AF logged per task per seed | pending | — |
| 3 | Mechanistic analysis | CKA similarity matrices and L2 weight-drift plots produced per layer type (windowed-attn, global-attn, MLP, LayerNorm, stem/conv) after each task for all three architectures | pending | — |
| 4 | Report | 6-page report with: intro/related work, experimental setup, quantitative results table, CKA/drift figures, discussion of ranking generalization, conclusion | pending | — |

## Open Questions

- [x] **Pilot accuracy gate**: ViT-Small reached 63.70% and ResNet-18 reached 63.86% on joint CIFAR-100 (2026-05-16). Gate passed; proceeding with CIFAR-100.
- [x] **EWC Fisher subset size**: Ablation (2026-05-16, scripts/ablation_hp.py) shows all subsample fractions 0.05–0.50 give identical BWT at lambda=1000. subsample=0.2 confirmed; see doc/hyperparameter_choices.md.
- [x] **ER buffer size**: Ablation (2026-05-16) shows 500 (5/class) delivers most of the 1000-exemplar benefit. buffer=500 confirmed; see doc/hyperparameter_choices.md.
- [ ] **CKA implementation**: Between-task CKA requires storing layer activations — confirm memory budget on the 5070 Ti for ViT-Small with a full task's validation set.
- [ ] **Instructor approval**: Self-proposed topic requires approval before implementation begins.
- [ ] **Workshop submission**: If results are strong, CLVision @ CVPR is the natural venue — requires a different format than the course report; decide after seeing results.
- [x] **Swin architecture for 32×32**: patch=4, window=4, 2 stages, embed_dim=192, depths=[2,6]. See doc/decision_log.md LOG-001–004.
- [x] **Swin layer naming for CKA/drift**: patch_embed, stages.0.{0,1}, patch_merging, stages.1.{0..5}, norm, head. See doc/decision_log.md LOG-005.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ViT fails to reach viable accuracy from scratch on CIFAR-100 | Medium | High — invalidates forgetting analysis | Joint-training pilot gating Milestone 1; fallback to Tiny ImageNet |
| All methods collapse equally in class-IL (near-zero AA) | Low-Medium | Medium — reduces discriminability; ER should survive with buffer | ER with 500 exemplars anchors at least one non-collapsed condition; report collapse as a finding |
| Compute budget: 30 runs × multi-hour training | Low | Medium — 5070 Ti is sufficient; 32×32 images keep run times short | Estimate: ~25–30 total GPU-hours at 32×32; run overnight |
| Method rankings don't differ (null result) | Medium | Low — null result is publishable if well-powered and mechanistic analysis is present | Hypothesis is two-sided; confirmation of transfer is also a valid finding |
| Conv stem complicates "pure ViT" narrative | Low | Low — well-precedented (CCT, CvT); framed explicitly as tokenizer, not backbone | One-sentence methods footnote; cite CCT |
| Swin-Tiny fails accuracy gate on CIFAR-100 32×32 | Medium | High — standard Swin window sizes (7×7) do not fit 32×32 images; requires architectural adaptation (smaller patches, smaller windows, fewer stages) | Adapt window size to 4×4 and patch size to 4×4; validate in milestone 1b joint pilot before CL runs; fallback to Tiny ImageNet applies to all three architectures equally |
| Swin adds scope / compute beyond course project budget | Low-Medium | Medium — 9 additional runs (~25–30% more GPU hours) | 27 runs still fits overnight on RTX 5070 Ti at 32×32; Swin-Tiny is lightweight (~28M params); course instructors reward ambition when executed cleanly |
| Report page budget tighter with three architectures | Medium | Low — 6-page limit forces tighter writing | Combine ViT and Swin CKA figures into a single panel; keep ResNet as the CNN baseline anchor; one results table covers all three |

---
*Status: DRAFT — requirements only. Implementation planning pending via /plan.*
