# Decision Log

> Architecture Decision Record — chronological log of choices made and why.
> Append only: new decisions go at the end. Do not edit past entries.
> Each entry: what was chosen, why, and what the other direction would have cost.
> Open questions and upcoming work live in ROADMAP.md.
> Technical specs live in ARCHITECTURE.md.

---
Marked [ASK] if the decision was surfaced before acting, [LOG] if documented and continued.

---

## M1b — Swin architecture for 32x32

### [LOG] Number of stages: 2
Standard Swin-Tiny has 4 stages. With 32x32 input and patch size 4, the feature map
starts at 8x8. After one downsampling it's 4x4. After two it would be 2x2 — too small
to do anything meaningful with. So 2 stages is the only reasonable option.
Other direction: not available; this is a hard constraint from the input size.

### [LOG] Patch size: 4
Produces an 8x8 = 64 token sequence, the same as ViT-Small's conv stem. Keeps token
counts comparable between the two transformer architectures.
Other direction: patch size 2 → 16x16 = 256 tokens, much heavier; patch size 8 →
4x4 = 16 tokens, too few for meaningful windowed attention in stage 0.

### [LOG] Window size: 4
With an 8x8 feature map, window size 4 gives 4 real windows — actual windowed
attention in stage 0, which is the structural difference we want to study. After
the merge (4x4 map), window size 4 covers the whole map, so stage 1 degrades to
global attention. This locality gradient (windowed → global) is the mechanistic
point of interest.
Other direction: window size 2 → 16 windows of 2x2, very local; window size 8 →
would need padding, no windowing at all in stage 0.

### [ASK — RESOLVED] Embedding width: 96 → 192
Chose 96 initially (standard Swin-Tiny base, ~3M params). Pilot run returned 50.58%
— below the 55% gate, and the capacity gap vs ViT-Small (~11M) and ResNet-18 (~11M)
would have weakened the research claim regardless.
Raised to embed_dim=192, NUM_HEADS=[6,12] → ~12M params. All three architectures now
comparable in size; differences in forgetting can be attributed to architecture, not
model capacity.

### [LOG] Depths: [2, 6]
Mirrors Swin-Tiny's original depth imbalance ([2, 2, 6, 2] across 4 stages). The
heavy stage (6 blocks) is placed where the feature map is smallest and attention is
effectively global — more computation where the representation is most abstract.
Other direction: [2, 2] (symmetric) — simpler but loses the depth asymmetry that
characterizes Swin-Tiny's design.

### [LOG] Layer naming for CKA/drift analysis
Module structure chosen to give predictable names for the analysis scripts:
  patch_embed, stages.0.{0,1}, patch_merging, stages.1.{0..5}, norm, head
This resolves the PRD open question on Swin layer taxonomy.
Other direction: use torchvision's SwinTransformer class (naming: features.0,
features.1.0, etc.) — reliable implementation but naming is harder to control and
may change between torchvision versions.
