 re-evaluate with known task ID (separate head per task) to verify accuracy jumps from ~9% to ~50%+

[PAPER] task-IL baseline: 3 runs (vanilla only, one seed each for ViT/ResNet/Swin) -- proves
  class-IL collapse is protocol-driven, not an implementation bug; add a two-column comparison
  table (class-IL vs task-IL AA) in Section IV to anchor the near-chance numbers

[PAPER] per-layer drift figure: plot L2 drift grouped by layer type (stem, early blocks,
  late blocks, head) from existing results/features/ npz data -- add to scripts/plot.py;
  shows which layers EWC anchors most (mechanistic depth for Section IV-E)
