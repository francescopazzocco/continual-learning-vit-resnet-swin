"""
ViT vs. ResNet under Continual Learning
================================================================================

Empirical study comparing continual learning (CL) method rankings across
vision architectures (ViT-Small, ResNet-18, Swin-Tiny) on class-incremental
Split-CIFAR-100.  Three CL methods: vanilla fine-tuning, EWC, Experience Replay.
Mechanistic contribution: layer-wise CKA similarity + L2 weight drift analysis.

"""
