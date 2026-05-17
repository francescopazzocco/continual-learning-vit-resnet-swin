"""
ViT vs. ResNet under Continual Learning: Do CNN-era Method Rankings Generalize?
================================================================================

Empirical study comparing continual learning (CL) method rankings across
vision architectures (ViT-Small, ResNet-18, Swin-Tiny) on class-incremental
Split-CIFAR-100.  Three CL methods: vanilla fine-tuning, EWC, Experience Replay.
Mechanistic contribution: layer-wise CKA similarity + L2 weight drift analysis.

Course project for Neural Networks and Deep Learning, Univ. Padova 2025-2026.

Team:
    Francesco Pazzocco (lead) <francesco.pazzocco@studenti.unipd.it>
    Riccardo Corte <riccardo.corte@studenti.unipd.it>
    Alberto Casellato <alberto.casellato@studenti.unipd.it>
"""

from __future__ import annotations

__version__ = "0.1.0"
__requires_python__ = ">=3.10"
__dependencies__ = [
    "torch>=2.7.0",
    "torchvision>=0.22.0",
    "numpy>=1.26",
    "matplotlib>=3.9",
    "pandas>=2.2",
    "tqdm>=4.66",
]
