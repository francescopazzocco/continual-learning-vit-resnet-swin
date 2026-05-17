"""Unit tests for the four core components identified as untested.

Run from code/ with:
    pytest tests/test_core.py -v

All tests run on CPU so no GPU is required.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

from src.metrics import compute_metrics


class TestComputeMetrics:
    def test_empty_matrix(self):
        m = compute_metrics(np.zeros((0, 0)))
        assert m == {"AA": 0.0, "BWT": 0.0, "AF": 0.0}

    def test_single_task(self):
        R = np.array([[0.8]])
        m = compute_metrics(R)
        assert m["AA"] == pytest.approx(0.8)
        assert m["BWT"] == 0.0
        assert m["AF"] == 0.0

    def test_perfect_memory(self):
        # Model that never forgets: diagonal stays constant in every row.
        R = np.array([
            [0.9, 0.0, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.7],
        ], dtype=float)
        m = compute_metrics(R)
        assert m["AA"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
        assert m["BWT"] == pytest.approx(0.0)   # R[T-1,j] == R[j,j] for all j
        assert m["AF"] == pytest.approx(0.0)    # max per column == diagonal

    def test_total_forgetting(self):
        # After task 1, task 0 accuracy drops to 0.
        R = np.array([
            [0.9, 0.0],
            [0.0, 0.8],
        ], dtype=float)
        m = compute_metrics(R)
        assert m["AA"] == pytest.approx((0.0 + 0.8) / 2)
        assert m["BWT"] == pytest.approx(0.0 - 0.9)   # R[1,0] - R[0,0]
        assert m["AF"]  == pytest.approx(0.9 - 0.0)   # max(col 0) - R[T-1,0]

    def test_aa_is_mean_of_last_row(self):
        rng = np.random.default_rng(0)
        R = np.tril(rng.random((5, 5)))
        m = compute_metrics(R)
        assert m["AA"] == pytest.approx(float(np.mean(R[4, :5])))

    def test_bwt_formula(self):
        rng = np.random.default_rng(1)
        R = np.tril(rng.random((4, 4)))
        m = compute_metrics(R)
        expected_bwt = float(np.mean(R[3, :3] - np.diag(R)[:3]))
        assert m["BWT"] == pytest.approx(expected_bwt)

    def test_af_formula(self):
        rng = np.random.default_rng(2)
        R = np.tril(rng.random((4, 4)))
        m = compute_metrics(R)
        forgetting = R[:4, :3].max(axis=0) - R[3, :3]
        assert m["AF"] == pytest.approx(float(np.mean(forgetting)))

    def test_bwt_positive_transfer(self):
        # If later training helps earlier tasks, BWT can be positive.
        R = np.array([
            [0.5, 0.0],
            [0.7, 0.8],
        ], dtype=float)
        m = compute_metrics(R)
        assert m["BWT"] == pytest.approx(0.7 - 0.5)


# ---------------------------------------------------------------------------
# ReservoirBuffer
# ---------------------------------------------------------------------------

from src.cl.er import ReservoirBuffer


class TestReservoirBuffer:
    def test_fill_phase_does_not_exceed_max_size(self):
        buf = ReservoirBuffer(max_size=10)
        buf.update(torch.zeros(15, 3), torch.zeros(15, dtype=torch.long))
        assert len(buf) == 10

    def test_empty_sample_returns_empty_tensors(self):
        buf = ReservoirBuffer(max_size=10)
        x, y = buf.sample(5)
        assert x.numel() == 0
        assert y.numel() == 0

    def test_sample_respects_requested_count(self):
        buf = ReservoirBuffer(max_size=20)
        buf.update(torch.zeros(20, 3), torch.arange(20))
        x, y = buf.sample(7)
        assert x.shape[0] == 7
        assert y.shape[0] == 7

    def test_sample_capped_at_buffer_size(self):
        buf = ReservoirBuffer(max_size=20)
        buf.update(torch.zeros(5, 3), torch.arange(5))
        x, y = buf.sample(100)
        assert x.shape[0] == 5

    def test_labels_match_inputs_during_fill(self):
        buf = ReservoirBuffer(max_size=50)
        x = torch.arange(50).float().unsqueeze(1)  # (50, 1)
        y = torch.arange(50)
        buf.update(x, y)
        # Every sample should be stored; label == value stored in x.
        for i in range(50):
            assert (buf._y[i] == buf._x[i, 0].long()).item()

    def test_reservoir_uniform_retention(self):
        # Feed max_size*20 samples labelled 0..N-1 in order.
        # After many updates, each slot should have been drawn from a
        # roughly uniform distribution over [0, N_total).
        # We use a chi-squared-style check: slot occupancy variance should
        # be much smaller than if only early samples were kept.
        torch.manual_seed(42)
        max_size = 100
        n_total  = max_size * 50   # 5000 samples
        buf = ReservoirBuffer(max_size=max_size)
        batch = 50
        for start in range(0, n_total, batch):
            ids = torch.arange(start, start + batch, dtype=torch.float32).unsqueeze(1)
            labels = torch.arange(start, start + batch)
            buf.update(ids, labels)

        # Under uniform sampling the buffer should represent the full range:
        # - at least one recent sample (max > 80% of n_total)
        # - mean ≈ n_total/2 (no bias toward early or late samples)
        # The expected minimum of max_size draws from [0, n_total) is only
        # n_total/(max_size+1) ≈ 50, so we do not assert on min.
        labels_kept = buf._y[:max_size].float()
        assert labels_kept.max().item() > n_total * 0.80, (
            "Reservoir appears biased toward early samples"
        )
        assert labels_kept.mean().item() == pytest.approx(n_total / 2, rel=0.15), (
            "Mean retained label deviates too far from n_total/2"
        )


# ---------------------------------------------------------------------------
# linear_cka
# ---------------------------------------------------------------------------

from src.analysis.cka import linear_cka


class TestLinearCKA:
    def test_identical_inputs_return_one(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((64, 32)).astype(np.float32)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-5)

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            X = rng.standard_normal((64, 16)).astype(np.float32)
            Y = rng.standard_normal((64, 24)).astype(np.float32)
            val = linear_cka(X, Y)
            assert 0.0 <= val <= 1.0 + 1e-6, f"CKA={val} outside [0, 1]"

    def test_symmetry(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((64, 16)).astype(np.float32)
        Y = rng.standard_normal((64, 24)).astype(np.float32)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-5)

    def test_orthogonal_features_near_zero(self):
        # Two blocks of independent standard normals should have low CKA.
        rng = np.random.default_rng(3)
        X = rng.standard_normal((256, 32)).astype(np.float32)
        Y = rng.standard_normal((256, 32)).astype(np.float32)
        assert linear_cka(X, Y) < 0.3

    def test_scaled_input_invariant(self):
        # CKA is invariant to isotropic scaling of either matrix.
        rng = np.random.default_rng(4)
        X = rng.standard_normal((64, 16)).astype(np.float32)
        Y = rng.standard_normal((64, 16)).astype(np.float32)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(X * 5.0, Y), abs=1e-4)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(X, Y * 3.0), abs=1e-4)

    def test_degenerate_zero_matrix_returns_zero(self):
        X = np.zeros((32, 16), dtype=np.float32)
        Y = np.random.default_rng(5).standard_normal((32, 16)).astype(np.float32)
        # denom will be 0; code should return 0.0, not NaN/inf.
        assert linear_cka(X, Y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Model forward shapes  (CPU only)
# ---------------------------------------------------------------------------

from src.models.vit    import get_vit_small
from src.models.resnet import get_resnet18
from src.models.swin   import get_swin_tiny
from configs.default   import Config


_BATCH      = 2
_INPUT      = torch.zeros(_BATCH, 3, 32, 32)
_N_CLASSES  = Config().n_classes   # 100


class TestModelForwardShapes:
    @pytest.fixture(scope="class")
    def vit(self):
        return get_vit_small(_N_CLASSES).eval()

    @pytest.fixture(scope="class")
    def resnet(self):
        return get_resnet18(_N_CLASSES).eval()

    @pytest.fixture(scope="class")
    def swin(self):
        return get_swin_tiny(_N_CLASSES).eval()

    def test_vit_output_shape(self, vit):
        with torch.no_grad():
            out = vit(_INPUT)
        assert out.shape == (_BATCH, _N_CLASSES)

    def test_resnet_output_shape(self, resnet):
        with torch.no_grad():
            out = resnet(_INPUT)
        assert out.shape == (_BATCH, _N_CLASSES)

    def test_swin_output_shape(self, swin):
        with torch.no_grad():
            out = swin(_INPUT)
        assert out.shape == (_BATCH, _N_CLASSES)

    def test_vit_no_pretrained_weights(self):
        # Weights must be random (not pretrained). Verify by checking that
        # two independently constructed models have different parameters.
        m1 = get_vit_small(_N_CLASSES)
        m2 = get_vit_small(_N_CLASSES)
        p1 = next(m1.parameters())
        p2 = next(m2.parameters())
        assert not torch.allclose(p1, p2), "ViT weights look identical -- possible pretrained load"

    def test_resnet_no_pretrained_weights(self):
        m1 = get_resnet18(_N_CLASSES)
        m2 = get_resnet18(_N_CLASSES)
        p1 = next(m1.parameters())
        p2 = next(m2.parameters())
        assert not torch.allclose(p1, p2)

    def test_models_have_100_output_classes(self, vit, resnet, swin):
        for model in (vit, resnet, swin):
            with torch.no_grad():
                out = model(_INPUT)
            assert out.shape[-1] == 100
