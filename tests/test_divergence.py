"""
tests/test_divergence.py
========================
Unit tests for JS-divergence utility functions.

Run with:
    pytest tests/test_divergence.py -v
"""

import numpy as np
import pytest
import torch
from ambedkar.utils.divergence import js_divergence_distributions, js_divergence_scalars


class TestJSDivergenceDistributions:

    def test_identical_distributions_zero(self):
        p = torch.softmax(torch.randn(100), dim=0)
        assert js_divergence_distributions(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_symmetric(self):
        p = torch.softmax(torch.randn(50), dim=0)
        q = torch.softmax(torch.randn(50), dim=0)
        assert js_divergence_distributions(p, q) == pytest.approx(
            js_divergence_distributions(q, p), rel=1e-4
        )

    def test_bounded(self):
        for _ in range(20):
            p = torch.softmax(torch.randn(50), dim=0)
            q = torch.softmax(torch.randn(50), dim=0)
            val = js_divergence_distributions(p, q)
            assert 0.0 <= val <= 1.0 + 1e-5  # JSD ∈ [0, log2] when using log base 2; ≤1 with base-e

    def test_returns_scalar_float(self):
        p = torch.softmax(torch.randn(20), dim=0)
        q = torch.softmax(torch.randn(20), dim=0)
        val = js_divergence_distributions(p, q)
        assert isinstance(val, float)

    def test_uniform_vs_one_hot(self):
        vocab = 10
        uniform = torch.ones(vocab) / vocab
        one_hot = torch.zeros(vocab)
        one_hot[0] = 1.0
        val = js_divergence_distributions(uniform, one_hot)
        assert val > 0.0

    def test_numerical_stability_near_zero_probs(self):
        """Distributions with near-zero entries should not produce NaN/Inf."""
        p = torch.tensor([0.999] + [1e-9] * 99)
        p = p / p.sum()
        q = torch.tensor([1e-9] + [0.999] + [1e-9] * 98)
        q = q / q.sum()
        val = js_divergence_distributions(p, q)
        assert not np.isnan(val)
        assert not np.isinf(val)


class TestJSDivergenceScalars:

    def test_returns_tensor(self):
        vocab = 30
        p = torch.softmax(torch.randn(5, vocab), dim=-1)   # [k, vocab]
        q = torch.softmax(torch.randn(5, vocab), dim=-1)
        result = js_divergence_scalars(p, q)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5,)

    def test_all_non_negative(self):
        vocab = 50
        p = torch.softmax(torch.randn(8, vocab), dim=-1)
        q = torch.softmax(torch.randn(8, vocab), dim=-1)
        result = js_divergence_scalars(p, q)
        assert (result >= 0).all()

    def test_identical_zero(self):
        vocab = 30
        p = torch.softmax(torch.randn(5, vocab), dim=-1)
        result = js_divergence_scalars(p, p)
        assert result.abs().max() < 1e-4
