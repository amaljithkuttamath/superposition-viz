"""Tests for the toy superposition model."""

import torch
from model import ToyModel, generate_data, feature_importance, compute_loss
from train import train_model, evaluate_model


class TestToyModel:
    def test_output_shape_matches_input(self):
        model = ToyModel(n_features=10, n_hidden=3)
        x = torch.randn(32, 10).abs()
        assert model(x).shape == x.shape

    def test_various_sizes(self):
        for n, m in [(5, 2), (20, 5), (50, 10), (8, 8)]:
            model = ToyModel(n_features=n, n_hidden=m)
            assert model(torch.rand(16, n)).shape == (16, n)

    def test_single_sample(self):
        model = ToyModel(n_features=6, n_hidden=3)
        assert model(torch.rand(1, 6)).shape == (1, 6)


class TestDataGeneration:
    def test_sparsity_rate(self):
        for target in [0.0, 0.5, 0.9, 0.99]:
            data = generate_data(10000, 20, target)
            actual = (data == 0).float().mean().item()
            assert abs(actual - target) < 0.05

    def test_values_in_range(self):
        data = generate_data(1000, 10, 0.5)
        assert data.min() >= 0.0 and data.max() <= 1.0

    def test_shape(self):
        assert generate_data(64, 15, 0.8).shape == (64, 15)


class TestFeatureImportance:
    def test_decay(self):
        imp = feature_importance(5)
        assert len(imp) == 5
        for i in range(4):
            assert imp[i] > imp[i + 1]

    def test_first_value(self):
        assert abs(feature_importance(3)[0].item() - 1.0) < 1e-6


class TestLoss:
    def test_zero_loss_for_perfect_reconstruction(self):
        x = torch.rand(32, 10)
        assert compute_loss(x, x, feature_importance(10)).item() < 1e-6

    def test_positive_loss_for_bad_reconstruction(self):
        x = torch.rand(32, 10)
        assert compute_loss(x, torch.zeros_like(x), feature_importance(10)).item() > 0


class TestTraining:
    def test_no_bottleneck_low_loss(self):
        """When m >= n, the model can represent all features."""
        model = train_model(n_features=5, n_hidden=5, sparsity=0.0,
                            n_steps=5000, batch_size=256)
        result = evaluate_model(model, sparsity=0.0)
        assert result["total_loss"] < 0.02

    def test_evaluate_returns_expected_keys(self):
        model = train_model(n_features=8, n_hidden=3, sparsity=0.5, n_steps=500)
        result = evaluate_model(model, sparsity=0.5)
        expected = {"mse_per_feature", "feature_benefit", "gram_matrix",
                    "avg_interference", "n_represented", "total_loss", "W"}
        assert set(result.keys()) == expected
