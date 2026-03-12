"""Toy model of superposition following Anthropic's 2022 paper."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    """Linear encoder/decoder with ReLU bottleneck and tied weights.

    Compresses n_features through n_hidden dimensions (n_hidden < n_features).
    The model must learn superposition to represent all features.
    """

    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.randn(n_hidden, n_features) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.W @ x.T + self.b.unsqueeze(1))  # (n_hidden, batch)
        x_hat = (self.W.T @ h).T  # (batch, n_features)
        return x_hat


def generate_data(
    batch_size: int, n_features: int, sparsity: float
) -> torch.Tensor:
    """Generate sparse synthetic data.

    Each feature is active with probability (1 - sparsity).
    When active, value is uniform [0, 1].
    """
    mask = (torch.rand(batch_size, n_features) > sparsity).float()
    values = torch.rand(batch_size, n_features)
    return mask * values


def feature_importance(n_features: int, decay: float = 0.7) -> torch.Tensor:
    """Importance weights: importance_i = decay^i."""
    return torch.tensor([decay**i for i in range(n_features)])


def compute_loss(
    x: torch.Tensor, x_hat: torch.Tensor, importance: torch.Tensor
) -> torch.Tensor:
    """Importance-weighted MSE reconstruction loss."""
    mse_per_feature = (x - x_hat).pow(2).mean(dim=0)
    return (importance * mse_per_feature).sum()
