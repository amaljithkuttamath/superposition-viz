"""Training loop for the toy superposition model."""

import torch
from model import ToyModel, generate_data, feature_importance, compute_loss


def train_model(n_features=20, n_hidden=5, sparsity=0.9, n_steps=10000,
                batch_size=256, lr=1e-3, importance_decay=0.7,
                verbose=False):
    """Train a toy model at a given sparsity level."""
    model = ToyModel(n_features, n_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    importance = feature_importance(n_features, decay=importance_decay)
    for step in range(n_steps):
        x = generate_data(batch_size, n_features, sparsity)
        x_hat = model(x)
        loss = compute_loss(x, x_hat, importance)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and (step + 1) % 2000 == 0:
            print(f"  Step {step + 1:5d} | Loss: {loss.item():.4f}")
    return model


def evaluate_model(model, sparsity, n_samples=2048):
    """Evaluate reconstruction quality per feature."""
    model.eval()
    with torch.no_grad():
        x = generate_data(n_samples, model.n_features, sparsity)
        x_hat = model(x)
        mse_per_feature = (x - x_hat).pow(2).mean(dim=0)
        var_per_feature = x.var(dim=0).clamp(min=1e-8)
        feature_benefit = (1.0 - mse_per_feature / var_per_feature).clamp(0, 1)
        W = model.W.detach()
        gram = W.T @ W
        off_diag = gram - torch.diag(gram.diag())
    return {
        "mse_per_feature": mse_per_feature.numpy(),
        "feature_benefit": feature_benefit.numpy(),
        "gram_matrix": gram.numpy(),
        "avg_interference": off_diag.abs().mean().item(),
        "n_represented": int((feature_benefit > 0.5).sum().item()),
        "total_loss": mse_per_feature.mean().item(),
        "W": W.numpy(),
    }
