"""Single entry point: train models, generate plots, print summary."""

import os
import torch
from train import train_model, evaluate_model
from visualize import (
    plot_feature_geometry, plot_phase_diagram,
    plot_dimensionality, plot_interference,
)

torch.manual_seed(42)

N_FEATURES, N_HIDDEN, IMPORTANCE_DECAY = 10, 5, 0.9
SPARSITIES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
N_STEPS = 20000
N_FEATURES_2D, N_HIDDEN_2D = 5, 2
SPARSITIES_2D = [0.0, 0.7, 0.9, 0.99]


def main():
    os.makedirs("results", exist_ok=True)

    print(f"Training {len(SPARSITIES)} models (n={N_FEATURES}, m={N_HIDDEN})...\n")
    all_results = {}
    for s in SPARSITIES:
        print(f"Sparsity {s:.2f}:")
        model = train_model(n_features=N_FEATURES, n_hidden=N_HIDDEN, sparsity=s,
                            n_steps=N_STEPS, importance_decay=IMPORTANCE_DECAY,
                            verbose=True)
        all_results[s] = evaluate_model(model, s)

    print(f"\nTraining {len(SPARSITIES_2D)} geometry models "
          f"(n={N_FEATURES_2D}, m={N_HIDDEN_2D})...\n")
    models_2d = {}
    for s in SPARSITIES_2D:
        print(f"Sparsity {s:.2f}:")
        model = train_model(n_features=N_FEATURES_2D, n_hidden=N_HIDDEN_2D,
                            sparsity=s, n_steps=N_STEPS,
                            importance_decay=IMPORTANCE_DECAY, verbose=True)
        models_2d[s] = evaluate_model(model, s)

    print("\nGenerating plots...")
    plot_feature_geometry(models_2d, n_features=N_FEATURES_2D)
    plot_phase_diagram(all_results, n_features=N_FEATURES)
    plot_dimensionality(all_results, n_hidden=N_HIDDEN, n_features=N_FEATURES)
    plot_interference(all_results[0.0], all_results[0.99], 0.0, 0.99)

    print("\n" + "=" * 70)
    print(f"{'Sparsity':>10} | {'Features Repr':>14} | {'Avg Interf':>11} | {'Recon Loss':>11}")
    print("-" * 70)
    for s in SPARSITIES:
        r = all_results[s]
        print(f"{s:>10.2f} | {r['n_represented']:>14d} | "
              f"{r['avg_interference']:>11.4f} | {r['total_loss']:>11.4f}")
    print("=" * 70)
    print(f"\nBottleneck: {N_HIDDEN} dims, {N_FEATURES} features")
    print("Plots saved to results/")


if __name__ == "__main__":
    main()
