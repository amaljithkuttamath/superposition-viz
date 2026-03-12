"""Generate publication-quality plots showing superposition."""

import numpy as np
import matplotlib.pyplot as plt

STYLE = "seaborn-v0_8-whitegrid"
RESULTS_DIR = "results"


def set_style():
    try:
        plt.style.use(STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")


def plot_feature_geometry(models_2d: dict, n_features: int = 5):
    """Plot encoder weight vectors in 2D for different sparsity levels."""
    set_style()
    sparsities = sorted(models_2d.keys())
    n_plots = len(sparsities)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 0.9, n_features))

    for ax, s in zip(axes, sparsities):
        W = models_2d[s]["W"]  # (2, n_features)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        circle = plt.Circle((0, 0), 1, fill=False, color="gray",
                             linestyle="--", linewidth=0.8)
        ax.add_patch(circle)

        for i in range(n_features):
            ax.annotate(
                "", xy=(W[0, i], W[1, i]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=2),
            )
            ax.text(
                W[0, i] * 1.12, W[1, i] * 1.12, f"f{i}",
                ha="center", va="center", fontsize=9, color=colors[i],
                fontweight="bold",
            )

        ax.set_aspect("equal")
        ax.set_title(f"Sparsity = {s}", fontsize=12)
        ax.set_xlabel("Hidden dim 1")
        ax.set_ylabel("Hidden dim 2")

    fig.suptitle(
        "Feature Geometry in 2D Bottleneck", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/feature_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phase_diagram(all_results: dict, n_features: int):
    """Heatmap of reconstruction quality by feature and sparsity."""
    set_style()
    sparsities = sorted(all_results.keys())
    benefit_matrix = np.array(
        [all_results[s]["feature_benefit"] for s in sparsities]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        benefit_matrix, aspect="auto", cmap="magma",
        vmin=0, vmax=1, origin="lower",
        extent=[-0.5, n_features - 0.5, -0.5, len(sparsities) - 0.5],
    )
    ax.set_yticks(range(len(sparsities)))
    ax.set_yticklabels([f"{s:.2f}" for s in sparsities])
    ax.set_xlabel("Feature Index (decreasing importance)", fontsize=12)
    ax.set_ylabel("Sparsity", fontsize=12)
    ax.set_title("Superposition Phase Diagram", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, label="Reconstruction Quality")
    cbar.set_label("Feature Benefit (1 - MSE/Var)", fontsize=11)

    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/phase_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dimensionality(all_results: dict, n_hidden: int, n_features: int):
    """Line plot: effective features represented vs sparsity."""
    set_style()
    sparsities = sorted(all_results.keys())
    n_represented = [all_results[s]["n_represented"] for s in sparsities]
    total_benefit = [all_results[s]["feature_benefit"].sum() for s in sparsities]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sparsities, n_represented, "o-", color="#4C72B0", linewidth=2,
            markersize=8, markerfacecolor="white", markeredgewidth=2,
            label="Well-represented features (benefit > 0.5)")
    ax.plot(sparsities, total_benefit, "s--", color="#55A868", linewidth=2,
            markersize=7, markerfacecolor="white", markeredgewidth=2,
            label="Sum of all feature benefits")
    ax.axhline(y=n_hidden, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Bottleneck dims (m={n_hidden})")
    ax.set_xlabel("Sparsity", fontsize=12)
    ax.set_ylabel("Effective Features", fontsize=12)
    ax.set_title(
        "Effective Dimensionality vs Sparsity", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0, top=n_features + 0.5)

    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/dimensionality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_interference(results_low: dict, results_high: dict,
                      sparsity_low: float, sparsity_high: float):
    """Side-by-side gram matrices showing interference patterns."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vmax = max(
        np.abs(results_low["gram_matrix"]).max(),
        np.abs(results_high["gram_matrix"]).max(),
    )

    for ax, res, s in [
        (ax1, results_low, sparsity_low),
        (ax2, results_high, sparsity_high),
    ]:
        gram = res["gram_matrix"]
        im = ax.imshow(gram, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"W^T W  (sparsity = {s})", fontsize=12)
        ax.set_xlabel("Feature j")
        ax.set_ylabel("Feature i")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Feature Interference (Gram Matrix)", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/interference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
