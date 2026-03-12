# Visualizing Superposition in Neural Networks

Neural networks can represent more features than they have dimensions by encoding features in **superposition**, a phenomenon that emerges when features are sparse. This project reproduces the key result from Anthropic's [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) paper and generates visualizations of the phase transition.

## Why it matters

If features live in superposition, you cannot read them off individual neurons. Understanding when and how superposition occurs is foundational to mechanistic interpretability.

## What this does

Trains a toy model (linear encoder/decoder with ReLU bottleneck and tied weights) that compresses sparse high-dimensional input through a narrow bottleneck. The model must reconstruct its input, so it learns to pack features into the available dimensions. As sparsity increases, the model transitions from representing only the most important features to representing all of them via superposition.

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

Results are saved to `results/`.

## Plots

**Feature Geometry** (`feature_geometry.png`): Weight vectors plotted in a 2D bottleneck. At low sparsity, only 2 features are represented. At high sparsity, all 5 features spread out across the plane.

**Phase Diagram** (`phase_diagram.png`): Heatmap of reconstruction quality by feature index and sparsity level. Reproduces the phase transition from Figure 2 of the paper. Low sparsity = only important features survive. High sparsity = superposition activates.

**Dimensionality** (`dimensionality.png`): Number of features effectively represented vs. sparsity. Shows the step where the model begins representing more features than bottleneck dimensions.

**Interference** (`interference.png`): Gram matrix (W^T W) comparing low and high sparsity. Superposition introduces small off-diagonal interference between features.

## Paper

Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html
