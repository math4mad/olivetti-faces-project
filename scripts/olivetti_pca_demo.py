from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from olivetti_faces.data import load_olivetti_faces
from olivetti_faces.model import make_model, reconstruct_data, transform_to_pcadata


def evaluate_component_sizes(X, component_sizes):
    scores = []
    for n_components in component_sizes:
        model = make_model(X, n_components=n_components)
        projected = transform_to_pcadata(model, X[:100])
        reconstructed = reconstruct_data(model, projected)
        mae = np.mean(np.abs(X[:100] - reconstructed))
        scores.append((n_components, mae))
    return scores


def main() -> None:
    (X_train, _), (_, _) = load_olivetti_faces(test_size=0.2, random_state=123)
    model = make_model(X_train, n_components=10)
    sample = X_train[:10]
    projected = transform_to_pcadata(model, sample)
    reconstructed = reconstruct_data(model, projected)

    print(f"train shape: {X_train.shape}")
    print(f"projected shape: {projected.shape}")
    print(f"reconstructed shape: {reconstructed.shape}")
    print(f"mean absolute error: {np.mean(np.abs(sample - reconstructed)):.4f}")

    out_dir = Path(__file__).resolve().parents[1] / "imgs"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        original = sample[i].reshape(64, 64)
        ax.imshow(original, cmap="gray")
        ax.set_title(f"Original {i + 1}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "olivetti_original_faces.png", dpi=150, bbox_inches="tight")

    fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes2.flat):
        recovered = reconstructed[i].reshape(64, 64)
        ax.imshow(recovered, cmap="gray")
        ax.set_title(f"Rebuilt {i + 1}")
        ax.axis("off")
    fig2.tight_layout()
    fig2.savefig(out_dir / "olivetti_reconstructed_faces.png", dpi=150, bbox_inches="tight")

    face_idx = 0
    original_face = X_train[face_idx].reshape(64, 64)
    one_model = make_model(X_train, n_components=1)
    one_projected = transform_to_pcadata(one_model, X_train[face_idx:face_idx + 1])
    one_reconstructed = reconstruct_data(one_model, one_projected)[0].reshape(64, 64)
    ten_reconstructed = reconstructed[0].reshape(64, 64)

    fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
    axes3[0].imshow(original_face, cmap="gray")
    axes3[0].set_title("Original face")
    axes3[0].axis("off")

    axes3[1].imshow(one_reconstructed, cmap="gray")
    axes3[1].set_title("1-component reconstruction")
    axes3[1].axis("off")

    axes3[2].imshow(ten_reconstructed, cmap="gray")
    axes3[2].set_title("10-component reconstruction")
    axes3[2].axis("off")

    fig3.tight_layout()
    fig3.savefig(out_dir / "reconstructed_face.png", dpi=150, bbox_inches="tight")

    component_sizes = [1, 2, 3, 5, 10, 20, 50, 100]
    scores = evaluate_component_sizes(X_train, component_sizes)
    comps = np.array([c for c, _ in scores], dtype=float)
    maes = np.array([m for _, m in scores], dtype=float)

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(comps, maes, marker="o", linewidth=2)
    ax4.set_xlabel("Number of PCA components")
    ax4.set_ylabel("Mean absolute error")
    ax4.set_title("PCA reconstruction accuracy by model size")
    ax4.grid(True, linestyle="--", alpha=0.4)
    fig4.tight_layout()
    fig4.savefig(out_dir / "model_accuracy_comparison.png", dpi=150, bbox_inches="tight")

    print(f"saved sample plots to {out_dir}")
    for n, mae in scores:
        print(f"  {n} components -> MAE {mae:.6f}")


if __name__ == "__main__":
    main()
