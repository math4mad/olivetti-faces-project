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
        recovered = reconstructed[i].reshape(64, 64)
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

    print(f"saved sample plots to {out_dir}")


if __name__ == "__main__":
    main()
