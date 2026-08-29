from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA


def _resolve_pca(model: Any):
    if hasattr(model, "transform") and hasattr(model, "inverse_transform"):
        return model
    if isinstance(model, dict) and "pca" in model:
        return model["pca"]
    raise TypeError("model must be a fitted sklearn PCA object or a dict containing 'pca'")


def make_model(X_train, n_components: int = 10):
    """Fit a PCA model on training data and return the fitted estimator."""
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    return {"pca": pca}


def transform_to_pcadata(model: Any, X):
    pca = _resolve_pca(model)
    return pca.transform(X)


def reconstruct_data(model: Any, X_projected):
    pca = _resolve_pca(model)
    return pca.inverse_transform(X_projected)
