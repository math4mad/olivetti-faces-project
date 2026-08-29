import numpy as np

from olivetti_faces.data import load_olivetti_faces
from olivetti_faces.model import make_model, reconstruct_data, transform_to_pcadata


def test_load_olivetti_faces_shapes():
    (X_train, X_test), (y_train, y_test) = load_olivetti_faces(test_size=0.2, random_state=123)

    assert X_train.shape == (320, 4096)
    assert X_test.shape == (80, 4096)
    assert y_train.shape == (320,)
    assert y_test.shape == (80,)
    assert np.unique(y_train).size == 40


def test_pca_transform_and_reconstruct_round_trip():
    (X_train, _), (_, _) = load_olivetti_faces(test_size=0.2, random_state=123)
    model = make_model(X_train, n_components=10)

    sample = X_train[:5]
    projected = transform_to_pcadata(model, sample)
    reconstructed = reconstruct_data(model, projected)

    assert projected.shape == (5, 10)
    assert reconstructed.shape == sample.shape
    assert np.allclose(reconstructed, sample, atol=60.0)
