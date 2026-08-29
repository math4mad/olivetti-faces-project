from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
_DATASET_PATH = _ROOT / "csv" / "scikit_fetch_olivetti_faces.csv.zip"


def _load_raw_matrix() -> np.ndarray:
    if not _DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset archive not found: {_DATASET_PATH}")

    with zipfile.ZipFile(_DATASET_PATH) as archive:
        if not archive.namelist():
            raise ValueError(f"Dataset archive is empty: {_DATASET_PATH}")
        with archive.open(archive.namelist()[0]) as raw_file:
            payload = raw_file.read().decode("utf-8")

    matrix = np.genfromtxt(io.StringIO(payload), delimiter=",", skip_header=1)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix.astype(np.float64)


def load_olivetti_faces(test_size: float = 0.2, random_state: int = 123):
    """Return train/test splits for the Olivetti faces dataset.

    The original Julia workflow uses a 0.8/0.2 split with label stratification,
    returning the tuple structure ((X_train, X_test), (y_train, y_test)).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    matrix = _load_raw_matrix()
    X = matrix[:, :-1]
    y = matrix[:, -1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return (X_train, X_test), (y_train, y_test)
