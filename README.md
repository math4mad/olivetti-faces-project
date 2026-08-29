# Olivetti Faces Project

This project originally implemented a Julia-based PCA workflow for the Olivetti faces dataset. The current migration keeps the original reconstruction logic while porting the working pipeline to Python for easier experimentation and testing.

## Scope

The Python migration intentionally does not include the legacy model artifacts in the `models/` directory or the removed `Stat2-julia/` folder. These are excluded from the repository submission and are not part of the active Python branch.

## Original Julia workflow reference

### 1. Data processing

The Julia version read the dataset CSV and returned training and test splits with labels.

```julia
(Xtrain, Xtest), (ytrain, ytest) = load_olivetti_faces()
```

### 2. Train and save model

The original workflow built a PCA model for a selected dimension and saved it to a `.jlso` artifact.

```julia
function make_model(Xtr)
    return (dim) -> begin
        model = PCA(maxoutdim=dim)
        mach = machine(model, Xtr) |> fit!
        try
            JLSO.save("$(pwd())/models/of-model-$(dim)pcs.jlso", :pca => mach)
            @info "$(dim) dimension pca model saved"
        catch e
            @warn "$(e) has problem"
        end
    end
end
```

### 3. Transform and reconstruct

The Julia flow used `transform` for projection and `inverse_transform` for reconstruction.

```julia
pcaX = transform(mach, imgs)
Xr = inverse_transform(mach, pcaX)
```

## Python migration

The active Python implementation mirrors the same workflow:

- load the Olivetti CSV data
- split into train/test folds
- fit a PCA model on training samples
- project samples to the reduced space
- reconstruct data back to the original feature space

### Python entry points

```python
from olivetti_faces.data import load_olivetti_faces
from olivetti_faces.model import make_model, transform_to_pcadata, reconstruct_data

(X_train, X_test), (y_train, y_test) = load_olivetti_faces(test_size=0.2, random_state=123)
model = make_model(X_train, n_components=10)
projected = transform_to_pcadata(model, X_train[:5])
reconstructed = reconstruct_data(model, projected)
```

### Validation

The Python migration has a regression test covering the dataset split and PCA round-trip reconstruction.

```bash
python -m pytest -q tests/test_pipeline.py
```

## Notes

- Use Python 3.10+
- Prefer `numpy`, `scipy`, `scikit-learn`, and `matplotlib`
- Keep the project focused on the Olivetti faces PCA workflow
- Do not include generated model files or the legacy `Stat2-julia` material in final submission


##  trigger action 2

