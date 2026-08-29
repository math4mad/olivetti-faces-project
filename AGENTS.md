# olivetti-faces-project

This project is being migrated from Julia to Python for face reconstruction.

## Python migration guidelines

- Use Python 3.10+.
- Replace Julia numerical workflows with `numpy` and `scipy`.
- Use `scikit-learn` to load and preprocess the Olivetti faces dataset.
- Use `matplotlib` for image and reconstruction visualizations.
- Preserve the existing reconstruction methodology, data splits, normalization, and evaluation metrics when translating Julia code.
- Prefer `venv` plus a `requirements.txt` file for dependency management.
- Keep reusable functionality in Python modules and place runnable experiments in clearly named scripts or notebooks.

## Suggested dependencies

```text
numpy
scipy
scikit-learn
matplotlib
jupyter
```
## workflows 
1. migrate  code to python  qmd  note  and test 
2. create a python branch 
3. create gh-pages,then pubish note as quarto website
## rules  
   style : qmd  note 
   plot  : publish standard 
   csv file as compressed , if you want to use , you can using uncompressed package



