# LAMPS

Model-Agnostic Feature Selection via LOCO-Guided Adaptive Minipatch Sampling.

LAMPS selects feature indices using adaptive minipatch sampling. It supports
custom prediction functions and grid-search hyperparameter tuning.

## Installation

After the first PyPI release:

```bash
python -m pip install lamps-fs
```

The distribution is named `lamps-fs`; import it as `lamps`.
Requires Python >=3.10 and NumPy >=1.25. NumPy is the only runtime dependency.
No R installation or experiment datasets are needed.

Before publication, developers can run `python -m pip install .` from the
[repository](https://github.com/liu-xuhui/lamps) root.

## Quick start

```python
import numpy as np
from lamps import lamps_select

rng = np.random.default_rng(100)
X = rng.normal(size=(80, 20))
y = 4 * X[:, 0] - 3 * X[:, 1] + rng.normal(size=80)

# A small demonstration budget; omit K and max_iter to use the defaults.
selected = lamps_select(X, y, K=300, max_iter=2, show_progress=False)
print(selected)  # Zero-based column indices into X.
```

`X` should be a numeric array of shape `(n_samples, n_features)` and `y` a
one-dimensional numeric response of length `n_samples`. The default model is
NumPy least-squares linear regression without an intercept. Center the data
or supply a custom model if you need an intercept.

## Tuning and detailed results

Pass a list for `n_ratio`, `m_ratio`, `fit_func`, or `delta` to search their
Cartesian product. LAMPS chooses the run with the lowest final-epoch
leave-one-out error and prints its hyperparameters.

```python
selected, results = lamps_select(
    X, y,
    n_ratio=[0.3, 0.4],
    delta=[0.7, 0.8],
    K=300,
    max_iter=2,
    return_complete_info=True,
    show_progress=False,
)
```

The default return value is an array of selected feature indices.
`return_complete_info=True` returns `(selected, results)`, where `results`
contains iteration-indexed diagnostics. `seed` controls the sampling randomness;
custom models must manage any randomness of their own.

For a custom model, pass a callable `fit_func(X_train, y_train, X_predict)`
that returns one prediction per row of `X_predict`. Install that model's
dependencies separately. A list of callables enables model tuning.

## Paper reproduction and license

The [GitHub repository](https://github.com/liu-xuhui/lamps) contains separate
paper experiment scripts and environment instructions. Those scripts, datasets,
and vendored comparison methods are not shipped in this package.

LAMPS is distributed under the MIT license, copyright 2026 LAMPS authors.
