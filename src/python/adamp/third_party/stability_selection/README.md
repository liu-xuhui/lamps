# Vendored: stability-selection

This directory contains a vendored copy of the `stability-selection` package,
used by `src/python/adamp/validation_methods.py` to provide the Stability
Selection and CPSS baselines in the paper comparisons. It is **not** part of the
LAMPS method and is **not** shipped in the `lamps-fs` PyPI distribution.

| | |
|---|---|
| Upstream | https://github.com/scikit-learn-contrib/stability-selection |
| Version taken | `master` branch, retrieved 2026-09-20 |
| License | BSD-3-Clause — see [LICENSE](LICENSE) |
| Copyright | Copyright (c) 2018, Thomas Huijskens. All rights reserved. |

The package is not published on PyPI and the upstream repository is no longer
actively maintained, which is why it is vendored here rather than declared as a
dependency in `env/requirements.txt`.

## Local modifications

Two import statements were changed so the code runs against modern
scikit-learn. No algorithmic or numerical behavior was modified.

1. `stability_selection/stability_selection.py`

   `sklearn.externals.joblib` was removed in scikit-learn 0.23, so the import
   now uses the standalone `joblib` package:

   ```diff
   -from sklearn.externals.joblib import Parallel, delayed
   +import joblib as jb
   +from joblib import Parallel, delayed
   ```

   (The `jb` alias is unused and could be dropped.)

2. `stability_selection/randomized_lasso.py`

   `sklearn.linear_model.base` was renamed to the private
   `sklearn.linear_model._base` in scikit-learn 0.22:

   ```diff
   -from sklearn.linear_model.base import _preprocess_data
   +from sklearn.linear_model._base import _preprocess_data
   ```

Apart from these two lines, the files are identical to upstream `master` as of
the retrieval date above.
