"""AdaMP feature-selection package."""

from .core import (
    MPRegFeatureScore_indept,
    buildMP_indept,
    calc_K_list,
    extract_selected_features_nonoracle,
    get_loco,
    indept_sample_array,
    indept_weight_sample_epochtuned_legacy,
    last_epoch_key,
    linear_regression_fit,
    predictMP_indept,
    adamp_select,
)
from .simulation_functions import (
    SimuFriedmanAdditive,
    SimuFriedmanNonAdditive,
    SimuLinear,
    SimuLinear_old,
    SimuLinear_sparsity,
)

__all__ = [
    "MPRegFeatureScore_indept",
    "buildMP_indept",
    "calc_K_list",
    "extract_selected_features_nonoracle",
    "get_loco",
    "indept_sample_array",
    "indept_weight_sample_epochtuned_legacy",
    "last_epoch_key",
    "linear_regression_fit",
    "predictMP_indept",
    "adamp_select",
    "SimuFriedmanAdditive",
    "SimuFriedmanNonAdditive",
    "SimuLinear",
    "SimuLinear_old",
    "SimuLinear_sparsity",
]

__version__ = "0.1.0"
