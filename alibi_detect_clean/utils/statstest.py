import numpy as np
from typing import Callable, Tuple, Union

def fdr(p_val: np.ndarray, q_val: float) -> Tuple[int, Union[float, np.ndarray]]:
    """
    Checks the significance of univariate tests on each variable between 2 samples of
    multivariate data via the False Discovery Rate (FDR) correction of the p-values.

    Parameters
    ----------
    p_val
        p-values for each univariate test.
    q_val
        Acceptable q-value threshold.

    Returns
    -------
    Whether any of the p-values are significant after the FDR correction
    and the max threshold value or array of potential thresholds if no p-values
    are significant.
    """
    n = p_val.shape[0]
    i = np.arange(n) + 1
    p_sorted = np.sort(p_val)
    q_threshold = q_val * i / n
    below_threshold = p_sorted < q_threshold
    try:
        idx_threshold = np.where(below_threshold)[0].max()
    except ValueError:  # sorted p-values not below thresholds
        return int(below_threshold.any()), q_threshold
    return int(below_threshold.any()), q_threshold[idx_threshold]
