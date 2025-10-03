"""
    Extra functions useful for Distns
"""
from ngboost.distns import RegressionDistn
import numpy as np

# pylint: disable=too-few-public-methods


def SurvivalDistnClass(Dist: RegressionDistn):
    """
    Creates a new dist class from a given dist. The new class has its implemented scores

    Parameters:
        Dist (RegressionDistn): a Regression distribution with censored scores implemented.

    Output:
        SurvivalDistn class, this is only used for Survival regression
    """

    class SurvivalDistn(Dist):
        # Stores the original distribution for pickling purposes
        _basedist = Dist
        scores = (
            Dist.censored_scores
        )  # set to the censored versions of the scores implemented for dist

        def fit(Y):
            """
            Parameters:
                Y : a object with keys {time, event}, each containing an array
            """
            return Dist.fit(Y["Time"])

    return SurvivalDistn

# ----------- for Multivariate distributions ----------- #

def cholesky_factor(lower_triangle_vals:np.ndarray, p:int) -> np.ndarray:
    """
    Args:
        lower_triangle_values: numpy array, shaped as the number of lower triangular
                        elements, number of observations.
                        The values ordered according to np.tril_indices(p).

        p: int, dimension of the multivariate distn

    Returns:
        Nxpxp numpy array, with the lower triangle filled in. The diagonal is exponentiated.

    """
    _, n_data = lower_triangle_vals.shape

    if not isinstance(lower_triangle_vals, np.ndarray):
        lower_triangle_vals = np.array(lower_triangle_vals)

    L = np.zeros((n_data, p, p))
    for par_ind, (k, l) in enumerate(zip(*np.tril_indices(p))):
        if k == l:
            # Add a small number to avoid singular matrices.
            L[:, k, l] = np.exp(lower_triangle_vals[par_ind, :]) + 1e-6
        else:
            L[:, k, l] = lower_triangle_vals[par_ind, :]
    return L

def get_tril_idxs(p):
    tril_indices = np.tril_indices(p)
    mask_diag = tril_indices[0] == tril_indices[1]

    off_diags = np.where(np.invert(mask_diag))[0]
    diags = np.where(mask_diag)[0]

    return tril_indices, diags, off_diags