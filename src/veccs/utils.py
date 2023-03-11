import numpy as np


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a permutation.

    Parameters
    ----------
    perm
        1-d array of the permutation

    Returns
    -------
    np.ndarray
        1-d array of the inverse permutation

    """
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return inv_perm
