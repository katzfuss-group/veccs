from dataclasses import dataclass

import numpy as np
import scipy
import scipy.spatial.distance

from .legacy import find_nns_l2, find_nns_naive
from .maxmin_ancestor_cpp import maxmin_ancestor_cpp as _maxmin_ancestor_cpp
from .maxmin_cpp import maxmin_cpp as _maxmin_cpp
from .orderings2 import find_preceding_neighbors

__all__ = [
    "maxmin_cpp",
    "maxmin_pred_cpp",
    "find_closest_to_mean",
    "maxmin_naive",
    "find_nns_l2",
    "find_nns_naive",
]


def find_closest_to_mean(locs: np.ndarray) -> np.intp:
    """
    Finds in a location array the index of the location that is closest to the
    mean of the locations.

    The location array is a m by n array of m observations in an n-dimensional
    space.

    Parameters
    ----------
    locs
        2-d location array

    Returns
    -------
    np.intp
        index of the location closest to the mean.

    """
    avg = np.expand_dims(np.mean(locs, axis=0), 0)
    idx_min = np.argmin(scipy.spatial.distance.cdist(avg, locs))
    return idx_min


def maxmin_naive(dist: np.ndarray, first: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs min-max ordering

    The implementation is naive and will not perform well for large inputs.

    Parameters
    ----------
    dist
        distance matrix
    first
        Index of the observation that should be sorted first

    Returns
    -------
    np.ndarray
        The minmax ordering
    np.ndarray
        Array with the distances to the location preceding in ordering
    """

    n = dist.shape[0]
    ord = np.zeros(n, dtype=np.int64)
    ord[0] = first
    dists = np.zeros(n)
    dists[0] = np.nan
    idx = np.arange(n)

    for i in range(1, n):
        # find min dist for each not selected loccation
        mask = ~np.isin(idx, ord[:i])
        min_d = np.min(dist[mask, :][:, ord[:i]], axis=1)

        # find max idx among those
        idx_max = np.argmax(min_d)

        # record dist
        dists[i] = min_d[idx_max]

        # adjust idx for the prevous removed rows
        idx_max = idx[mask][idx_max]
        ord[i] = idx_max
    return ord, dists


def maxmin_cpp(locs: np.ndarray) -> np.ndarray:
    """
    Returns a maxmin ordering based on the Euclidean distance.

    Parameters
    ----------
    locs
        A m by n array of m observations in an n-dimensional space


    Returns
    -------
    np.ndarray
        Returns the indices of the permutation.

    Notes
    -----
    The implementation is based on the work of Schäfer et al. [1]_, Schäfer et
    al. [2]_. The code is an adaptation of [3]_.

    References
    ----------
    .. [1] Schäfer, F., Katzfuss, M. and Owhadi, H. Sparse Cholesky
        Factorization by Kullback--Leibler Minimization. SIAM Journal on
        Scientific Computing, 43(3), 2021. https://doi.org/10.1137/20M1336254
    .. [2] Schäfer, F., Sullivan, T.J. and Owhadi, H. Compression, Inversion,
        and Approximate PCA of Dense Kernel Matrices at Near-Linear
        Computational Complexity. Multiscale Modeling & Simulation, 19(12),
        2021. https://doi.org/10.1137/19M129526X
    .. [3] https://github.com/f-t-s/
           cholesky_by_KL_minimization/blob/f9a7d10932c422bde9f1fcfc950321c8c7b460a2/src/SortSparse.jl.

    """

    if not isinstance(locs, np.ndarray):
        raise TypeError("locs must be a numpy array")

    idx = _maxmin_cpp(locs)
    return np.array(idx)


def maxmin_pred_cpp(locs: np.ndarray, pred_locs: np.ndarray) -> np.ndarray:
    """
    Returns a maxmin ordering based on the Euclidean distance where the
    locations in locs are preceeding the locations in pred_locs.

    Parameters
    ----------
    locs
        A m by n array of m observations in an n-dimensional space

    pred_locs
        A k by n array of k observations in an n-dimensional space


    Returns
    -------
    np.ndarray
        Returns the indices of the permutation for the cocatenated array of locs
        and pred_locs, e.g., np.concatenate((locs, pred_locs), axis=0).

    Notes
    -----
    The implementation is based on C++ implementation provided by Myeongjong
    Kang which also can be found in [1]_.

    References
    ----------
    .. [1] https://github.com/katzfuss-group/variationalVecchia/blob/
           4ce03ddb53f3006b5cd1d1e3fe0268744e408039/external/maxmin_cpp/maxMin.cpp
    """

    if not isinstance(locs, np.ndarray):
        raise TypeError("locs must be a numpy array")

    if not isinstance(pred_locs, np.ndarray):
        raise TypeError("pred_locs must be a numpy array")

    locs_all = np.concatenate((locs, pred_locs), axis=0)
    npred = pred_locs.shape[0]

    first_idx = find_closest_to_mean(locs)

    ord_list = _maxmin_ancestor_cpp(locs_all, 1.0005, first_idx, npred)[0]
    return np.asarray(ord_list)


@dataclass
class AncestorOrdering:
    maximin_order: np.ndarray
    """
    The indices of the permutation for the cocatenated array of locs
        and pred_locs, e.g., np.concatenate((locs, pred_locs), axis=0).
    """
    sparsity: np.ndarray
    """
    sparsity index pairs for the inverse Cholesky factor
    """
    ancestor_set_reduced: np.ndarray
    """
    the reduced ancestor set (similar format as the sparsity index pairs)
    """


def maxmin_cpp_ancestor(
    locs: np.ndarray, pred_locs: np.ndarray, rho: float
) -> AncestorOrdering:
    """
    Returns a maxmin ordering based on the Euclidean distance where the
    locations in locs are preceeding the locations in pred_locs.

    Parameters
    ----------
    locs
        A m by n array of m observations in an n-dimensional space

    pred_locs
        A k by n array of k observations in an n-dimensional space

    rho
        A float value controling the radius of conditioning set and reduced
        ancestor set

    Returns
    -------
    AncestorOrdering
        An object holding the maximin ordering, the sparsity index pairs and the
        reduced ancestor set.

    Notes
    -----
    The implementation is based on C++ implementation provided by Myeongjong
    Kang which also can be found in [1]_.

    References
    ----------
    .. [1] https://github.com/katzfuss-group/variationalVecchia/blob/
           4ce03ddb53f3006b5cd1d1e3fe0268744e408039/external/maxmin_cpp/maxMin.cpp
    """

    if not isinstance(locs, np.ndarray):
        raise TypeError("locs must be a numpy array")

    if not isinstance(pred_locs, np.ndarray):
        raise TypeError("pred_locs must be a numpy array")

    locs_all = np.concatenate((locs, pred_locs), axis=0)
    npred = pred_locs.shape[0]

    first_idx = find_closest_to_mean(locs)

    orderObj = _maxmin_ancestor_cpp(locs_all, rho, first_idx, npred)
    ancestorApprox = np.array([orderObj[3], orderObj[2]])
    sparsity = ancestorApprox[:, orderObj[4]]
    ancestorApprox = ancestorApprox[:, ancestorApprox[1] >= 0]
    sparsity = sparsity[:, sparsity[1] >= 0]
    maxmin_order = np.asarray(orderObj[0])
    ordering = AncestorOrdering(
        maximin_order=maxmin_order,
        sparsity=sparsity,
        ancestor_set_reduced=ancestorApprox,
    )
    return ordering


def find_nns_l2_mf(locs_all: list[np.ndarray], max_nn: int = 10) -> np.ndarray:
    """
    Finds the max_nn nearest neighbors preceding in the ordering for
    every fidelity, plus the max_nn nearest neighbors in in the preceding
    fidelity

    Parameters
    ----------
    locs_all
        A list of observations in dimension p at different fidelities,
        where each fidelity has n_1, ..., n_R observations. You have to
        pass the locations for each fidelity in order from lower to
        highest fidelity.
    max_nn
        The max number of nearest neighbors considered within or between
        each fidelity (could consider different numbers of nearest neighbors
        within and between but that is not implemented now)

    Returns
    -------
    np.ndarray
        Returns the indices of the nearest neighbors, where -1 mean no nearest
        neighbors. Indices go from 0 to N = n_1 + n_2 + ... + n_R.
        The array is then of size N by 2 max_nn
    """

    R = len(locs_all)
    ns = np.zeros(R, dtype=int)
    NN_list = []
    NN_preb_list = []
    for r, locs in enumerate(locs_all):
        ns[r] = locs.shape[0]
        NNr, _ = find_preceding_neighbors(locs, np.arange(locs.shape[0]), max_nn)
        if r == 0:
            NN_preb = -np.ones((ns[r], max_nn), dtype=int)  # no nearest neighbors on
            # previous fidelity level for first fidelity level, use -1 for mask
        else:
            NNr = NNr + sum(ns[0:r])  # sum ns[0:r] because we need to
            # start counting all fidelities til this one
            # revert ruining which are -1
            NNr[NNr == sum(ns[0:r]) - 1] = -1
            # in previous line, kinda dumb hack but I guess it works
            distM = scipy.spatial.distance.cdist(locs_all[r], locs_all[r - 1])
            odrM = np.argsort(distM)
            NN_preb = odrM[:, :max_nn] + sum(ns[0 : r - 1])  # we need to start
            # counting all fidelities til last one.
        NN_list.append(NNr)
        NN_preb_list.append(NN_preb)

    NN = np.vstack(NN_list)
    NN_preb = np.vstack(NN_preb_list)
    NN_all = np.hstack((NN, NN_preb))

    return NN_all
