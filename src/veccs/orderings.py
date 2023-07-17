import warnings
from collections.abc import Callable
from dataclasses import dataclass

import faiss
import numpy as np
import scipy
import scipy.spatial.distance
import sklearn.neighbors

from .maxmin_ancestor_cpp import maxmin_ancestor_cpp as _maxmin_ancestor_cpp
from .maxmin_cpp import maxmin_cpp as _maxmin_cpp


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


def maxmin_naive(dist: np.ndarray, first: np.intp) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs min-max ordering

    The implementation is naive and will not perform well for large inputs.

    Parameters
    ----------
    dist
        distrance matrix
    first
        Index of the observation that should be sorted first

    Returns
    -------
    np.ndarray
        The minmax ordering
    np.ndarray
        Array with the distrances to the location preceding in ordering
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


def find_nns_naive(
    locs: np.ndarray, dist_fun: Callable | str = "euclidean", max_nn: int = 10, **kwargs
) -> np.ndarray:
    """
    Finds the max_nn nearest neighbors preceding in the ordering.

    The method is naivly implemented and will not perform well for large inputs.

    Parameters
    ----------
    locs
        an n x m array of ordered locations
    dist_fun
        a distrance function
    max_nn
        number of nearest neighbours
    kwargs
        supplied dist_func

    Returns
    -------
    np.ndarray
        Returns an n x max_nn array holding the indices of the nearest neighbors
        preceding in the ordering where -1 indicates missing neighbors.
    """

    n = locs.shape[0]
    nns = np.zeros((n, max_nn), dtype=np.int64) - 1
    for i in range(1, n):
        nn = sklearn.neighbors.BallTree(locs[:i], metric=dist_fun, **kwargs)
        k = np.min(min(i, max_nn))
        nn_res = nn.query(locs[[i], :], k=k, return_distance=False)
        nns[i, :k] = nn_res
    return nns


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

    idx = _maxmin_cpp(locs)
    return np.array(idx)


def find_nns_l2(locs: np.ndarray, max_nn: int = 10) -> np.ndarray:
    """
    Finds the max_nn nearest neighbors preceding in the ordering.

    The distrance between neighbors is based on the Euclidien distrance.

    This code is copied from https://github.com/katzfuss-group/BaTraMaSpa_py/
    blob/d75974961317a5b1e30d6f2fcc14862e1cb0535b/NNarray.py and adjusted to fit
    the different imports. Also, compared to the original code, first column of
    the array returned is removed which was pointing to the element itself.

    Parameters
    ----------
    locs
        an n x m array of ordered locations
    max_nn
        number of nearest neighbours

    Returns
    -------
    np.ndarray
        Returns an n x max_nn array holding the indices of the nearest neighbors
        preceding in the ordering where -1 indicates missing neighbors.
    """
    n, d = locs.shape
    NN = -np.ones((n, max_nn + 1), dtype=int)
    mult = 2
    maxVal = min(max_nn * mult + 1, n)
    distM = scipy.spatial.distance.cdist(locs[:maxVal, :], locs[:maxVal, :])
    odrM = np.argsort(distM)
    for i in range(maxVal):
        NNrow = odrM[i, :]
        NNrow = NNrow[NNrow <= i]
        NNlen = min(NNrow.shape[0], max_nn + 1)
        NN[i, :NNlen] = NNrow[:NNlen]
    queryIdx = np.arange(maxVal, n)
    mSearch = max_nn
    while queryIdx.size > 0:
        maxIdx = queryIdx.max()
        mSearch = min(maxIdx + 1, 2 * mSearch)
        if n < 1e5:
            index = faiss.IndexFlatL2(d)
        else:
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, min(maxIdx + 1, 1024))
            index.train(locs[: maxIdx + 1, :])
            index.nprobe = min(maxIdx + 1, 256)
        index.add(locs[: maxIdx + 1, :])
        _, NNsub = index.search(locs[queryIdx, :], int(mSearch))
        lessThanI = NNsub <= queryIdx[:, None]
        numLessThanI = lessThanI.sum(1)
        idxLessThanI = np.nonzero(np.greater_equal(numLessThanI, max_nn + 1))[0]
        for i in idxLessThanI:
            NN[queryIdx[i]] = NNsub[i, lessThanI[i, :]][: max_nn + 1]
            if NN[queryIdx[i], 0] != queryIdx[i]:
                try:
                    idx = np.nonzero(NN[queryIdx[i]] == queryIdx[i])[0][0]
                    NN[queryIdx[i], idx] = NN[queryIdx[i], 0]
                    NN[queryIdx[i], 0] = queryIdx[i]
                except IndexError:
                    NN[queryIdx[i], 0] = queryIdx[i]
        queryIdx = np.delete(queryIdx, idxLessThanI, 0)

    if any(NN[:, 0] != np.arange(n)):
        warnings.warn("There are very close locations and NN[:, 0] != np.arange(n)\n")
    return NN.astype(np.int64)[:, 1:]


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
    locs_all = np.concatenate((locs, pred_locs), axis=0)
    npred = pred_locs.shape[0]

    first_idx = find_closest_to_mean(locs)

    orderObj = _maxmin_ancestor_cpp(locs_all, rho, first_idx, npred)
    ancestorApprox = np.array([orderObj[3], orderObj[2]])
    sparsity = ancestorApprox[:, orderObj[4]]
    ancestorApprox = ancestorApprox[:, ancestorApprox[1] >= 0]
    sparsity = sparsity[:, sparsity[1] >= 0]
    maxmin_order = orderObj[0]
    ordering = AncestorOrdering(
        maximin_order=maxmin_order,
        sparsity=sparsity,
        ancestor_set_reduced=ancestorApprox,
    )
    return ordering
