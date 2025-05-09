import warnings

import faiss
import numpy as np
import scipy.spatial
from deprecated import deprecated


@deprecated
def find_nns_l2(locs: np.ndarray, max_nn: int = 10) -> np.ndarray:
    """
    Finds the max_nn nearest neighbors preceding in the ordering.

    The distance between neighbors is based on the Euclidien distance.

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
