import numpy as np
import sklearn.neighbors
from deprecated import deprecated


@deprecated
def find_nns_naive(
    locs: np.ndarray, dist_fun: str = "euclidean", max_nn: int = 10, **kwargs
) -> np.ndarray:
    """
    Finds the max_nn nearest neighbors preceding in the ordering.

    The method is naivly implemented and will not perform well for large inputs.

    Parameters
    ----------
    locs
        an n x m array of ordered locations
    dist_fun
        a distance metric used in sklearn.neighbors.BallTree
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
