import numpy as np
import pytest

from veccs.orderings2 import (
    farthest_first_ordering,
    preceding_neighbors,
    reorder_farthest_first_with_neighbors,
)


@pytest.fixture
def locations_2d():
    return np.array([[0, 0], [-1, 0], [1.1, 0], [-1.9, 0], [2.3, 0]])


@pytest.fixture
def locations_2d_mf():
    locs_lf = np.array([[-1, -1], [1.1, 1.1]])
    locs_mf = np.array([[-1, -1], [1.1, 1.1], [0, 0]])
    locs_hf = np.array([[0, 0], [-1, 0], [1.1, 0], [-1.9, 0], [2.3, 0]])
    locs_all = [locs_lf, locs_mf, locs_hf]
    return locs_all


def test_farthest_first_ordering(locations_2d):
    ord, min_dists = farthest_first_ordering(locations_2d, 0)

    correct_order = np.array([0, 4, 3, 2, 1])
    assert np.all(correct_order == ord)

    # correct min distances to prev points of the ordered points
    correct_dists = np.array([0.0, 2.3, 1.9, 1.1, 0.9])
    assert np.allclose(correct_dists, min_dists[ord])


def test_farthest_first_ordering_with_groups(locations_2d):
    ord, min_dists = farthest_first_ordering(
        locations_2d, 0, partition=[[0, 1], [2, 3, 4]]
    )

    correct_order = np.array([0, 1, 4, 2, 3])
    assert np.all(correct_order == ord)

    # correct min distances to prev points of the ordered points
    correct_dists = np.array([0.0, 1.0, 2.3, 1.1, 0.9])
    assert np.allclose(correct_dists, min_dists[ord])


def test_find_prev_nearest_neighbors_chunked(locations_2d):
    correct_order = np.array([0, 4, 3, 2, 1])
    locs_ord = locations_2d[correct_order, :]
    cond_set, dists = preceding_neighbors(
        locs_ord,
        np.arange(correct_order.shape[0]),
        num_neighbors=2,
        rebuild_frequency=2,
    )

    correct_result = np.array(
        [
            [-1, -1],
            [0, -1],
            [0, 1],
            [0, 1],
            [2, 0],
        ]
    )

    assert np.all(correct_result == cond_set)

    # set uninitialized distances to -1
    print(dists)
    dists[cond_set == -1] = -1.0

    correct_dists = np.array(
        [
            [-1.0, -1.0],
            [2.3, -1.0],
            [1.9, 4.2],
            [1.1, 1.2],
            [0.9, 1.0],
        ]
    )
    assert np.allclose(correct_dists, dists)


def test_reorder_farthest_first_with_neighbors(locations_2d):
    res = reorder_farthest_first_with_neighbors(locations_2d, num_neighbors=2)
    correct_order = np.array([0, 4, 3, 2, 1])
    perm = np.argsort(correct_order)

    assert np.all(perm == res.inverse_permutation)

    ord, dists = farthest_first_ordering(locations_2d)
    locations_2d = locations_2d[ord]
    dists = dists[ord]
    assert np.allclose(res.separation_distances, dists)
    assert np.allclose(res.coordinates, locations_2d)

    nei, ndists = preceding_neighbors(locations_2d, np.arange(len(ord)), 2)
    assert np.allclose(res.neighbor_indices, nei)
    assert np.allclose(res.neighbor_distances, ndists)
