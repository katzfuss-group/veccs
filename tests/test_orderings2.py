import numpy as np
import pytest

from veccs.orderings2 import (
    find_prev_nearest_neighbors,
    maximin_ordering,
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


def test_maxmin_kdtree_heap(locations_2d):
    ord, min_dists = maximin_ordering(locations_2d, 0)

    correct_order = np.array([0, 4, 3, 2, 1])
    assert np.all(correct_order == ord)

    # correct min distances to prev points of the ordered points
    correct_dists = np.array([0.0, 2.3, 1.9, 1.1, 0.9])
    assert np.allclose(correct_dists, min_dists[ord])


def test_maxmin_kdtree_heap_with_groups(locations_2d):
    ord, min_dists = maximin_ordering(locations_2d, 0, groups=[[0, 1], [2, 3, 4]])

    correct_order = np.array([0, 1, 4, 2, 3])
    assert np.all(correct_order == ord)

    # correct min distances to prev points of the ordered points
    correct_dists = np.array([0.0, 1.0, 2.3, 1.1, 0.9])
    assert np.allclose(correct_dists, min_dists[ord])


def test_find_prev_nearest_neighbors_chunked(locations_2d):
    correct_order = np.array([0, 4, 3, 2, 1])
    locs_ord = locations_2d[correct_order, :]
    cond_set, dists = find_prev_nearest_neighbors(
        locs_ord, np.arange(correct_order.shape[0]), max_nn=2, chunk_size=2
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
