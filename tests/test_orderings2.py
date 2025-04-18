import numpy as np
import pytest

from veccs.orderings2 import (
    find_prev_nearest_neighbors,
    find_prev_nearest_neighbors_not_chunked,
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
    ord = maximin_ordering(locations_2d, 0)
    correct_order = np.array([0, 4, 3, 2, 1])
    assert np.all(correct_order == ord)


def test_find_prev_nearest_neighbors(locations_2d):
    correct_order = np.array([0, 4, 3, 2, 1])
    locs_ord = locations_2d[correct_order, :]
    cond_set = find_prev_nearest_neighbors_not_chunked(
        locs_ord, np.arange(correct_order.shape[0]), max_nn=2
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


def test_find_prev_nearest_neighbors_chunked(locations_2d):
    correct_order = np.array([0, 4, 3, 2, 1])
    locs_ord = locations_2d[correct_order, :]
    cond_set = find_prev_nearest_neighbors(
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
