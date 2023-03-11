import numpy as np
import pytest
import scipy.spatial.distance

from veccs.orderings import find_nns_l2, find_nns_naive, maxmin_cpp, maxmin_naive


@pytest.fixture
def locations_2d():
    return np.array([[0, 0], [-1, 0], [1.1, 0], [-1.9, 0], [2.3, 0]])


def test_maxmin_naive(locations_2d):
    dist_mat = scipy.spatial.distance.cdist(locations_2d, locations_2d)
    ord, _ = maxmin_naive(dist_mat, 0)
    correct_order = np.array([0, 4, 3, 2, 1])
    assert np.alltrue(correct_order == ord)


def test_maxmin_cpp(locations_2d):
    ord = maxmin_cpp(locations_2d)
    correct_order = np.array([0, 4, 3, 2, 1])
    assert np.alltrue(correct_order == ord)


def test_maxmin():
    gen = np.random.Generator(np.random.MT19937(10))
    locs = gen.uniform(0, 1, size=(10, 5))
    dist_mat = scipy.spatial.distance.cdist(locs, locs)
    ord0, _ = maxmin_naive(dist_mat, 0)

    ord1 = maxmin_cpp(locs)

    assert np.alltrue(ord0 == ord1)


def test_cond_set_naive(locations_2d):
    correct_order = np.array([0, 4, 3, 2, 1])
    locs_ord = locations_2d[correct_order, :]
    cond_set = find_nns_naive(locs_ord, max_nn=2)

    correct_result = np.array(
        [
            [-1, -1],
            [0, -1],
            [0, 1],
            [0, 1],
            [2, 0],
        ]
    )

    assert np.alltrue(correct_result == cond_set)


def test_cond_set_faiss(locations_2d):
    correct_order = np.array([0, 4, 3, 2, 1])
    locs_ord = locations_2d[correct_order, :]
    cond_set = find_nns_l2(locs_ord, max_nn=2)

    correct_result = np.array(
        [
            [-1, -1],
            [0, -1],
            [0, 1],
            [0, 1],
            [2, 0],
        ]
    )

    print(cond_set)

    assert np.alltrue(correct_result == cond_set)
