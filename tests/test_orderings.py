import typing

import numpy as np
import pytest
import scipy.spatial.distance

from veccs.orderings import (
    find_nns_l2,
    find_nns_l2_mf,
    find_nns_naive,
    maxmin_cpp,
    maxmin_cpp_ancestor,
    maxmin_naive,
    maxmin_pred_cpp,
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


def test_maxmin_naive(locations_2d):
    dist_mat = scipy.spatial.distance.cdist(locations_2d, locations_2d)
    ord, _ = maxmin_naive(dist_mat, 0)
    correct_order = np.array([0, 4, 3, 2, 1])
    assert np.all(correct_order == ord)


def test_maxmin_cpp(locations_2d):
    ord = maxmin_cpp(locations_2d)
    correct_order = np.array([0, 4, 3, 2, 1])
    assert np.all(correct_order == ord)


def test_maxmin():
    gen = np.random.Generator(np.random.MT19937(10))
    locs = gen.uniform(0, 1, size=(10, 5))
    dist_mat = scipy.spatial.distance.cdist(locs, locs)
    ord0, _ = maxmin_naive(dist_mat, 0)

    ord1 = maxmin_cpp(locs)

    assert np.all(ord0 == ord1)


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

    assert np.all(correct_result == cond_set)


def test_cond_set_mf(locations_2d_mf):
    correct_order_lf = np.array([0, 1])
    correct_order_mf = np.array([2, 1, 0])
    correct_order_hf = np.array([0, 4, 3, 2, 1])
    locs_all = [
        locations_2d_mf[0][correct_order_lf, :],
        locations_2d_mf[1][correct_order_mf, :],
        locations_2d_mf[2][correct_order_hf, :],
    ]
    cond_set = find_nns_l2_mf(locs_all, max_nn=2)

    correct_result = np.array(
        [
            [-1, -1, -1, -1],
            [0, -1, -1, -1],
            [-1, -1, 0, 1],
            [2, -1, 1, 0],
            [2, 3, 0, 1],
            [-1, -1, 2, 4],
            [5, -1, 3, 2],
            [5, 6, 4, 2],
            [5, 6, 2, 3],
            [7, 5, 2, 4],
        ]
    )

    assert np.all(correct_result == cond_set)


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
    assert np.all(correct_result == cond_set)


def test_maxmin_pred_one_cpp(locations_2d):
    ord = maxmin_pred_cpp(locations_2d, locations_2d[:1, :] + 1)
    correct_order = np.array([0, 4, 3, 2, 1, 5])
    assert np.all(correct_order == ord)


def test_maxmin_pred_cpp(locations_2d):
    shift = np.array(
        [
            [0, 100.0],
            [0, 50.0],
            [0, 77.0],
            [0, 88.0],
            [0, 95.5],
        ]
    )
    ord = maxmin_pred_cpp(locations_2d, locations_2d + shift)
    correct_order = np.array([0, 4, 3, 2, 1, 5, 6, 7, 8, 9])
    assert np.all(correct_order == ord)


def test_maxmin_cpp_ancestor(locations_2d):
    shift = np.array(
        [
            [0, 100.0],
            [0, 50.0],
            [0, 77.0],
            [0, 88.0],
            [0, 95.5],
        ]
    )
    ret_obj = maxmin_cpp_ancestor(locations_2d, locations_2d + shift, 100.0)
    ord = ret_obj.maximin_order
    correct_order = np.array([0, 4, 3, 2, 1, 5, 6, 7, 8, 9])
    assert np.all(correct_order == ord)
    assert np.all(ret_obj.sparsity == ret_obj.ancestor_set_reduced)


@typing.no_type_check
def test_typechecks_cpp() -> None:
    """
    Test type checks for the numpy arguments in the cpp function wrappers.

    Type checks are disabled for this function to test the runtime type checks.
    """
    with pytest.raises(TypeError):
        maxmin_cpp(1.0)

    with pytest.raises(TypeError):
        maxmin_pred_cpp(1.0, np.array([[1.0]]))

    with pytest.raises(TypeError):
        maxmin_pred_cpp(np.array([[1.0]]), 1.0)

    with pytest.raises(TypeError):
        maxmin_cpp_ancestor(1.0, np.array([[1.0]]), 1.005)

    with pytest.raises(TypeError):
        maxmin_cpp_ancestor(np.array([[1.0]]), 1.0, 1.005)
