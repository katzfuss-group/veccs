import numpy as np
import pytest

from veccs.legacy import find_nns_l2, find_nns_naive, has_legacy_support


@pytest.fixture
def locations_2d():
    return np.array([[0, 0], [-1, 0], [1.1, 0], [-1.9, 0], [2.3, 0]])


@pytest.mark.skipif(
    not has_legacy_support(),
    reason="Legacy support not available.",
)
@pytest.mark.filterwarnings("ignore:.*find_nns_l2:DeprecationWarning")
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


@pytest.mark.skipif(
    not has_legacy_support(),
    reason="Legacy support not available.",
)
@pytest.mark.filterwarnings("ignore:.*find_nns_naive:DeprecationWarning")
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
