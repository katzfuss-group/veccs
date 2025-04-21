import itertools as it
import timeit
from typing import Callable

import numpy as np

from veccs import orderings, orderings2


class MeanStd:
    mean: float
    std: float

    def __init__(self, timings: list[float]):
        self.mean = np.mean(timings).item()
        self.std = np.std(timings).item()


def time_fn(fn: Callable):
    return timeit.repeat(fn, repeat=7, number=5)


def setup_and_time(n: int, num_nbrs: int) -> tuple[MeanStd, MeanStd, MeanStd, MeanStd]:
    points = np.linspace(0, 100, 2 * n).reshape(-1, 2)

    old_maxmin_times = time_fn(lambda: orderings.maxmin_cpp(points))
    new_maxmin_times = time_fn(lambda: orderings2.maximin_ordering(points, 0))

    points = points[orderings2.maximin_ordering(points, 0)]

    old_nbr_times = time_fn(lambda: orderings.find_nns_l2(points, num_nbrs))
    new_nbr_times = time_fn(
        lambda: orderings2.find_prev_nearest_neighbors(points, np.arange(n), num_nbrs)
    )

    return (
        MeanStd(old_maxmin_times),
        MeanStd(new_maxmin_times),
        MeanStd(old_nbr_times),
        MeanStd(new_nbr_times),
    )


def main():
    ns = [100, 300, 500, 1000, 3000, 5000]
    nns = [10, 20, 30, 50]

    print(
        "size,neighbors,old_maxmin_mean,old_maxmin_std,new_maxmin_mean,new_maxmin_std,old_nbrs_mean,old_nbrs_std,new_nbrs_mean,new_nbrs_std"
    )
    for n, nn in it.product(ns, nns):
        old_maxmin, new_maxmin, old_nbrs, new_nbrs = setup_and_time(n, nn)
        print(
            f"{n},{nn},{old_maxmin.mean},{old_maxmin.std},{new_maxmin.mean},{new_maxmin.std},{old_nbrs.mean},{old_nbrs.std},{new_nbrs.mean},{new_nbrs.std}"
        )


if __name__ == "__main__":
    main()
