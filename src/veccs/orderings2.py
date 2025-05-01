from collections.abc import Sequence
from heapq import heapify, heappop, heappush

import numpy as np
import scipy.spatial


def middle_of_bounding_box(points: np.ndarray) -> np.ndarray:
    """
    Compute the middle of the bounding box of a set of points.

    Parameters
    ----------
    points : np.ndarray, shape (n_points, D)
        The input point set.

    Returns
    -------
    middle : np.ndarray, shape (D,)
        The middle of the bounding box.
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return (min_coords + max_coords) / 2


def middle_of_convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of the convex hull of a set of points.

    Parameters
    ----------
    points : np.ndarray, shape (n_points, D)
        The input point set.

    Returns
    -------
    centroid : np.ndarray, shape (D,)
        The centroid of the convex hull.
    """
    hull = scipy.spatial.ConvexHull(points)
    hull_points = points[hull.vertices]
    return np.mean(hull_points, axis=0)


def maximin_ordering(
    points: np.ndarray,
    start_index: int | None = None,
    closest_to: np.ndarray | None = None,
    groups: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a max-min (farthest-first) ordering of a set of points in R^D.

    This is a greedy algorithm that picks the first point, then
    repeatedly picks the point that is farthest from the set of
    previously-selected points.

    Implementation is based on KD-trees and a priority queue (heap).
    If start_index and closest_to are both None, the first point is chosen closest to
    the centroid of the bounding box. If closest_to is not None and start_index is not
    None, a ValueError is raised.


    ----------
    points : np.ndarray, shape (n_points, D)
        The input point set.
    start_index
        The index of the first point to pick.

    closest_to : np.ndarray, shape (D,) | None
        If not None, the closest point in the set will be the first point in the
        returned ordering.

    groups : Sequence[Sequence[int]] | None
        If not None, the ordering will be done within each group, while taking the
        points from earlier groups into account. Each group is a sequence of indices
        into `points`. The ordering will be done separately for each group, and the
        results will be concatenated.


    Returns
    -------
    order : np.ndarray, shape (n_points,)
        A permutation of 0..n_points-1 giving the max-min ordering.

    min_dists : np.ndarray, shape (n_points,)
        The distance from each point to the closest previously ordered point.
    """
    n_pts = points.shape[0]

    if closest_to is not None and start_index is not None:
        raise ValueError("Cannot specify both start_index and closest_to.")

    if closest_to is None and start_index is None:
        closest_to = middle_of_bounding_box(points)

    if groups is None:
        groups = [list(range(n_pts))]
    else:
        # check that the indices in groups are valid and unique
        all_indices: set[int] = set()
        for group in groups:
            all_indices.update(group)
        if len(all_indices) != n_pts:
            raise ValueError(
                "All indices in groups must be unique and cover all points."
            )

    # Build a reverse lookup: point -> group ID
    group_of = np.empty(n_pts, dtype=int)
    for gi, grp in enumerate(groups):
        group_of[grp] = gi

    # Build a KD‑tree on all points (static)
    tree = scipy.spatial.KDTree(points)

    if closest_to is not None:
        if closest_to.shape != points.shape[1:]:
            raise ValueError(
                "closest_to must have the same shape as a point in points."
            )
        start_index = tree.query(closest_to, k=1)[1]

    # tell mypy that start_index is now definitely an int
    assert start_index is not None, "start_index must be set"

    if not (0 <= start_index < n_pts):
        raise ValueError("start_index must be in [0, n_points)")

    # Track which points are selected
    selected = np.zeros(n_pts, dtype=bool)
    selected[start_index] = True

    # Output ordering
    order = [start_index]

    # For each point i, min_dist[i] = distance to the closest selected point so far.
    # Initialize with distances to the start point.
    diffs = points - points[start_index]  # (n_pts, D)
    min_dist = np.linalg.norm(diffs, axis=1)
    min_dist[start_index] = 0.0  # so it never gets picked again

    # Build a max‑heap of (min_dist, index) via pushing (-min_dist, idx)
    heap = [(group_of[i], -d, i) for i, d in enumerate(min_dist)]
    heapify(heap)

    # Main loop
    while len(order) < n_pts:
        # Extract the farthest‑first candidate
        grp, neg_d, idx = heappop(heap)
        # skip if selected or stale
        if selected[idx] or -neg_d != min_dist[idx]:
            continue

        # Select it
        selected[idx] = True
        order.append(idx)
        radius = min_dist[idx]  # this was the max of min_dist

        # Radius query - only points whose current min_dist could shrink
        neighbors = tree.query_ball_point(points[idx], r=radius)
        # filter out already-selected ones
        to_check = [j for j in neighbors if not selected[j]]
        if not to_check:
            continue

        # Compute true distances for those neighbors
        subset = points[to_check] - points[idx]  # (k, D)
        dists = np.linalg.norm(subset, axis=1)

        # Wherever dists < min_dist, update and push to heap
        for j, dj in zip(to_check, dists):
            if dj < min_dist[j]:
                min_dist[j] = dj
                heappush(heap, (group_of[j], -dj, j))

    order_array = np.array(order, dtype=int)
    min_dists_array = np.array(min_dist, dtype=float)

    return order_array, min_dists_array


def find_prev_nearest_neighbors(
    points: np.ndarray,
    ordering: np.ndarray,
    max_nn: int,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each point, find its max_nn nearest neighbors among those
    appearing earlier in the given 'ordering'.

    Distances are calculated using a KD-tree, which is rebuilt every
    chunk_size points and a brute-force search is used for the rest
    points not in the tree.

    Parameters
    ----------
    points   : np.ndarray, shape (n, D)
               Your data points.
    ordering : np.ndarray, shape (n,)
               A permutation of [0..n-1] (e.g. maximin order).
    max_nn   : int
               Number of neighbors to return for each point.
    chunk_size : int
               The KD-tree is rebuild every chunk_size points.

    Returns
    -------
    neighbors: np.ndarray, shape (n, k)
               neighbors[i] are the indices (into `points`) of the
               k closest points to i that come before i in `ordering`.
               If fewer than k exist (for the very first points),
               entries are -1.

    neighbor_dists: np.ndarray, shape (n, k)
                neighbor_dists[i] are the distances to the k closest
                points to i that come before i in `ordering`.
                Entries in the corresponding to -1 neighbors are uninitialized.

    """
    npts = points.shape[0]
    if ordering.shape[0] != npts:
        raise ValueError("'ordering' must be length n")
    if max_nn < 1 or chunk_size < 1:
        raise ValueError("max_nn and chunk_size must both be at least 1")

    # Output array, default = -1
    neighbors = -np.ones((npts, max_nn), dtype=int)
    neighbor_dists = np.empty_like(neighbors, dtype=float)

    tree = None
    prev_idx_tree = None
    base_idx = 0  # rank at which current tree was built

    for r, idx in enumerate(ordering):
        if r == 0:
            continue  # no previous points

        # Rebuild the tree every B points
        if r % chunk_size == 0:
            prev_idx_tree = ordering[:r]
            prev_pts_tree = points[prev_idx_tree]
            tree = scipy.spatial.KDTree(prev_pts_tree) if r > 0 else None
            base_idx = r

        cand_idxs = []
        cand_dists = []

        # Query the static tree (all points up through base_idx)
        if tree is not None and prev_idx_tree is not None and base_idx > 0:
            q = min(max_nn, base_idx)
            d_tree, locs_tree = tree.query(points[idx], k=q)
            # unify shapes when q=1
            if q == 1:
                d_tree = np.array([d_tree])
                locs_tree = np.array([locs_tree])
            global_tree = prev_idx_tree[locs_tree]
            cand_idxs.extend(global_tree.tolist())
            cand_dists.extend(d_tree.tolist())

        # Brute‑force search on the new points since last rebuild
        new_size = r - base_idx
        if new_size > 0:
            prev_idx_chunk = ordering[base_idx:r]
            pts_chunk = points[prev_idx_chunk]
            diffs = pts_chunk - points[idx]  # (new_size, D)
            d_chunk = np.linalg.norm(diffs, axis=1)  # (new_size,)

            m = min(max_nn, new_size)
            if new_size <= m:
                sel = np.argsort(d_chunk)
            else:
                # partial sort to get the m smallest
                part = np.argpartition(d_chunk, m - 1)[:m]
                sel = part[np.argsort(d_chunk[part])]

            global_chunk = prev_idx_chunk[sel]
            cand_idxs.extend(global_chunk.tolist())
            cand_dists.extend(d_chunk[sel].tolist())

        # Merge and pick the best max_nn
        if cand_idxs:
            # pair up and sort by distance
            merged = list(zip(cand_dists, cand_idxs))
            merged.sort(key=lambda x: x[0])
            for j, (dist, neigh) in enumerate(merged[:max_nn]):
                neighbors[idx, j] = neigh
                neighbor_dists[idx, j] = dist

    return neighbors, neighbor_dists
