from collections.abc import Sequence
from heapq import heapify, heappop, heappush
from typing import NamedTuple

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


def farthest_first_ordering(
    coordinates: np.ndarray,
    start_index: int | None = None,
    reference_point: np.ndarray | None = None,
    partition: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a farthest-first (maximin) ordering of points in space.

    This greedy algorithm selects points sequentially, each time choosing the point
    that maximizes the minimum distance to all previously selected points.

    Implementation uses KD-trees and a priority queue (heap) for efficiency.

    Parameters
    ----------
    coordinates : np.ndarray, shape (n, D)
        The point coordinates in D-dimensional space.

    start_index : int | None, default=None
        The index of the first point in the ordering. If None and reference_point
        is also None, the point closest to the middle of the bounding box is
        selected.

    reference_point : np.ndarray, shape (D,) | None, default=None
        If provided, the first point will be the one closest to this coordinate.
        Cannot be specified together with start_index.

    partition : Sequence[Sequence[int]] | None, default=None
        Optional grouping of point indices. If provided, ordering is performed
        within each group while considering points from earlier groups. Each group
        is a sequence of indices into `coordinates`.


    Returns
    -------
    ordering : np.ndarray, shape (n,)
        A permutation of point indices (0...n-1) giving the maximin ordering.

    distances : np.ndarray, shape (n,)
        For each point, its distance to the closest previously ordered point.
        The first point has distance 0.
    """
    n_pts = coordinates.shape[0]

    if reference_point is not None and start_index is not None:
        raise ValueError("Cannot specify both start_index and closest_to.")

    if reference_point is None and start_index is None:
        reference_point = middle_of_bounding_box(coordinates)

    if partition is None:
        partition = [list(range(n_pts))]
    else:
        # check that the indices in groups are valid and unique
        all_indices: set[int] = set()
        for group in partition:
            all_indices.update(group)
        if len(all_indices) != n_pts:
            raise ValueError(
                "All indices in groups must be unique and cover all points."
            )

    # Build a reverse lookup: point -> group ID
    group_of = np.empty(n_pts, dtype=int)
    for gi, grp in enumerate(partition):
        group_of[grp] = gi

    # Build a KD‑tree on all points (static)
    tree = scipy.spatial.KDTree(coordinates)

    if reference_point is not None:
        if reference_point.shape != coordinates.shape[1:]:
            raise ValueError(
                "closest_to must have the same shape as a point in points."
            )
        start_index = tree.query(reference_point, k=1)[1]

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
    diffs = coordinates - coordinates[start_index]  # (n_pts, D)
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
        neighbors = tree.query_ball_point(coordinates[idx], r=radius)
        # filter out already-selected ones
        to_check = [j for j in neighbors if not selected[j]]
        if not to_check:
            continue

        # Compute true distances for those neighbors
        subset = coordinates[to_check] - coordinates[idx]  # (k, D)
        dists = np.linalg.norm(subset, axis=1)

        # Wherever dists < min_dist, update and push to heap
        for j, dj in zip(to_check, dists):
            if dj < min_dist[j]:
                min_dist[j] = dj
                heappush(heap, (group_of[j], -dj, j))

    ordering = np.array(order, dtype=int)
    distances = np.array(min_dist, dtype=float)

    return ordering, distances


def find_preceding_neighbors(
    coordinates: np.ndarray,
    sequence: np.ndarray,
    num_neighbors: int,
    rebuild_frequency: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find nearest neighbors that precede each point in a given sequence.

    For each point, identifies the nearest neighbors among those that appear
    earlier in the specified sequence. Uses a KD-tree with periodic rebuilding
    for efficiency.

    Distances are calculated using a KD-tree, which is rebuilt every
    chunk_size points and a brute-force search is used for the rest
    points not in the tree.

    Parameters
    ----------
    data_points : np.ndarray, shape (n, D)
        The point set in D-dimensional space.
    sequence : np.ndarray, shape (n,)
        A permutation of [0..n-1] defining processing order (e.g. maximin order).
    num_neighbors : int
        Maximum number of neighbors to find for each point.
    rebuild_frequency : int, default=1000
        How often to rebuild the KD-tree (in terms of points processed).

    Returns
    -------
    neighbor_indices : np.ndarray, shape (n, k)
        neighbor_indices[i] contains indices of the k closest points to i
        that come before i in the sequence. Unfilled entries are -1.

    distances : np.ndarray, shape (n, k)
        distances[i] contains the corresponding distances to the neighbors.
        Entries corresponding to -1 neighbors are uninitialized.
    """

    npts = coordinates.shape[0]
    if sequence.shape[0] != npts:
        raise ValueError("'ordering' must be length n")
    if num_neighbors < 1 or rebuild_frequency < 1:
        raise ValueError("max_nn and chunk_size must both be at least 1")

    # Output array, default = -1
    neighbors = -np.ones((npts, num_neighbors), dtype=int)
    neighbor_dists = np.empty_like(neighbors, dtype=float)

    tree = None
    prev_idx_tree = None
    base_idx = 0  # rank at which current tree was built

    for r, idx in enumerate(sequence):
        if r == 0:
            continue  # no previous points

        # Rebuild the tree every B points
        if r % rebuild_frequency == 0:
            prev_idx_tree = sequence[:r]
            prev_pts_tree = coordinates[prev_idx_tree]
            tree = scipy.spatial.KDTree(prev_pts_tree) if r > 0 else None
            base_idx = r

        cand_idxs = []
        cand_dists = []

        # Query the static tree (all points up through base_idx)
        if tree is not None and prev_idx_tree is not None and base_idx > 0:
            q = min(num_neighbors, base_idx)
            d_tree, locs_tree = tree.query(coordinates[idx], k=q)
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
            prev_idx_chunk = sequence[base_idx:r]
            pts_chunk = coordinates[prev_idx_chunk]
            diffs = pts_chunk - coordinates[idx]  # (new_size, D)
            d_chunk = np.linalg.norm(diffs, axis=1)  # (new_size,)

            m = min(num_neighbors, new_size)
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
            for j, (dist, neigh) in enumerate(merged[:num_neighbors]):
                neighbors[idx, j] = neigh
                neighbor_dists[idx, j] = dist

    return neighbors, neighbor_dists


class FarthestFirstResult(NamedTuple):
    """
    Results from farthest-first (maximin) ordering with neighbor computation.

    This contains the reordered points and their relationships in the
    farthest-first sequence.

    Attributes
    ----------
    reordered_points : np.ndarray, shape (n, D)
        The input points reordered according to the farthest-first traversal.

    separation_distances : np.ndarray, shape (n,)
        For each point, its distance to the closest previously ordered point.
        The first point has distance 0.

    neighbor_indices : np.ndarray, shape (n, k)
        For each point, the indices of its k nearest neighbors that appear earlier
        in the ordering. Unfilled entries are -1.

    neighbor_distances : np.ndarray, shape (n, k)
        The corresponding distances to the k nearest neighbors.
        Entries corresponding to -1 in neighbor_indices are uninitialized.

    inverse_permutation : np.ndarray, shape (n,)
        Maps from the new ordering back to the original indices.
        Can be used to undo the permutation.
    """

    coordinates: np.ndarray
    separation_distances: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    inverse_permutation: np.ndarray


def reorder_farthest_first_with_neighbors(
    coordinates: np.ndarray,
    num_neighbors: int,
    start_index: int | None = None,
    reference_point: np.ndarray | None = None,
    rebuild_frequency: int = 1000,
) -> FarthestFirstResult:
    """
    Compute farthest-first ordering and find nearest neighbors in one function.

    This is a convenience function that combines `farthest_first_ordering` and
    `find_preceding_neighbors` for simple use cases without partitions. It returns
    the points reordered according to the maximin (farthest-first) traversal, along
    with their preceding neighbors and distance information.

    Parameters
    ----------
    coordinates : np.ndarray, shape (n, D)
        The point coordinates in D-dimensional space.

    num_neighbors : int
        Maximum number of neighbors to find for each point.

    start_index : int | None, default=None
        The index of the first point in the ordering. If None and reference_point
        is also None, the point closest to the middle of the bounding box is
        selected.

    reference_point : np.ndarray, shape (D,) | None, default=None
        If provided, the first point will be the one closest to this coordinate.
        Cannot be specified together with start_index.

    rebuild_frequency : int, default=1000
        How often to rebuild the KD-tree in the neighbors search.

    Returns
    -------
    FarthestFirstResult
        A named tuple with fields:
        - coordinates: The coordinates reordered according to the farthest-first
            traversal
        - separation_distances: For each point, its distance to the closest previously
            ordered point
        - neighbor_indices: For each point, indices of its k nearest preceding
            neighbors
        - neighbor_distances: Corresponding distances to the k nearest neighbors
        - inverse_permutation: Map from the new ordering back to original indices
    """
    # Compute the farthest-first ordering
    ordering, distances = farthest_first_ordering(
        coordinates, start_index=start_index, reference_point=reference_point
    )

    # Reorder the coordinates according to the ordering
    reordered_coordinates = coordinates[ordering]
    reordered_distances = distances[ordering]

    # Create a sequence of indices for the reordered coordinates
    sequence = np.arange(len(ordering))

    # Find nearest neighbors in the reordered space
    neighbor_indices, neighbor_distances = find_preceding_neighbors(
        reordered_coordinates,
        sequence,
        num_neighbors=num_neighbors,
        rebuild_frequency=rebuild_frequency,
    )

    # Return all outputs as a named tuple
    return FarthestFirstResult(
        coordinates=reordered_coordinates,
        separation_distances=reordered_distances,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        inverse_permutation=np.argsort(ordering),
    )
