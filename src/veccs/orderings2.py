from heapq import heapify, heappop, heappush

import numpy as np
import scipy.spatial


def maximin_ordering(points: np.ndarray, start_index: int) -> np.ndarray:
    """
    Compute a max-min (farthest-first) ordering of a set of points in R^D.

    This is a greedy algorithm that picks the first point, then
    repeatedly picks the point that is farthest from the set of
    previously-selected points.

    Implementation is based on KD-trees and a priority queue (heap).

    Parameters
    ----------
    points : np.ndarray, shape (n_points, D)
        The input point set.
    start_index : int
        The index of the first point to pick.

    Returns
    -------
    order : np.ndarray, shape (n_points,)
        A permutation of 0..n_points-1 giving the max-min ordering.
    """
    n_pts = points.shape[0]
    if not (0 <= start_index < n_pts):
        raise ValueError("start_index must be in [0, n_points)")

    # Build a KD‑tree on all points (static)
    tree = scipy.spatial.KDTree(points)

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
    heap = [(-d, i) for i, d in enumerate(min_dist)]
    heapify(heap)

    # Main loop
    while len(order) < n_pts:
        # Extract the farthest‑first candidate
        neg_d, idx = heappop(heap)
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
                heappush(heap, (-dj, j))

    return np.array(order, dtype=int)


def find_prev_nearest_neighbors(
    points: np.ndarray,
    ordering: np.ndarray,
    max_nn: int,
    chunk_size: int = 1000,
) -> np.ndarray:
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
    """
    npts = points.shape[0]
    if ordering.shape[0] != npts:
        raise ValueError("'ordering' must be length n")
    if max_nn < 1 or chunk_size < 1:
        raise ValueError("max_nn and chunk_size must both be at least 1")

    # Output array, default = -1
    neighbors = -np.ones((npts, max_nn), dtype=int)

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
            for j, (_, neigh) in enumerate(merged[:max_nn]):
                neighbors[idx, j] = neigh

    return neighbors


def find_prev_nearest_neighbors_not_chunked(
    points: np.ndarray,
    ordering: np.ndarray,
    max_nn: int,
) -> np.ndarray:
    """
    For each point, find its max_nn nearest neighbors among those
    appearing earlier in the given 'ordering'.

    Parameters
    ----------
    points   : np.ndarray, shape (n, D)
               All your points in R^D.
    ordering : np.ndarray, shape (n,)
               A permutation of 0..n-1 (e.g. maximin order).
    max_nn   : int
               Number of neighbors to return.

    Returns
    -------
    nbrs     : np.ndarray, shape (n, k)
               nbrs[i] are the indices (into points) of the k closest
               points to i that come before i in 'ordering'.
               If fewer than k exist (for the earliest points),
               we pad with -1.
    """
    n = points.shape[0]
    if ordering.shape[0] != n:
        raise ValueError("'ordering' must be length n")
    if max_nn < 1:
        raise ValueError("max_nn must be at least 1")
    # if chunk_size < 1:
    #     raise ValueError("chunk_size must be at least 1")

    # Prepare output array, default = -1
    nbrs = -np.ones((n, max_nn), dtype=int)

    # Build a map from point-index -> its rank in 'ordering'
    rank_of = np.empty(n, dtype=int)
    for r, idx in enumerate(ordering):
        rank_of[idx] = r

    # For each point in increasing rank order...
    for r, idx in enumerate(ordering):
        if r == 0:
            continue  # no earlier points at all

        # All earlier indices and their coords
        prev_idx = ordering[:r]
        prev_pts = points[prev_idx]

        # Build a KDTree on those previous points
        tree = scipy.spatial.KDTree(prev_pts)

        # Query up to k neighbors
        kk = min(max_nn, r)
        dists, locs = tree.query(points[idx], k=kk)

        # If kk==1, make locs, dists iterable
        if kk == 1:
            locs = np.array([locs])
            dists = np.array([dists])

        # Sort by distance
        sorted_indices = np.argsort(dists)
        locs = locs[sorted_indices]
        dists = dists[sorted_indices]

        # Map back to global indices and store
        nbrs[idx, :kk] = prev_idx[locs]

    return nbrs
