import numpy as np
from numba import njit


def degrees(sources, targets, n_nodes):
    d_in = np.bincount(targets, minlength=n_nodes)
    d_out = np.bincount(sources, minlength=n_nodes)
    return d_in, d_out


def tail_counts(deg):
    hist = np.bincount(deg)
    n = hist.sum()
    return n - np.cumsum(hist)


@njit(cache=True)
def _trajectory_kernel(sources, targets, t0, snap_times, imax, counts_out):
    n_edges = len(targets)
    max_node = 0
    for t in range(n_edges):
        if sources[t] > max_node:
            max_node = sources[t]
        if targets[t] > max_node:
            max_node = targets[t]
    indeg = np.zeros(max_node + 1, dtype=np.int64)
    seen = np.zeros(max_node + 1, dtype=np.uint8)
    counts = np.zeros(imax + 2, dtype=np.int64)

    snap_idx = 0
    n_snaps = len(snap_times)

    for t in range(n_edges):
        s = sources[t]
        w = targets[t]
        if seen[s] == 0:
            seen[s] = 1
            counts[0] += 1
        if seen[w] == 0:
            seen[w] = 1
            counts[0] += 1
        d = indeg[w]
        if d <= imax:
            counts[d] -= 1
        if d + 1 <= imax:
            counts[d + 1] += 1
        indeg[w] = d + 1

        while snap_idx < n_snaps and (t + 1) == snap_times[snap_idx]:
            for i in range(imax + 1):
                counts_out[snap_idx, i] = counts[i]
            snap_idx += 1
    return snap_idx


def trajectory_Xi(sources, targets, t0, snap_times, imax=5):
    snap_times = np.asarray(np.sort(snap_times), dtype=np.int64)
    counts_out = np.zeros((len(snap_times), imax + 1), dtype=np.int64)
    _trajectory_kernel(sources, targets, t0, snap_times, imax, counts_out)
    return counts_out


# ----------------------------------------------------------------------
#  Estymator Hilla
# ----------------------------------------------------------------------
def hill_k(deg, power=0.5, min_k=50):
    M = int(np.sum(deg > 0))
    if M < 3:
        return 2
    k = int(M ** power)
    return int(max(min(max(k, min_k), M - 1), 2))


def hill_estimator(deg, k_tail):
    d = np.sort(deg[deg > 0])[::-1].astype(float)
    M = len(d)
    if M < 4:
        return np.nan
    k = int(min(max(k_tail, 2), M - 1))
    x_k = d[k]
    gamma_hat = float(np.mean(np.log(d[:k] / x_k)))
    if not np.isfinite(gamma_hat) or gamma_hat <= 0.0:
        return np.nan
    return 1.0 + 1.0 / gamma_hat


def hill_info(deg, k_tail):
    d = np.sort(deg[deg > 0])[::-1].astype(float)
    M = len(d)
    if M < 4:
        return dict(M=M, k=0, threshold=np.nan, tie_frac=np.nan)
    k = int(min(max(k_tail, 2), M - 1))
    thr = d[k]
    return dict(M=M, k=k, threshold=float(thr),
                tie_frac=float(np.mean(d[:k] == thr)))
