import numpy as np
from scipy.optimize import brentq
from pa_stats import degrees, tail_counts


def _score_delta(delta, tail_diff, n_born_zero_side, N_hist, mask_steps, t_grid):
    i_arr = np.arange(len(tail_diff), dtype=float)
    s1 = np.sum(tail_diff / (i_arr + delta))
    s2 = n_born_zero_side / delta
    Nt = N_hist[mask_steps]
    tt = t_grid[mask_steps]
    s3 = np.sum(Nt / (tt + delta * Nt))
    return s1 - s2 - s3


def mle_full_history(sources, targets, schemes, n_nodes,
                     initial_edges, initial_nodes,
                     eps=1e-4, K=50.0):
    t0 = len(initial_edges)
    T = len(sources)
    steps = schemes[t0:]
    nA = np.sum(steps == 0)
    nB = np.sum(steps == 1)
    nC = np.sum(steps == 2)
    n_steps = T - t0

    alpha_hat = nA / n_steps
    beta_hat = nB / n_steps
    gamma_hat = nC / n_steps

    d_in, d_out = degrees(sources, targets, n_nodes)
    d_in0 = np.bincount(initial_edges[:, 1], minlength=initial_nodes)
    d_out0 = np.bincount(initial_edges[:, 0], minlength=initial_nodes)

    tail_in_T, tail_out_T = tail_counts(d_in), tail_counts(d_out)
    tail_in_0, tail_out_0 = tail_counts(d_in0), tail_counts(d_out0)

    def tail_diff(tT, t0v):
        out = tT.astype(float).copy()
        out[:len(t0v)] -= t0v
        return out

    tdiff_in = tail_diff(tail_in_T, tail_in_0)
    tdiff_out = tail_diff(tail_out_T, tail_out_0)

    new_node = (steps == 0) | (steps == 2)
    N_after = initial_nodes + np.cumsum(new_node)
    N_prev = np.empty(n_steps, dtype=float)
    N_prev[0] = initial_nodes
    N_prev[1:] = N_after[:-1]
    t_grid = np.arange(t0, T, dtype=float)

    mask_in = (steps == 0) | (steps == 1)
    mask_out = (steps == 1) | (steps == 2)

    def solve(tdiff, n_born, mask):
        f = lambda d: _score_delta(d, tdiff, n_born, N_prev, mask, t_grid)
        f_lo, f_hi = f(eps), f(K)
        if f_lo < 0:
            return eps, True
        if f_hi > 0:
            return K, True
        return brentq(f, eps, K, xtol=1e-8), False

    delta_in_hat, hit_in = solve(tdiff_in, nC, mask_in)
    delta_out_hat, hit_out = solve(tdiff_out, nA, mask_out)

    def fisher_emp(tdiff, n_born, delta, share, n):
        i_arr = np.arange(len(tdiff), dtype=float)
        s = np.sum((tdiff / n) / (i_arr + delta) ** 2)
        return (s - (n_born / n) / delta ** 2
                - share * (1 - beta_hat) ** 2 / (1 + delta * (1 - beta_hat)) ** 2)

    I_in = fisher_emp(tdiff_in, nC, delta_in_hat, alpha_hat + beta_hat, n_steps)
    I_out = fisher_emp(tdiff_out, nA, delta_out_hat, gamma_hat + beta_hat, n_steps)

    return dict(alpha=alpha_hat, beta=beta_hat, gamma=gamma_hat,
                delta_in=delta_in_hat, delta_out=delta_out_hat,
                I_in=I_in, I_out=I_out, n_steps=n_steps,
                hit_in=hit_in, hit_out=hit_out)
