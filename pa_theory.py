import numpy as np


def c1_c2(alpha, beta, delta_in, delta_out):
    gamma = 1.0 - alpha - beta
    c1 = (alpha + beta) / (1.0 + delta_in * (alpha + gamma))
    c2 = (beta + gamma) / (1.0 + delta_out * (alpha + gamma))
    return c1, c2


def tau_in_out(alpha, beta, delta_in, delta_out):
    c1, c2 = c1_c2(alpha, beta, delta_in, delta_out)
    return 1.0 + 1.0 / c1, 1.0 + 1.0 / c2


def p_in(alpha, beta, delta_in, delta_out, imax):
    gamma = 1.0 - alpha - beta
    c1, _ = c1_c2(alpha, beta, delta_in, delta_out)
    p = np.zeros(imax + 1)
    p[0] = alpha / (1.0 + c1 * delta_in)
    if imax >= 1:
        p[1] = (delta_in * p[0] + gamma / c1) / (1.0 + delta_in + 1.0 / c1)
        for i in range(2, imax + 1):
            p[i] = (i - 1.0 + delta_in) / (i + delta_in + 1.0 / c1) * p[i - 1]
    return p


def q_out(alpha, beta, delta_in, delta_out, jmax):
    gamma = 1.0 - alpha - beta
    return p_in(gamma, beta, delta_out, delta_in, jmax)


def _tail_from_p(p_vec, total_mass):
    return total_mass - np.cumsum(p_vec)


def fisher_in(alpha, beta, delta_in, delta_out, imax=200000):
    gamma = 1.0 - alpha - beta
    p = p_in(alpha, beta, delta_in, delta_out, imax)
    p_gt = _tail_from_p(p, 1.0 - beta)
    p_gt = np.clip(p_gt, 0.0, None)
    i_arr = np.arange(imax + 1, dtype=float)
    s = np.sum(p_gt / (i_arr + delta_in) ** 2)
    return (s - gamma / delta_in ** 2
            - (alpha + beta) * (1 - beta) ** 2 / (1 + delta_in * (1 - beta)) ** 2)


def fisher_out(alpha, beta, delta_in, delta_out, jmax=200000):
    gamma = 1.0 - alpha - beta
    return fisher_in(gamma, beta, delta_out, delta_in, jmax)
