import numpy as np
from numba import njit


@njit(cache=True)
def _core_simulation(alpha, beta, delta_in, delta_out, target_edges,
                     sources, targets, schemes, initial_nodes, t0, seed):
    np.random.seed(seed)
    N = initial_nodes
    t = t0
    p1 = alpha
    p2 = alpha + beta

    for _ in range(target_edges - t0):
        U = np.random.random()

        if U < p1:
            v1 = N
            if delta_in == 0.0:
                v2 = targets[int(np.random.random() * t)]
            else:
                W2 = np.random.random() * (t + N * delta_in)
                if W2 < t:
                    v2 = targets[int(W2)]
                else:
                    v2 = int((W2 - t) / delta_in)
                    if v2 >= N:
                        v2 = N - 1
            N += 1
            schemes[t] = 0

        elif U < p2:
            if delta_out == 0.0:
                v1 = sources[int(np.random.random() * t)]
            else:
                W1 = np.random.random() * (t + N * delta_out)
                if W1 < t:
                    v1 = sources[int(W1)]
                else:
                    v1 = int((W1 - t) / delta_out)
                    if v1 >= N:
                        v1 = N - 1
            if delta_in == 0.0:
                v2 = targets[int(np.random.random() * t)]
            else:
                W2 = np.random.random() * (t + N * delta_in)
                if W2 < t:
                    v2 = targets[int(W2)]
                else:
                    v2 = int((W2 - t) / delta_in)
                    if v2 >= N:
                        v2 = N - 1
            schemes[t] = 1

        else:
            if delta_out == 0.0:
                v1 = sources[int(np.random.random() * t)]
            else:
                W1 = np.random.random() * (t + N * delta_out)
                if W1 < t:
                    v1 = sources[int(W1)]
                else:
                    v1 = int((W1 - t) / delta_out)
                    if v1 >= N:
                        v1 = N - 1
            v2 = N
            N += 1
            schemes[t] = 2

        sources[t] = v1
        targets[t] = v2
        t += 1

    return N


def simulate_pa_network(alpha, beta, delta_in, delta_out,
                        initial_edges, initial_nodes, target_edges, seed=0):
    t0 = len(initial_edges)
    if target_edges <= t0:
        raise ValueError("target_edges musi byc > liczby krawedzi poczatkowych")

    sources = np.empty(target_edges, dtype=np.int64)
    targets = np.empty(target_edges, dtype=np.int64)
    schemes = np.full(target_edges, -1, dtype=np.int8)

    if t0 > 0:
        sources[:t0] = initial_edges[:, 0]
        targets[:t0] = initial_edges[:, 1]

    final_nodes = _core_simulation(alpha, beta, delta_in, delta_out,
                                   target_edges, sources, targets, schemes,
                                   initial_nodes, t0, seed)
    return sources, targets, schemes, final_nodes


DEFAULT_G0 = (np.array([[0, 0]], dtype=np.int64), 1)
