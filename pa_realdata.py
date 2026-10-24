import numpy as np


def load_temporal_edgelist(path):
    raw = np.loadtxt(path, dtype=np.int64)
    order = np.argsort(raw[:, 2], kind="stable")
    raw = raw[order]
    U, V = raw[:, 0], raw[:, 1]

    id_map = {}
    def nid(x):
        i = id_map.get(x, -1)
        if i < 0:
            i = len(id_map); id_map[x] = i
        return i

    sources, targets, schemes = [], [], []
    nA = nB = nC = n_xi = n_loops = 0

    u0, v0 = int(U[0]), int(V[0])
    s0, t0 = nid(u0), nid(v0)
    sources.append(s0); targets.append(t0); schemes.append(-1)
    seen = {u0, v0}

    for k in range(1, len(U)):
        u, v = int(U[k]), int(V[k])
        su, sv = (u in seen), (v in seen)
        if not su and not sv:
            n_xi += 1
            continue
        if u == v:
            n_loops += 1
        if su and sv:
            sc = 1; nB += 1
        elif (not su) and sv:
            sc = 0; nA += 1
        else:
            sc = 2; nC += 1
        su_id = nid(u); sv_id = nid(v)
        sources.append(su_id); targets.append(sv_id); schemes.append(sc)
        seen.add(u); seen.add(v)

    sources = np.array(sources, dtype=np.int64)
    targets = np.array(targets, dtype=np.int64)
    schemes = np.array(schemes, dtype=np.int8)
    n_nodes = len(id_map)
    initial_edges = np.column_stack([sources[:1], targets[:1]])
    initial_nodes = len(set(initial_edges.ravel().tolist()))

    n_kept = len(sources)
    n_steps = nA + nB + nC
    stats = dict(
        n_raw=len(U), n_kept=n_kept, n_nodes=n_nodes,
        nA=nA, nB=nB, nC=nC, n_xi=n_xi, n_loops=n_loops,
        fracA=nA / n_steps, fracB=nB / n_steps, fracC=nC / n_steps,
        frac_xi=n_xi / len(U), frac_loops=n_loops / n_kept,
    )
    return dict(sources=sources, targets=targets, schemes=schemes,
                n_nodes=n_nodes, initial_edges=initial_edges,
                initial_nodes=initial_nodes, stats=stats)


def load_reddit_hyperlinks(path):
    import datetime as _dt
    U, V, TS = [], [], []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                t = _dt.datetime.strptime(parts[3], "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                continue
            U.append(parts[0]); V.append(parts[1]); TS.append(t)
    order = np.argsort(np.array(TS), kind="stable")
    U = [U[i] for i in order]; V = [V[i] for i in order]
    return _build_trajectory(np.array(U, dtype=object), np.array(V, dtype=object))


def _build_trajectory(U, V):
    id_map = {}
    def nid(x):
        i = id_map.get(x, -1)
        if i < 0:
            i = len(id_map); id_map[x] = i
        return i
    sources, targets, schemes = [], [], []
    nA = nB = nC = n_xi = n_loops = 0
    def key(x):
        return int(x) if isinstance(x, (int, np.integer)) else str(x)
    u0, v0 = key(U[0]), key(V[0])
    sources.append(nid(u0)); targets.append(nid(v0)); schemes.append(-1)
    seen = {u0, v0}
    for k in range(1, len(U)):
        u, v = key(U[k]), key(V[k])
        su, sv = (u in seen), (v in seen)
        if not su and not sv:
            n_xi += 1
            continue
        if u == v:
            n_loops += 1
        if su and sv:
            sc = 1; nB += 1
        elif (not su) and sv:
            sc = 0; nA += 1
        else:
            sc = 2; nC += 1
        sources.append(nid(u)); targets.append(nid(v)); schemes.append(sc)
        seen.add(u); seen.add(v)
    sources = np.array(sources, dtype=np.int64)
    targets = np.array(targets, dtype=np.int64)
    schemes = np.array(schemes, dtype=np.int8)
    n_nodes = len(id_map)
    initial_edges = np.column_stack([sources[:1], targets[:1]])
    initial_nodes = len(set(initial_edges.ravel().tolist()))
    n_kept = len(sources); n_steps = nA + nB + nC
    stats = dict(n_raw=len(U), n_kept=n_kept, n_nodes=n_nodes,
                 nA=nA, nB=nB, nC=nC, n_xi=n_xi, n_loops=n_loops,
                 fracA=nA / n_steps, fracB=nB / n_steps, fracC=nC / n_steps,
                 frac_xi=n_xi / len(U), frac_loops=n_loops / n_kept)
    return dict(sources=sources, targets=targets, schemes=schemes,
                n_nodes=n_nodes, initial_edges=initial_edges,
                initial_nodes=initial_nodes, stats=stats)
