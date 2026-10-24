import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from pa_realdata import load_temporal_edgelist, load_reddit_hyperlinks
from pa_simulator import simulate_pa_network
from pa_mle import mle_full_history
from pa_stats import degrees, hill_estimator, hill_k, hill_info
from pa_theory import c1_c2
from pa_experiments import _setup_style, _save, _save_csv, COLORS, MARKERS, OUT

os.makedirs(OUT, exist_ok=True)

DATA_DIR = os.environ.get(
    "PA_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)

_CACHE = {}


def _load(name):
    if name in _CACHE:
        return _CACHE[name]
    U = DATA_DIR + os.sep
    if name == "email-Eu":
        d = load_temporal_edgelist(U + "email-Eu-core-temporal.txt")
    elif name == "CollegeMsg":
        d = load_temporal_edgelist(U + "CollegeMsg.txt")
    elif name == "MathOverflow":
        d = load_temporal_edgelist(U + "sx-mathoverflow-a2q.txt")
    elif name == "Reddit":
        d = load_reddit_hyperlinks(U + "soc-redditHyperlinks-body.tsv")
    else:
        raise ValueError(name)
    _CACHE[name] = d
    return d


DATASET_NAMES = ["email-Eu", "CollegeMsg", "Reddit", "MathOverflow"]


def _ccdf(deg, grid):
    deg = np.asarray(deg)
    n = len(deg)
    srt = np.sort(deg)
    out = (n - np.searchsorted(srt, grid, side="left")) / float(n)
    return np.clip(out, 1e-12, None)


# ----------------------------------------------------------------------
#  Podsumowanie + MLE + test spojnosci tau
# ----------------------------------------------------------------------
def real_summary(names=DATASET_NAMES, ks_map=None):
    rows = []
    for name in names:
        d = _load(name)
        s = d["stats"]
        m = mle_full_history(d["sources"], d["targets"], d["schemes"],
                             d["n_nodes"], d["initial_edges"], d["initial_nodes"])
        c1, c2 = c1_c2(m["alpha"], m["beta"], m["delta_in"], m["delta_out"])
        tin_mle, tout_mle = 1 + 1 / c1, 1 + 1 / c2
        din, dout = degrees(d["sources"], d["targets"], d["n_nodes"])

        k_in, k_out = hill_k(din), hill_k(dout)
        tin_h, tout_h = hill_estimator(din, k_in), hill_estimator(dout, k_out)
        info_in = hill_info(din, k_in)

        rows.append(dict(name=name, n_nodes=d["n_nodes"], n_edges=s["n_kept"],
                         fracA=s["fracA"], fracB=s["fracB"], fracC=s["fracC"],
                         frac_xi=s["frac_xi"], frac_loops=s["frac_loops"],
                         alpha=m["alpha"], beta=m["beta"],
                         din=m["delta_in"], dout=m["delta_out"],
                         hit_in=float(m["hit_in"]), hit_out=float(m["hit_out"]),
                         tin_mle=tin_mle, tin_h=tin_h,
                         tout_mle=tout_mle, tout_h=tout_h,
                         k_hill=info_in["k"], hill_threshold=info_in["threshold"],
                         hill_tie_frac=info_in["tie_frac"]))
        print(f"[{name}] N={d['n_nodes']} E={s['n_kept']} "
              f"A/B/C={s['fracA']:.3f}/{s['fracB']:.3f}/{s['fracC']:.3f} "
              f"xi={s['frac_xi']:.4f} petle={s['frac_loops']:.4f}")
        print(f"    theta_hat=({m['alpha']:.4f},{m['beta']:.4f},"
              f"{m['delta_in']:.3f},{m['delta_out']:.3f})  "
              f"brzeg: in={m['hit_in']}, out={m['hit_out']}")
        print(f"    tau_in  MLE={tin_mle:.2f} Hill={tin_h:.2f} "
              f"| tau_out MLE={tout_mle:.2f} Hill={tout_h:.2f}")
        print(f"    prog Hilla: k={info_in['k']}, D_(k+1)={info_in['threshold']:.0f}, "
              f"remisy={info_in['tie_frac']:.2f}")

    keys = [k for k in rows[0].keys() if k != "name"]
    _save_csv("real_summary", {k: [r[k] for r in rows] for k in keys})
    with open(f"{OUT}/real_summary_names.txt", "w") as fh:
        fh.write("\n".join(r["name"] for r in rows) + "\n")
    _write_summary_table(rows, ks_map)
    return rows


def _write_summary_table(rows, ks_map=None):
    def f(x, d=3): return f"{x:.{d}f}"
    has_ks = ks_map is not None
    colspec = "lccccc" + ("c" if has_ks else "")
    head = (r"        sieć & $N$ & A/B/C [\%] & $\hat\beta$ & "
            r"$\hat\tau_{in}$ (MLE/Hill) & $\hat\tau_{out}$ (MLE/Hill)")
    if has_ks:
        head += r" & KS$_{in}$ (model/kontrola)"
    lines = [r"\begin{table}[htbp]", r"    \centering",
             r"    \caption{Charakterystyka sieci (udziały schematów A/B/C), estymaty MLE, "
             r"test spójności wykładnika ogona (MLE wobec Hilla) oraz odległość "
             r"Kołmogorowa--Smirnowa w~walidacji predykcyjnej, zestawiona z~kontrolą "
             r"bazową (prognoza \enquote{nic się nie zmienia}).}",
             r"    \label{tab:real-summary}", r"    \small",
             rf"    \begin{{tabular}}{{{colspec}}}", r"        \toprule",
             head + r" \\", r"        \midrule"]
    for r in rows:
        abc = f"{100*r['fracA']:.1f}/{100*r['fracB']:.1f}/{100*r['fracC']:.1f}"
        row = (f"        \\texttt{{{r['name']}}} & {r['n_nodes']} & {abc} & "
               f"{f(r['beta'])} & "
               f"{f(r['tin_mle'],2)}\\,/\\,{f(r['tin_h'],2)} & "
               f"{f(r['tout_mle'],2)}\\,/\\,{f(r['tout_h'],2)}")
        if has_ks:
            km, kb = ks_map.get(r["name"], (float("nan"), float("nan")))
            row += f" & {km:.3f}\\,/\\,{kb:.3f}"
        lines.append(row + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    with open(f"{OUT}/real_summary.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"      -> {OUT}/real_summary.tex")


# ----------------------------------------------------------------------
#  Walidacja predykcyjna out-of-sample + kontrola bazowa
# ----------------------------------------------------------------------
def _oos_compute(name, train_frac=0.7, R=40, seed0=0):
    d = _load(name)
    src, dst, sch = d["sources"], d["targets"], d["schemes"]
    n2 = len(src); n1 = int(train_frac * n2)

    init_e = np.column_stack([src[:1], dst[:1]])
    init_n = len(set(init_e.ravel().tolist()))
    m = mle_full_history(src[:n1], dst[:n1], sch[:n1],
                         int(max(src[:n1].max(), dst[:n1].max())) + 1,
                         init_e, init_n)
    theta = (m["alpha"], m["beta"], m["delta_in"], m["delta_out"])

    N2 = int(max(src.max(), dst.max())) + 1
    din_real, dout_real = degrees(src, dst, N2)

    init_edges = np.column_stack([src[:n1], dst[:n1]])
    init_nodes = int(max(src[:n1].max(), dst[:n1].max())) + 1
    din_frozen, dout_frozen = degrees(src[:n1], dst[:n1], init_nodes)

    grid = np.arange(1, max(din_real.max(), dout_real.max()) + 1)
    ccdf_in, ccdf_out, ks_in, ks_out = [], [], [], []
    for r in range(R):
        s2, t2, _, Nf = simulate_pa_network(*theta, init_edges, init_nodes,
                                            n2, seed=seed0 + r)
        di, do = degrees(s2, t2, Nf)
        ccdf_in.append(_ccdf(di, grid)); ccdf_out.append(_ccdf(do, grid))
        ks_in.append(sps.ks_2samp(di, din_real).statistic)
        ks_out.append(sps.ks_2samp(do, dout_real).statistic)

    ks_in_base = sps.ks_2samp(din_frozen, din_real).statistic
    ks_out_base = sps.ks_2samp(dout_frozen, dout_real).statistic

    return dict(name=name, theta=theta, grid=grid, n1=n1, n2=n2,
                N_frozen=init_nodes, N_final=N2,
                ccdf_in=np.array(ccdf_in), ccdf_out=np.array(ccdf_out),
                data_in=_ccdf(din_real, grid), data_out=_ccdf(dout_real, grid),
                frozen_in=_ccdf(din_frozen, grid), frozen_out=_ccdf(dout_frozen, grid),
                ks_in=(float(np.mean(ks_in)), float(np.std(ks_in))),
                ks_out=(float(np.mean(ks_out)), float(np.std(ks_out))),
                ks_in_base=float(ks_in_base), ks_out_base=float(ks_out_base),
                n_nodes=d["n_nodes"])


def _panel_oos(ax, grid, cc, data, frozen, nn, xlabel):
    lo, hi = np.percentile(cc, [5, 95], axis=0)
    ax.fill_between(grid, lo, hi, color=COLORS[0], alpha=0.30,
                    label="symulacja (5–95 perc.)")
    ax.plot(grid, data, color=COLORS[1], lw=1.4, label="dane $G(n_2)$")
    ax.plot(grid, frozen, color="0.35", lw=0.9, ls=":",
            label="kontrola: zamrożony $G(n_1)$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(r"$\mathbb{P}(D\geq k)$")
    ax.set_ylim(1 / nn / 2, 1.5)


def real_out_of_sample_all(names=DATASET_NAMES, train_frac=0.7, R=40):
    _setup_style()
    res = [_oos_compute(n, train_frac, R) for n in names]
    nrow = len(names)
    fig, axes = plt.subplots(nrow, 2, figsize=(7.2, 2.4 * nrow), squeeze=False)
    for i, r in enumerate(res):
        _panel_oos(axes[i][0], r["grid"], r["ccdf_in"], r["data_in"], r["frozen_in"],
                   r["n_nodes"], "stopień wejściowy $k$" if i == nrow - 1 else "")
        _panel_oos(axes[i][1], r["grid"], r["ccdf_out"], r["data_out"], r["frozen_out"],
                   r["n_nodes"], "stopień wyjściowy $k$" if i == nrow - 1 else "")
        axes[i][0].set_ylabel(r["name"] + "\n" + r"$\mathbb{P}(D\geq k)$")
        axes[i][1].set_ylabel("")
        axes[i][0].text(0.05, 0.08,
                        f"KS={r['ks_in'][0]:.3f} (kontrola {r['ks_in_base']:.3f})",
                        transform=axes[i][0].transAxes, fontsize=7.5, va="bottom")
        axes[i][1].text(0.05, 0.08,
                        f"KS={r['ks_out'][0]:.3f} (kontrola {r['ks_out_base']:.3f})",
                        transform=axes[i][1].transAxes, fontsize=7.5, va="bottom")
    axes[0][0].legend(loc="upper right", fontsize=7)
    fig.tight_layout(); _save(fig, "real_oos_all")

    for r in res:
        lo_i, hi_i = np.percentile(r["ccdf_in"], [5, 95], axis=0)
        lo_o, hi_o = np.percentile(r["ccdf_out"], [5, 95], axis=0)
        _save_csv(f"real_oos_curves_{r['name']}", dict(
            k=r["grid"],
            data_in=r["data_in"], sim_in_p5=lo_i, sim_in_p50=np.median(r["ccdf_in"], axis=0),
            sim_in_p95=hi_i, frozen_in=r["frozen_in"],
            data_out=r["data_out"], sim_out_p5=lo_o,
            sim_out_p50=np.median(r["ccdf_out"], axis=0),
            sim_out_p95=hi_o, frozen_out=r["frozen_out"]))

    _save_csv("real_oos_summary", dict(
        idx=np.arange(len(res)),
        n1=[r["n1"] for r in res], n2=[r["n2"] for r in res],
        N_frozen=[r["N_frozen"] for r in res], N_final=[r["N_final"] for r in res],
        alpha=[r["theta"][0] for r in res], beta=[r["theta"][1] for r in res],
        delta_in=[r["theta"][2] for r in res], delta_out=[r["theta"][3] for r in res],
        ks_in=[r["ks_in"][0] for r in res], ks_in_sd=[r["ks_in"][1] for r in res],
        ks_in_baseline=[r["ks_in_base"] for r in res],
        ks_out=[r["ks_out"][0] for r in res], ks_out_sd=[r["ks_out"][1] for r in res],
        ks_out_baseline=[r["ks_out_base"] for r in res]))
    with open(f"{OUT}/real_oos_summary_names.txt", "w") as fh:
        fh.write("\n".join(r["name"] for r in res) + "\n")

    print("[oos-all] KS (model | kontrola bazowa):")
    for r in res:
        print(f"    {r['name']:>14}: in {r['ks_in'][0]:.3f} +- {r['ks_in'][1]:.3f} | "
              f"{r['ks_in_base']:.3f}   ||   out {r['ks_out'][0]:.3f} +- "
              f"{r['ks_out'][1]:.3f} | {r['ks_out_base']:.3f}")
    return res


def real_oos_trainfrac_scan(names=DATASET_NAMES, fracs=(0.5, 0.7, 0.9), R=20):
    rows, labels = [], []
    for name in names:
        for tf in fracs:
            r = _oos_compute(name, train_frac=tf, R=R)
            rows.append([tf, r["n1"], r["n2"],
                         r["ks_in"][0], r["ks_in"][1], r["ks_in_base"],
                         r["ks_out"][0], r["ks_out"][1], r["ks_out_base"]])
            labels.append(f"{name}@{tf}")
            print(f"[oos-scan] {name:>14} train={tf:.2f}: "
                  f"KS_in {r['ks_in'][0]:.3f} vs kontrola {r['ks_in_base']:.3f} | "
                  f"KS_out {r['ks_out'][0]:.3f} vs kontrola {r['ks_out_base']:.3f}")
    rows = np.array(rows)
    _save_csv("real_oos_trainfrac", dict(
        train_frac=rows[:, 0], n1=rows[:, 1], n2=rows[:, 2],
        ks_in=rows[:, 3], ks_in_sd=rows[:, 4], ks_in_baseline=rows[:, 5],
        ks_out=rows[:, 6], ks_out_sd=rows[:, 7], ks_out_baseline=rows[:, 8]))
    with open(f"{OUT}/real_oos_trainfrac_names.txt", "w") as fh:
        fh.write("\n".join(labels) + "\n")
    return rows


# ----------------------------------------------------------------------
#  Stabilnosc parametrow w czasie
# ----------------------------------------------------------------------
def _stability_compute(name, n_points=15):
    d = _load(name)
    src, dst, sch = d["sources"], d["targets"], d["schemes"]
    n = len(src)
    ms = np.unique(np.linspace(int(0.1 * n), n, n_points).astype(int))
    init_e = np.column_stack([src[:1], dst[:1]])
    init_n = len(set(init_e.ravel().tolist()))
    est = []
    for mm in ms:
        Nm = int(max(src[:mm].max(), dst[:mm].max())) + 1
        r = mle_full_history(src[:mm], dst[:mm], sch[:mm], Nm, init_e, init_n)
        est.append([r["alpha"], r["beta"], r["delta_in"], r["delta_out"],
                    float(r["hit_in"]), float(r["hit_out"])])
    return ms, ms / n, np.array(est)


def real_stability_all(names=DATASET_NAMES, n_points=15):
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    dump = {}
    for i, name in enumerate(names):
        ms, frac, est = _stability_compute(name, n_points)
        axes[0].plot(frac, est[:, 2], "-", marker=MARKERS[i % len(MARKERS)],
                     color=COLORS[i % len(COLORS)], mfc="white", ms=4, label=name)
        axes[1].plot(frac, est[:, 3], "-", marker=MARKERS[i % len(MARKERS)],
                     color=COLORS[i % len(COLORS)], mfc="white", ms=4, label=name)
        _save_csv(f"real_stability_{name}", dict(
            m=ms, frac=frac, alpha=est[:, 0], beta=est[:, 1],
            delta_in=est[:, 2], delta_out=est[:, 3],
            hit_in=est[:, 4], hit_out=est[:, 5]))
        dump[name] = est
        print(f"[stab {name}] delta_in: {est[0,2]:.3f} -> {est[-1,2]:.3f}   "
              f"delta_out: {est[0,3]:.3f} -> {est[-1,3]:.3f}   "
              f"beta: {est[0,1]:.3f} -> {est[-1,1]:.3f}   "
              f"przyciec do brzegu: {int(est[:,4].sum())}/{int(est[:,5].sum())}")
    axes[0].set_ylabel(r"$\hat\delta_{in}$"); axes[1].set_ylabel(r"$\hat\delta_{out}$")
    for ax in axes:
        ax.set_xlabel("ułamek historii $m/n$"); ax.legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, "real_stability_all")
    return dump


if __name__ == "__main__":
    oos = real_out_of_sample_all(R=40)
    ks_map = {r["name"]: (r["ks_in"][0], r["ks_in_base"]) for r in oos}
    real_summary(ks_map=ks_map)
    real_oos_trainfrac_scan(R=20)
    real_stability_all()
    print("\n=== eksperymenty na danych rzeczywistych zakonczone ===")
