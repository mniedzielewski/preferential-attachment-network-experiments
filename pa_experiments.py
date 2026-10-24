import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from pa_simulator import simulate_pa_network, DEFAULT_G0
from pa_theory import p_in, q_out, tau_in_out, fisher_in, fisher_out
from pa_stats import degrees, trajectory_Xi, hill_estimator, hill_k, hill_info
from pa_mle import mle_full_history

OUT = "out"
os.makedirs(OUT, exist_ok=True)


# ======================================================================
#  WSPOLNY STYL 
# ======================================================================
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

FIG_1PANEL = (5.0, 3.6)
FIG_2PANEL = (7.2, 3.3)
FIG_4PANEL = (9.6, 2.7)


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
    })


def _save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"      -> {OUT}/{name}.{{pdf,png}}")


def _save_csv(name, cols):
    keys = list(cols.keys())
    arr = np.column_stack([np.asarray(cols[k], dtype=float) for k in keys])
    np.savetxt(f"{OUT}/{name}.csv", arr, delimiter=",",
               header=",".join(keys), comments="", fmt="%.10g")
    print(f"      -> {OUT}/{name}.csv")


def _latex_table(path, header, rows, caption, label, colspec, note=None,
                 small=False):
    lines = [r"\begin{table}[htbp]", r"    \centering",
             rf"    \caption{{{caption}}}", rf"    \label{{{label}}}"]
    if small:
        lines.append(r"    \small")
    lines += [rf"    \begin{{tabular}}{{{colspec}}}", r"        \toprule",
              "        " + " & ".join(header) + r" \\", r"        \midrule"]
    for row in rows:
        lines.append("        " + " & ".join(row) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}"]
    if note:
        lines.append(rf"    \\[2pt] \footnotesize {note}")
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"      -> {path}")


def _freq_vectors(alpha, beta, delta_in, delta_out, T, R, imax, seed0):
    G0_edges, G0_nodes = DEFAULT_G0
    emp_in = np.zeros((R, imax + 1))
    emp_out = np.zeros((R, imax + 1))
    for r in range(R):
        src, dst, sch, N = simulate_pa_network(alpha, beta, delta_in, delta_out,
                                               G0_edges, G0_nodes, int(T),
                                               seed=seed0 + r)
        emp_in[r] = np.bincount(np.bincount(dst, minlength=N),
                                minlength=imax + 1)[:imax + 1] / T
        emp_out[r] = np.bincount(np.bincount(src, minlength=N),
                                 minlength=imax + 1)[:imax + 1] / T
    return emp_in, emp_out


# ======================================================================
#  Eksperyment 1a: p_i, q_j teoretyczne vs empiryczne
# ======================================================================
def exp1a(alpha=0.3, beta=0.5, delta_in=2.0, delta_out=1.0,
          T=100_000, R=200, imax=20, seed0=1000):
    print(f"[1a] T={T}, R={R}")
    _setup_style()
    emp_in, emp_out = _freq_vectors(alpha, beta, delta_in, delta_out,
                                    T, R, imax, seed0)
    p_theo = p_in(alpha, beta, delta_in, delta_out, imax)
    q_theo = q_out(alpha, beta, delta_in, delta_out, imax)
    i_arr = np.arange(imax + 1)

    fig, axes = plt.subplots(1, 2, figsize=FIG_2PANEL)
    for ax, emp, theo, lab in [(axes[0], emp_in, p_theo, r"stopnie wejściowe"),
                               (axes[1], emp_out, q_theo, r"stopnie wyjściowe")]:
        m, s = emp.mean(axis=0), emp.std(axis=0)
        ax.plot(i_arr, theo, "-", lw=1.4, color="0.15", zorder=2, label="teoria")
        ax.errorbar(i_arr, m, yerr=2 * s, fmt=MARKERS[0], ms=4, mfc="white",
                    mec=COLORS[0], ecolor=COLORS[0], color=COLORS[0],
                    capsize=2.5, lw=0.9, zorder=3, label=r"symulacja $\pm 2$SD")
        ax.set_yscale("log")
        ax.set_xlabel("$i$"); ax.set_ylabel(lab)
        ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "exp1a_pi_qi")

    m_in, s_in = emp_in.mean(0), emp_in.std(0)
    m_out, s_out = emp_out.mean(0), emp_out.std(0)
    z_in = (m_in - p_theo) / (s_in / np.sqrt(R))
    z_out = (m_out - q_theo) / (s_out / np.sqrt(R))
    rel_in = np.abs(m_in - p_theo) / p_theo
    rel_out = np.abs(m_out - q_theo) / q_theo

    _save_csv("exp1a_table", dict(
        i=i_arr, p_i_theo=p_theo, p_i_emp_mean=m_in, p_i_emp_sd=s_in,
        p_i_rel_err=rel_in, p_i_z=z_in,
        q_j_theo=q_theo, q_j_emp_mean=m_out, q_j_emp_sd=s_out,
        q_j_rel_err=rel_out, q_j_z=z_out))

    print(f"[1a] max blad wzgledny srednich: in={rel_in.max():.3%} (i={rel_in.argmax()}), "
          f"out={rel_out.max():.3%} (j={rel_out.argmax()})")
    print(f"[1a] max |z| srednich: in={np.abs(z_in).max():.2f}, out={np.abs(z_out).max():.2f}; "
          f"liczba |z|>3: in={np.sum(np.abs(z_in) > 3)}, out={np.sum(np.abs(z_out) > 3)}")
    return rel_in.max(), rel_out.max()


# ======================================================================
#  Eksperyment 1b: wykladniki ogona na siatce parametrow
# ======================================================================
def exp1b(param_grid=None, T=300_000, R=40, seed0=5000):
    print(f"[1b] T={T}, R={R}")
    if param_grid is None:
        param_grid = [
            (0.3, 0.5, 2.0, 1.0),
            (0.3, 0.5, 0.5, 0.5),
            (0.2, 0.6, 1.0, 1.0),
            (0.41, 0.49, 0.30, 0.6),
            (0.1, 0.8, 1.0, 0.2),
        ]
    G0_edges, G0_nodes = DEFAULT_G0
    rows, tex_rows = [], []
    for row_idx, (a, b, din, dout) in enumerate(param_grid):
        t_in, t_out = tau_in_out(a, b, din, dout)
        hin, hout, thr, tie, kk = [], [], [], [], []
        for r in range(R):
            src, dst, sch, N = simulate_pa_network(a, b, din, dout,
                                                   G0_edges, G0_nodes, T,
                                                   seed=seed0 + 100_000 * row_idx + r)
            d_in = np.bincount(dst, minlength=N)
            d_out = np.bincount(src, minlength=N)
            k_i, k_o = hill_k(d_in), hill_k(d_out)
            hin.append(hill_estimator(d_in, k_i))
            hout.append(hill_estimator(d_out, k_o))
            info = hill_info(d_in, k_i)
            thr.append(info["threshold"]); tie.append(info["tie_frac"]); kk.append(info["k"])

        hin_m, hin_s = float(np.nanmean(hin)), float(np.nanstd(hin))
        hout_m, hout_s = float(np.nanmean(hout)), float(np.nanstd(hout))
        k_m, thr_m, tie_m = float(np.mean(kk)), float(np.mean(thr)), float(np.mean(tie))

        rows.append([a, b, din, dout, t_in, hin_m, hin_s, t_out, hout_m, hout_s,
                     k_m, thr_m, tie_m])
        tex_rows.append([f"{a:.2f}", f"{b:.2f}", f"{din:.2f}", f"{dout:.2f}",
                          f"{t_in:.3f}", f"${hin_m:.3f} \\pm {hin_s:.3f}$",
                          f"{t_out:.3f}", f"${hout_m:.3f} \\pm {hout_s:.3f}$"])

        print(f"[1b] a={a} b={b} din={din} dout={dout}: "
              f"tau_in={t_in:.3f} hill={hin_m:.3f}+-{hin_s:.3f} "
              f"(k={k_m:.0f}, prog={thr_m:.1f}, remisy={tie_m:.2f}) | "
              f"tau_out={t_out:.3f} hill={hout_m:.3f}+-{hout_s:.3f}")

    rows = np.array(rows)
    _save_csv("exp1b_table", dict(
        alpha=rows[:, 0], beta=rows[:, 1], delta_in=rows[:, 2], delta_out=rows[:, 3],
        tau_in_theo=rows[:, 4], tau_in_hill_mean=rows[:, 5], tau_in_hill_sd=rows[:, 6],
        tau_out_theo=rows[:, 7], tau_out_hill_mean=rows[:, 8], tau_out_hill_sd=rows[:, 9],
        k=rows[:, 10], threshold_in=rows[:, 11], tie_frac_in=rows[:, 12]))

    _latex_table(
        f"{OUT}/exp1b_table.tex",
        header=[r"$\alpha$", r"$\beta$", r"$\delta_{in}$", r"$\delta_{out}$",
                r"$\tau_{in}$", r"$\hat\tau_{in}$ (Hill)",
                r"$\tau_{out}$", r"$\hat\tau_{out}$ (Hill)"],
        rows=tex_rows,
        caption=(r"Por\'ownanie teoretycznych wyk\l adnik\'ow pot\k egowych "
                 r"$\tau_{in},\tau_{out}$ z~estymatami Hilla przy regule "
                 r"$k=M^{1/2}$, gdzie $M$ to liczba wierzcho\l k\'ow o~dodatnim "
                 rf"stopniu (\'srednia $\pm$ odchylenie standardowe po $R={R}$ "
                 r"replikacjach, $T=3\cdot10^5$)."),
        label="tab:exp1b",
        colspec="cccc|cc|cc",
        note=(r"Wiersz $(0.41,0.49,0.30,0.60)$ odpowiada kalibracji do grafu WWW "
              r"z~pracy~\cite{B}."))
    return rows


# ======================================================================
#  Eksperyment 1c: tempo bledu O(sqrt(t) log t)
# ======================================================================
def exp1c(alpha=0.3, beta=0.5, delta_in=2.0, delta_out=1.0,
          T=1_000_000, R=100, i_track=(0, 1, 2, 5, 10), n_snaps=60,
          transient_min=20, n_boot=1000, seed0=9000, boot_seed=17):
    print(f"[1c] T={T}, R={R}")
    _setup_style()
    G0_edges, G0_nodes = DEFAULT_G0
    imax = max(i_track)
    snap = np.unique(np.round(np.logspace(3, np.log10(T), n_snaps)).astype(np.int64))
    p_theo = p_in(alpha, beta, delta_in, delta_out, imax)

    dev = np.zeros((R, len(snap), imax + 1))
    for r in range(R):
        src, dst, sch, N = simulate_pa_network(alpha, beta, delta_in, delta_out,
                                               G0_edges, G0_nodes, T, seed=seed0 + r)
        Xi = trajectory_Xi(src, dst, len(G0_edges), snap, imax=imax)
        dev[r] = np.abs(Xi - p_theo[None, :] * snap[:, None])

    t = snap.astype(float)
    norm_full = np.sqrt(t) * np.log(t)
    norm_sqrt = np.sqrt(t)
    rng = np.random.default_rng(boot_seed)

    fig, axes = plt.subplots(1, 2, figsize=FIG_2PANEL, sharex=True)
    long_rows, slope_rows = [], []
    for k, i in enumerate(i_track):
        col, mk = COLORS[k % len(COLORS)], MARKERS[k % len(MARKERS)]
        mask = t >= transient_min / p_theo[i]
        tt = t[mask]
        d_i = dev[:, mask, i]
        m_dev = d_i.mean(axis=0)
        s_dev = d_i.std(axis=0)
        m_full = m_dev / norm_full[mask]
        m_sqrt = m_dev / norm_sqrt[mask]
        axes[0].plot(tt, m_full, "-", marker=mk, ms=3, color=col,
                     mfc="white", label=f"$i={i}$")
        axes[1].plot(tt, m_sqrt, "-", marker=mk, ms=3, color=col,
                     mfc="white", label=f"$i={i}$")

        h = len(tt) // 2
        lt = np.log(tt[h:])
        sl = np.polyfit(lt, np.log(m_sqrt[h:]), 1)[0]
        sl_full = np.polyfit(lt, np.log(m_full[h:]), 1)[0]
        boot = np.empty(n_boot)
        for bcount in range(n_boot):
            idx = rng.integers(0, R, R)
            mb = d_i[idx].mean(axis=0) / norm_sqrt[mask]
            boot[bcount] = np.polyfit(lt, np.log(mb[h:]), 1)[0]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ref_log = float(np.mean(1.0 / lt))
        slope_rows.append([i, sl, lo, hi, sl_full, ref_log, tt[h], tt[-1]])

        for a_, b_, c_, d_, e_ in zip(tt, m_dev, s_dev, m_sqrt, m_full):
            long_rows.append([i, a_, b_, c_, d_, e_])

    axes[0].set_ylabel(r"$\overline{|X_i(t)-p_i t|}\,/\,(\sqrt{t}\log t)$")
    axes[1].set_ylabel(r"$\overline{|X_i(t)-p_i t|}\,/\,\sqrt{t}$")
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$t$"); ax.legend(ncol=2, loc="lower left")
    fig.tight_layout()
    _save(fig, "exp1c_error_rate")

    sr = np.array(slope_rows)
    _save_csv("exp1c_slopes", dict(
        i=sr[:, 0], slope_sqrt=sr[:, 1], ci_lo=sr[:, 2], ci_hi=sr[:, 3],
        slope_sqrtlog=sr[:, 4], ref_slope_if_log=sr[:, 5],
        fit_t_min=sr[:, 6], fit_t_max=sr[:, 7]))
    lr = np.array(long_rows)
    _save_csv("exp1c_curves", dict(
        i=lr[:, 0], t=lr[:, 1], dev_mean=lr[:, 2], dev_sd=lr[:, 3],
        dev_over_sqrt_t=lr[:, 4], dev_over_sqrt_t_log_t=lr[:, 5]))

    print("[1c] nachylenia log-log |dev|/sqrt(t) na drugiej polowie okna:")
    print("      i   nachylenie   95% CI (bootstrap)    nachylenie przy czynniku log t")
    for row in slope_rows:
        print(f"     {int(row[0]):>2}   {row[1]:+.4f}    [{row[2]:+.4f}, {row[3]:+.4f}]"
              f"          {row[5]:.4f}")
    return snap, dev, sr


# ======================================================================
#  Eksperyment C: niezaleznosc blokow, pokrycie CI
# ======================================================================
def expC(alpha=0.3, beta=0.5, delta_in=2.0, delta_out=1.0,
         T=100_000, R=1000, seed0=20_000, ci_level=0.95):
    print(f"[C] T={T}, R={R}")
    G0_edges, G0_nodes = DEFAULT_G0
    theta = np.array([alpha, beta, delta_in, delta_out])
    sig_theo = np.array([np.sqrt(alpha * (1 - alpha)),
                         np.sqrt(beta * (1 - beta)),
                         np.sqrt(1.0 / fisher_in(alpha, beta, delta_in, delta_out)),
                         np.sqrt(1.0 / fisher_out(alpha, beta, delta_in, delta_out))])
    est = np.zeros((R, 4))
    sig = np.zeros((R, 4))
    hits = np.zeros((R, 2))
    for r in range(R):
        src, dst, sch, N = simulate_pa_network(alpha, beta, delta_in, delta_out,
                                               G0_edges, G0_nodes, T, seed=seed0 + r)
        m = mle_full_history(src, dst, sch, N, G0_edges, G0_nodes)
        est[r] = [m["alpha"], m["beta"], m["delta_in"], m["delta_out"]]
        sig[r] = [np.sqrt(m["alpha"] * (1 - m["alpha"])),
                  np.sqrt(m["beta"] * (1 - m["beta"])),
                  np.sqrt(1.0 / m["I_in"]), np.sqrt(1.0 / m["I_out"])]
        hits[r] = [m["hit_in"], m["hit_out"]]
        if (r + 1) % max(1, R // 10) == 0:
            print(f"[C] replikacja {r+1}/{R}")

    n = T - len(G0_edges)

    corr = np.corrcoef(est.T)
    thresh = 1.96 / np.sqrt(R)
    off_block = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    print(f"[C] prog istotnosci |r| przy R={R}: {thresh:.4f}")
    for (a_, b_) in off_block:
        verdict = "ISTOTNA (!)" if abs(corr[a_, b_]) > thresh else "zerowa (szum)"
        print(f"    corr[{a_},{b_}] = {corr[a_, b_]:+.4f}  -> {verdict}")
    r_ab_theo = -alpha * beta / np.sqrt(alpha * (1 - alpha) * beta * (1 - beta))
    print(f"    corr[0,1] = {corr[0,1]:+.4f}  vs teoria {r_ab_theo:+.4f}")
    _save_csv("expC_corr", dict(
        row=np.arange(4), alpha=corr[:, 0], beta=corr[:, 1],
        delta_in=corr[:, 2], delta_out=corr[:, 3]))

    lab = [r"$\hat\alpha$", r"$\hat\beta$", r"$\hat\delta_{in}$", r"$\hat\delta_{out}$"]
    tex_rows = [[lab[i]] + [f"$\\phantom{{-}}{corr[i,j]:.3f}$" if corr[i, j] >= 0
                            else f"${corr[i,j]:.3f}$" for j in range(4)]
                for i in range(4)]
    _latex_table(
        f"{OUT}/expC_corr.tex",
        header=[""] + lab, rows=tex_rows,
        caption=(rf"Macierz korelacji pr\'obkowej estymat MLE po $R={R}$ "
                 rf"replikacjach ($n=10^5$)."),
        label="tab:expC-corr", colspec="c|cccc")

    zq = sps.norm.ppf(0.5 + ci_level / 2)
    cover_emp = np.mean(np.abs(est - theta) <= zq * sig / np.sqrt(n), axis=0)
    cover_theo = np.mean(np.abs(est - theta) <= zq * sig_theo / np.sqrt(n), axis=0)
    se_cov = np.sqrt(cover_emp * (1 - cover_emp) / R)
    _save_csv("expC_coverage", dict(
        theta_true=theta, mean=est.mean(0), sd=est.std(0),
        sd_theo=sig_theo / np.sqrt(n),
        coverage_emp_sigma=cover_emp, coverage_se=se_cov,
        cov_ci_lo=cover_emp - 1.96 * se_cov, cov_ci_hi=cover_emp + 1.96 * se_cov,
        coverage_theo_sigma=cover_theo,
        boundary_hits=np.array([0, 0, hits[:, 0].sum(), hits[:, 1].sum()])))
    _save_csv("expC_raw_estimates", dict(
        alpha=est[:, 0], beta=est[:, 1], delta_in=est[:, 2], delta_out=est[:, 3],
        sig_alpha=sig[:, 0], sig_beta=sig[:, 1],
        sig_delta_in=sig[:, 2], sig_delta_out=sig[:, 3]))

    print(f"[C] srednie estymaty: {np.round(est.mean(0), 5)}")
    print(f"[C] sd empiryczne:    {np.round(est.std(0), 5)}")
    print(f"[C] sd teoretyczne:   {np.round(sig_theo / np.sqrt(n), 5)}")
    print(f"[C] pokrycie {ci_level:.0%} CI (sigma empiryczna): {np.round(cover_emp, 3)} "
          f"+- {np.round(1.96 * se_cov, 3)}")
    print(f"[C] pokrycie {ci_level:.0%} CI (sigma teoretyczna): {np.round(cover_theo, 3)}")
    print(f"[C] przyciec do brzegu [eps,K]: delta_in={int(hits[:,0].sum())}, "
          f"delta_out={int(hits[:,1].sum())}")
    return est, corr, cover_emp


# ======================================================================
#  Eksperyment C-onset: prog wejscia w rezim normalny
# ======================================================================
def expC_onset(alpha=0.3, beta=0.5, delta_in=2.0, delta_out=1.0,
               n_list=(1000, 3000, 10_000, 30_000, 100_000),
               R=800, seed0=50_000, ci_level=0.95):
    print(f"[C-onset] n_list={n_list}, R={R}")
    _setup_style()
    G0_edges, G0_nodes = DEFAULT_G0
    theta = np.array([alpha, beta, delta_in, delta_out])
    sig_theo = np.array([np.sqrt(alpha * (1 - alpha)),
                         np.sqrt(beta * (1 - beta)),
                         np.sqrt(1.0 / fisher_in(alpha, beta, delta_in, delta_out)),
                         np.sqrt(1.0 / fisher_out(alpha, beta, delta_in, delta_out))])
    zq = sps.norm.ppf(0.5 + ci_level / 2)

    rows = []
    for jn, n in enumerate(n_list):
        est = np.zeros((R, 4))
        sig_emp = np.zeros((R, 4))
        hits = np.zeros((R, 2))
        for r in range(R):
            src, dst, sch, N = simulate_pa_network(alpha, beta, delta_in, delta_out,
                                                   G0_edges, G0_nodes, n,
                                                   seed=seed0 + 1000 * jn + r)
            m = mle_full_history(src, dst, sch, N, G0_edges, G0_nodes)
            est[r] = [m["alpha"], m["beta"], m["delta_in"], m["delta_out"]]
            sig_emp[r] = [np.sqrt(max(m["alpha"] * (1 - m["alpha"]), 1e-12)),
                          np.sqrt(max(m["beta"] * (1 - m["beta"]), 1e-12)),
                          np.sqrt(1.0 / m["I_in"]) if m["I_in"] > 0 else np.nan,
                          np.sqrt(1.0 / m["I_out"]) if m["I_out"] > 0 else np.nan]
            hits[r] = [m["hit_in"], m["hit_out"]]
        neff = n - len(G0_edges)
        z = np.sqrt(neff) * (est - theta) / sig_theo
        sw_p = [sps.shapiro(z[:, k]).pvalue for k in range(4)]
        n_uniq = [len(np.unique(est[:, k])) for k in range(4)]
        cover = np.nanmean(np.abs(est - theta) <= zq * sig_emp / np.sqrt(neff), axis=0)
        bias = est.mean(0) - theta
        sd_emp = est.std(0)
        rows.append([n, *sw_p, *cover, *bias, *sd_emp, *n_uniq,
                     hits[:, 0].sum(), hits[:, 1].sum()])
        print(f"[C-onset] n={n:>7}: Shapiro p = "
              + ", ".join(f"{p:.4f}" for p in sw_p)
              + " | pokrycie = " + ", ".join(f"{c:.3f}" for c in cover)
              + f" | liczba roznych wartosci (alpha,beta) = {n_uniq[0]},{n_uniq[1]}"
              + f" | przyciec = {int(hits[:,0].sum())},{int(hits[:,1].sum())}")

    rows = np.array(rows)
    se_cov = np.sqrt(0.95 * 0.05 / R)
    _save_csv("expC_onset_table", dict(
        n=rows[:, 0],
        swp_alpha=rows[:, 1], swp_beta=rows[:, 2], swp_din=rows[:, 3], swp_dout=rows[:, 4],
        cov_alpha=rows[:, 5], cov_beta=rows[:, 6], cov_din=rows[:, 7], cov_dout=rows[:, 8],
        bias_alpha=rows[:, 9], bias_beta=rows[:, 10], bias_din=rows[:, 11], bias_dout=rows[:, 12],
        sd_alpha=rows[:, 13], sd_beta=rows[:, 14], sd_din=rows[:, 15], sd_dout=rows[:, 16],
        nuniq_alpha=rows[:, 17], nuniq_beta=rows[:, 18],
        nuniq_din=rows[:, 19], nuniq_dout=rows[:, 20],
        hits_din=rows[:, 21], hits_dout=rows[:, 22],
        cov_band_halfwidth=np.full(len(rows), 1.96 * se_cov)))

    names = [r"$\hat\alpha$", r"$\hat\beta$", r"$\hat\delta_{in}$", r"$\hat\delta_{out}$"]
    fig, axes = plt.subplots(1, 2, figsize=FIG_2PANEL)
    for k in range(4):
        axes[0].plot(rows[:, 0], rows[:, 1 + k], "-", marker=MARKERS[k],
                     color=COLORS[k], mfc="white", label=names[k])
        axes[1].plot(rows[:, 0], rows[:, 5 + k], "-", marker=MARKERS[k],
                     color=COLORS[k], mfc="white", label=names[k])
    axes[0].axhline(0.05, color="0.35", ls="--", lw=1.0, label="poziom $0.05$")
    axes[0].set_ylabel("$p$-wartość (Shapiro–Wilk)"); axes[0].set_yscale("log")
    band = zq * np.sqrt(ci_level * (1 - ci_level) / R)
    axes[1].axhspan(ci_level - band, ci_level + band, color="0.75", alpha=0.35,
                    label=r"pas $\pm 1.96\sqrt{p(1-p)/R}$")
    axes[1].axhline(ci_level, color="0.35", ls="--", lw=1.0)
    axes[1].set_ylabel(f"pokrycie CI {ci_level:.0%}")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("$n$"); ax.legend()
    fig.tight_layout()
    _save(fig, "expC_onset")
    return rows


if __name__ == "__main__":
    exp1a(T=100_000, R=200)
    exp1b(T=300_000, R=40)
    exp1c(T=1_000_000, R=100)
    expC(T=100_000, R=1000)
    expC_onset(R=800)
    print("\n=== wszystkie eksperymenty symulacyjne zakonczone ===")
