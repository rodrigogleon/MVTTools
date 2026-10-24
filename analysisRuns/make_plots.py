"""
Benchmark plots for GRB 230307A and GRB 211211A, made only from saved results (no reruns).

  P1  MVT against bin width: our median MVT (16-84% errors) and Bala et al.'s Table 3, per detector set
  P2  validation curve: Bala et al.'s 95% band (from their mvt_snr_fit_model.npz) with our rows
      (Bala-compatible SNR, used for the class, and physical SNR) and Bala et al.'s own rows
  P3  detector selection: combined SNR against number of detectors added, for each step
  P4  SNR_MVT divided by Bala et al.'s Table 3 SNR (both at their median MVT) against number of detectors,
      physical (all photons) and Bala-compatible (merged)
  P5  the 300 Monte Carlo MVT values of GRB 230307A row F: all photons against merged (valid, upper limits,
      failed), with Bala et al.'s median and errors
  P6  GRB 230307A light curve in 1.8 ms bins from row F's saved 0.01 ms light curves, all photons against merged:
      whole interval, zoom on the brightest bin, and the fraction lost per bin against the bin's counts

Inputs
  results/snr_classify/<GRB>/summary.json      (written by run_grb_mvt.py --post)
  results/detector_selection/<GRB>_<t0>_<t1>.json  (written by run_detector_selection.py)
  results/bala_rows_8-900keV_rs_haarfix/RunF and results/merged_mc/GRB230307A_RunF  (mvt_values_seconds.npy,
      mvt_uncertainties_seconds.npy, counts.npy, summary.json)
  Bala et al.'s clasify_mvt_value.py and mvt_snr_fit_model.npz
Output: results/plots/P1..P6 (.png and .pdf), and a printed table of every plotted value, to check against the
saved tables.

Usage
  python make_plots.py              all plots
  python make_plots.py --only P4    only the plots listed
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/mnt/data/rgl36/FYP/results"
BALA_CODE = "/mnt/data/rgl36/FYP/external/GBM_MVT_paper/MVTfermi"
PLOT_DIR = os.path.join(RESULTS, "plots")
GRBS = ["GRB230307A", "GRB211211A"]
SKIP_RUNS = {"Ds"}                                   # our own variant, not a Table 3 row
SET_ORDER = ["1det", "3det", "5det", "7det", "all"]
SET_LABEL = {"1det": "1 det", "3det": "3 det", "5det": "5 det", "7det": "7 det", "all": "12 det"}
SET_N = {"1det": 1, "3det": 3, "5det": 5, "7det": 7, "all": 12}
SET_COLOR = {s: c for s, c in zip(SET_ORDER, plt.rcParams["axes.prop_cycle"].by_key()["color"])}
P1_SET_SPACING = 1.18                                # P1: horizontal spacing factor between detector sets
P4_ROW_SPACING = 1.06                                # P4: horizontal spacing factor between rows in a column
ROW_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]   # P4: one colour per row within a column
P5_RUN = "F"                                         # GRB 230307A row for P5 (merged runs exist for B and F)
P6_RUN = "F"                                         # GRB 230307A row whose saved light curves P6 uses
P6_BIN_MS = 1.8                                      # plotted bin width (about the MVT)
P6_ZOOM_S = 0.25                                     # zoom half-width around the brightest bin
SELECTION_FILES = [("GRB230307A", "GRB230307A_-1.6_45.3.json"), ("GRB230307A", "GRB230307A_-1.64_18.json"),
                   ("GRB211211A", "GRB211211A_-0.9_15.json")]


def loadRuns(grb):
    """Rows of one burst from the --post summary: detector set, bin width, ours, Bala et al., SNRs."""
    runs = json.load(open(os.path.join(RESULTS, "snr_classify", grb, "summary.json")))["runs"]
    rows = []
    for run, r in runs.items():
        if run in SKIP_RUNS:
            continue
        o, b = r["table"]["ours"], r["table"]["bala"]
        rows.append(dict(run=run, set=r["detectorSet"], bw_ms=r["binSize_s"] * 1e3,
                         med=o["median_ms"], plus=o["plus_ms"], minus=o["minus_ms"],
                         snrPhys=r["snrAtOurMedian"]["observed"], snrB=r["snrBalaDefAtOurMedian"],
                         snrPhysAtB=r["snrAtBalaMedian"]["observed"], snrBAtB=r["snrBalaDefAtBalaMedian"],
                         bMed=b["median"], bPlus=b["plus"], bMinus=b["minus"], bSnr=b["snr"]))
    return rows


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"{name}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"saved {PLOT_DIR}/{name}.png/.pdf")


def plotP1(allRows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, grb in zip(axes, GRBS):
        rows = allRows[grb]
        sets = [s for s in SET_ORDER if any(r["set"] == s for r in rows)]
        for i, s in enumerate(sets):
            sub = sorted([r for r in rows if r["set"] == s], key=lambda r: r["bw_ms"])
            shift = P1_SET_SPACING ** (i - (len(sets) - 1) / 2)   # equal horizontal offsets between detector sets
            x = np.array([r["bw_ms"] for r in sub]) * shift
            ax.errorbar(x / 1.03, [r["med"] for r in sub], yerr=[[r["minus"] for r in sub], [r["plus"] for r in sub]],
                        fmt="o-", color=SET_COLOR[s], ms=5, lw=1, capsize=3, label=f"{SET_LABEL[s]} (ours)")
            ax.errorbar(x * 1.03, [r["bMed"] for r in sub], yerr=[[r["bMinus"] for r in sub], [r["bPlus"] for r in sub]],
                        fmt="s--", color=SET_COLOR[s], mfc="none", ms=5, lw=1, capsize=3,
                        label=f"{SET_LABEL[s]} (Bala et al.)")
            for r, xi in zip(sub, x):
                ax.annotate(r["run"], (xi, r["med"] + r["plus"]), textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_xticks([0.003, 0.01, 0.1])
        ax.set_xticklabels(["0.003", "0.01", "0.1"])
        ax.set_xlabel("bin width (ms)")
        ax.set_ylabel("median MVT (ms)")
        ax.set_title(grb)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=7)
    save(fig, "P1_mvt_vs_binwidth")


def validationBand():
    """Lower, median and upper SNR of Bala et al.'s 95% band on a grid of MVT (ms)."""
    sys.path.insert(0, BALA_CODE)
    import clasify_mvt_value as validation
    validation.MODEL_FILE = os.path.join(BALA_CODE, "mvt_snr_fit_model.npz")
    model = validation.load_model()
    if model is None:
        sys.exit(f"Could not load the validation curve from {validation.MODEL_FILE}")
    interpolators, mvtRangeLog = model
    lo, hi = (10 ** x for x in mvtRangeLog)
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    band = []
    for m in grid:
        _, bounds = validation.classify_mvt_value(m, 1.0, interpolators, mvtRangeLog)
        band.append([10 ** x for x in bounds])
    return grid, np.array(band), (lo, hi)


def plotP2(allRows):
    grid, band, (lo, hi) = validationBand()
    print(f"validation curve: MVT range {lo:.2f}-{hi:.0f} ms")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, grb in zip(axes, GRBS):
        ax.fill_betweenx(grid, band[:, 0], band[:, 2], color="0.85", label="Bala et al. 95% band")
        ax.plot(band[:, 1], grid, "k-", lw=1, label="Bala et al. median")
        rows = allRows[grb]
        for s in [s for s in SET_ORDER if any(r["set"] == s for r in rows)]:
            sub = [r for r in rows if r["set"] == s]
            c = SET_COLOR[s]
            ax.errorbar([r["snrB"] for r in sub], [r["med"] for r in sub],
                        yerr=[[r["minus"] for r in sub], [r["plus"] for r in sub]], fmt="o", color=c, ms=5,
                        capsize=2, lw=0.8, label=f"{SET_LABEL[s]}: ours, Bala-compatible SNR")
            ax.plot([r["snrPhys"] for r in sub], [r["med"] for r in sub], "o", mfc="none", color=c, ms=6,
                    label=f"{SET_LABEL[s]}: ours, physical SNR")
            ax.plot([r["bSnr"] for r in sub], [r["bMed"] for r in sub], "x", color=c, ms=7,
                    label=f"{SET_LABEL[s]}: Bala et al.")
            for r in sub:
                ax.annotate(r["run"], (r["snrB"], r["med"]), textcoords="offset points", xytext=(-9, 0),
                            ha="center", va="center", fontsize=7)
        allSnr = [v for r in rows for v in (r["snrB"], r["snrPhys"], r["bSnr"])]
        allMvt = [v for r in rows for v in (r["med"] - r["minus"], r["med"] + r["plus"], r["bMed"])]
        ax.set_xlim(min(allSnr) / 2, max(allSnr) * 2)
        ax.set_ylim(min(allMvt) / 2, max(allMvt) * 2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("SNR_MVT")
        ax.set_ylabel("MVT (ms)")
        ax.set_title(grb)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=6, loc="upper left")
    save(fig, "P2_validation_curve")


def drawStep(ax, step, color, label):
    """Combined SNR against number of detectors for one selection step; each point labelled with the detector
    added at that point (the order can differ between steps); the maximum is marked with a star."""
    cum = step["cumulative"]
    k = [c["k"] for c in cum]
    snr = [c["snr"] for c in cum]
    ax.plot(k, snr, "o-", color=color, ms=4, lw=1, label=label)
    for c in cum:
        ax.annotate(c["added"], (c["k"], c["snr"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=6, color=color, zorder=5)
    best = int(np.argmax(snr))
    ax.plot(k[best], snr[best], "*", color=color, ms=11, mec="k", zorder=3)
    return k, snr, best


def plotP3(allRows=None):
    n = len(SELECTION_FILES)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print("\nP3 values (detector selection)")
    for col, (grb, fname) in enumerate(SELECTION_FILES):
        rep = json.load(open(os.path.join(RESULTS, "detector_selection", fname)))
        title = f"{grb}, {rep['interval_s'][0]:g} to {rep['interval_s'][1]:g} s"
        # top row: the first step (64 ms, or 16 ms for short bursts); bottom row: the repeats at bin width = MVT
        for row, steps in ((0, rep["steps"][:1]), (1, rep["steps"][1:])):
            ax = axes[row, col]
            for i, step in enumerate(steps):
                bw = step["binWidth_s"] * 1e3
                k, snr, best = drawStep(ax, step, colors[i if row == 0 else i + 1], f"{bw:g} ms bins")
                print(f"  {title}, {bw:g} ms: order {[c['added'] for c in step['cumulative']]}, "
                      f"SNR {[round(s, 1) for s in snr]}, maximum at {k[best]} detectors, "
                      f"saved optimal set {step['optimal']}")
            ax.set_xticks(range(1, 13))
            ax.set_xlabel("number of detectors combined (in ranked order)")
            ax.set_ylabel("combined SNR")
            ax.set_title(title + ("" if row == 0 else ", repeated at bin width = MVT"), fontsize=10)
            ax.grid(True, ls=":", alpha=0.5)
            if steps:
                ax.legend(fontsize=7)
    save(fig, "P3_detector_selection")


def plotP4(allRows):
    """SNR_MVT divided by Bala et al.'s Table 3 SNR, both at Bala et al.'s median MVT (same bin width on both
    sides), against the number of detectors: physical (all photons) and Bala-compatible (merged)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"GRB230307A": "o", "GRB211211A": "s"}
    print("\nP4 values (SNR at Bala et al.'s median MVT, divided by Table 3)")
    print(f"{'GRB':11s}{'run':4s}{'dets':>5s}{'Bala MVT':>10s}{'Table 3':>9s}{'physical':>10s}{'ratio':>7s}"
          f"{'Bala-comp':>11s}{'ratio':>7s}")
    # rows grouped by number of detectors (one column each, both bursts together); within a column every row gets
    # its own colour and position, and the letters alternate above and below the markers
    columns = {}
    for grb in GRBS:
        for r in sorted(allRows[grb], key=lambda r: r["bw_ms"]):
            columns.setdefault(SET_N[r["set"]], []).append((grb, r))
    for n, members in sorted(columns.items()):
        for j, (grb, r) in enumerate(members):
            x = n * P4_ROW_SPACING ** (j - (len(members) - 1) / 2)
            color = ROW_COLORS[j % len(ROW_COLORS)]
            rp, rb = r["snrPhysAtB"] / r["bSnr"], r["snrBAtB"] / r["bSnr"]
            ax.plot(x, rp, markers[grb], mfc="none", color=color, ms=7)
            ax.plot(x, rb, markers[grb], color=color, ms=6)
            above = j % 2 == 0
            ax.annotate(r["run"], (x, rp), textcoords="offset points", xytext=(0, 7 if above else -8),
                        ha="center", va="bottom" if above else "top", fontsize=7, color=color)
            print(f"{grb:11s}{r['run']:4s}{n:>5d}{r['bMed']:>10.1f}{r['bSnr']:>9.1f}{r['snrPhysAtB']:>10.1f}{rp:>7.3f}"
                  f"{r['snrBAtB']:>11.1f}{rb:>7.3f}")
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.margins(y=0.1)                                     # room for the letters above the top markers
    for grb, m in markers.items():
        ax.plot([], [], m, color="0.3", mfc="none", label=f"{grb}: physical SNR (all photons)")
        ax.plot([], [], m, color="0.3", label=f"{grb}: Bala-compatible SNR (merged)")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 7, 12])
    ax.set_xticklabels(["1", "3", "5", "7", "12"])
    ax.set_xlabel("number of detectors combined")
    ax.set_ylabel("SNR_MVT / Bala et al. Table 3 SNR_MVT")
    ax.set_title("SNR_MVT compared with Bala et al. (at their median MVT)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "P4_snr_ratio")


def loadRealisations(folder):
    """MVT values (ms) of the 300 realisations of one run, split as in run_grb_mvt.py: failed (not finite or
    <= 0), upper limit (uncertainty not > 0), valid (the rest)."""
    v = np.load(os.path.join(folder, "mvt_values_seconds.npy")) * 1e3
    u = np.load(os.path.join(folder, "mvt_uncertainties_seconds.npy")) * 1e3
    failed = ~np.isfinite(v) | (v <= 0)
    limit = ~failed & ~(u > 0)
    valid = ~failed & ~limit
    return v, valid, limit, failed


def plotP5(allRows, run=P5_RUN):
    """Distribution of the 300 Monte Carlo MVT values of one GRB 230307A row: all photons against merged."""
    ctrlDir = os.path.join(RESULTS, "bala_rows_8-900keV_rs_haarfix", f"Run{run}")
    mergedDir = os.path.join(RESULTS, "merged_mc", f"GRB230307A_Run{run}")
    bala = json.load(open(os.path.join(mergedDir, "summary.json")))["bala"]
    saved = {"all photons": json.load(open(os.path.join(ctrlDir, "summary.json"))),
             "merged": json.load(open(os.path.join(mergedDir, "summary.json")))["test"]}
    data = {"all photons": loadRealisations(ctrlDir), "merged": loadRealisations(mergedDir)}
    finite = np.concatenate([v[~f] for v, _, _, f in data.values()])
    edges = np.logspace(np.log10(finite.min()), np.log10(finite.max()), 40)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    print(f"\nP5 values (GRB230307A row {run}, 300 realisations)")
    print(f"{'':13s}{'valid':>7s}{'UL':>5s}{'fail':>6s}{'median':>8s}{'p16':>7s}{'p84':>7s} | saved: "
          f"{'valid%':>7s}{'median':>8s}{'p84':>7s}")
    for ax, (name, (v, valid, limit, failed)) in zip(axes, data.items()):
        med, p16, p84 = np.median(v[valid]), *np.percentile(v[valid], [16, 84])
        s = saved[name]
        sValid = s.get("validPercent", s.get("valid"))
        sMed = s.get("median_ms", s.get("median"))
        sP84 = s.get("p84_ms", s.get("p84"))
        print(f"{name:13s}{valid.sum():>7d}{limit.sum():>5d}{failed.sum():>6d}{med:>8.2f}{p16:>7.2f}{p84:>7.2f} | "
              f"{'':7s}{sValid:>7.1f}{sMed:>8.2f}{sP84:>7.2f}")
        if abs(100 * valid.sum() / v.size - sValid) > 0.05 or abs(med - sMed) > 0.005 or abs(p84 - sP84) > 0.005:
            print(f"  WARNING: {name} does not match its saved summary.json")
        ax.axvspan(bala["median"] - bala["minus"], bala["median"] + bala["plus"], color="0.88",
                   label=f"Bala et al. {bala['median']} (+{bala['plus']}/-{bala['minus']}) ms")
        ax.axvline(bala["median"], color="0.5", lw=1)
        ax.hist(v[valid], bins=edges, color="C0", alpha=0.8, label=f"valid ({valid.sum()})")
        ax.hist(v[limit], bins=edges, histtype="step", color="C3", lw=1.5, label=f"upper limit ({limit.sum()})")
        ax.axvline(med, color="k", lw=1.5, label=f"median of valid {med:.2f} ms")
        ax.axvline(p16, color="k", lw=1, ls="--", label=f"16-84%: {p16:.2f}-{p84:.2f} ms")
        ax.axvline(p84, color="k", lw=1, ls="--")
        ax.plot([], [], " ", label=f"failed: {failed.sum()}")
        ax.set_ylabel("number of realisations")
        ax.set_title(f"GRB230307A row {run}, {name}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, ls=":", alpha=0.5)
    axes[-1].set_xscale("log")
    ticks = [t for t in (0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20) if edges[0] <= t <= edges[-1]]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{t:g}" for t in ticks])
    axes[-1].minorticks_off()
    axes[-1].set_xlabel("MVT of each realisation (ms)")
    save(fig, f"P5_mvt_distribution_{run}")


def plotP6(allRows, run=P6_RUN):
    """GRB 230307A light curve (8-900 keV) in bins of P6_BIN_MS, all photons against merged, from the saved
    0.01 ms light curves of one row: whole interval, zoom on the brightest bin, and the fraction lost per bin."""
    ctrlDir = os.path.join(RESULTS, "bala_rows_8-900keV_rs_haarfix", f"Run{run}")
    mergedDir = os.path.join(RESULTS, "merged_mc", f"GRB230307A_Run{run}")
    s = json.load(open(os.path.join(mergedDir, "summary.json")))      # written by run_merged_mc.py
    bwS, (tmin, tmax) = s["binSize_s"], s["timeRange_s"]
    ctrl = np.load(os.path.join(ctrlDir, "counts.npy"))
    merged = np.load(os.path.join(mergedDir, "counts.npy"))
    if ctrl.size != merged.size:
        sys.exit("STOP: the two saved light curves have different numbers of bins")
    if int(ctrl.sum()) != s["photonsControl"] or int(merged.sum()) != s["photonsMerged"]:
        print(f"  WARNING: light-curve totals {int(ctrl.sum())}, {int(merged.sum())} differ from the saved "
              f"summary {s['photonsControl']}, {s['photonsMerged']}")
    k = int(round(P6_BIN_MS / 1e3 / bwS))                 # fine bins per plotted bin
    n = ctrl.size // k
    c = ctrl[:n * k].reshape(n, k).sum(axis=1)
    m = merged[:n * k].reshape(n, k).sum(axis=1)
    t = tmin + (np.arange(n) + 0.5) * k * bwS             # bin centres (s)
    lost = np.where(c > 0, 1 - m / np.where(c > 0, c, 1), 0.0)
    ip = int(np.argmax(c))
    print(f"\nP6 values (GRB230307A row {run} light curves, {bwS * 1e3:g} ms bins summed to {k * bwS * 1e3:g} ms)")
    print(f"total counts: all photons {int(ctrl.sum())}, merged {int(merged.sum())}, "
          f"lost {100 * (1 - merged.sum() / ctrl.sum()):.1f}%")
    print(f"brightest bin at t = {t[ip]:.4f} s: all photons {int(c[ip])}, merged {int(m[ip])}, lost {100 * lost[ip]:.1f}%")
    print(f"merged peak (any bin): {int(m.max())} at t = {t[int(np.argmax(m))]:.4f} s; "
          f"peak lost {100 * (1 - m.max() / c.max()):.1f}%")
    top10 = np.argsort(c)[-10:]
    print(f"10 brightest bins: mean lost {100 * lost[top10].mean():.1f}%;  "
          f"bins below 10% of the peak: mean lost {100 * lost[c < 0.1 * c.max()].mean():.1f}%")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), gridspec_kw=dict(height_ratios=[2, 2, 1.3]))
    ax = axes[0]
    ax.step(t, c, where="mid", color="C0", lw=0.6, label="all photons")
    ax.step(t, m, where="mid", color="C3", lw=0.6, label="merged")
    ax.set_xlim(tmin, tmax)
    ax.set_title(f"GRB230307A, row {run} detectors, 8-900 keV, {P6_BIN_MS:g} ms bins", fontsize=10)
    ax2 = axes[1]
    zoom = np.abs(t - t[ip]) <= P6_ZOOM_S
    ax2.step(t[zoom], c[zoom], where="mid", color="C0", lw=1, label="all photons")
    ax2.step(t[zoom], m[zoom], where="mid", color="C3", lw=1, label="merged")
    ax2.set_title(f"zoom: {P6_ZOOM_S:g} s either side of the brightest bin (t = {t[ip]:.3f} s)", fontsize=10)
    for a in (ax, ax2):
        a.set_xlabel("time since trigger (s)")
        a.set_ylabel(f"counts per {P6_BIN_MS:g} ms")
        a.grid(True, ls=":", alpha=0.5)
        a.legend(fontsize=8)
    ax3 = axes[2]
    ax3.plot(c, 100 * lost, ".", ms=2, color="0.3")
    ax3.set_xlabel(f"counts per {P6_BIN_MS:g} ms bin (all photons)")
    ax3.set_ylabel("lost in merge (%)")
    ax3.set_title("fraction of each bin removed by the merge, against the bin's counts", fontsize=10)
    ax3.grid(True, ls=":", alpha=0.5)
    save(fig, f"P6_lightcurve_{run}")


PLOTS = {"P1": plotP1, "P2": plotP2, "P3": plotP3, "P4": plotP4, "P5": plotP5, "P6": plotP6}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(PLOTS), help="make only these plots (default: all)")
    args = parser.parse_args()
    todo = args.only or list(PLOTS)
    os.makedirs(PLOT_DIR, exist_ok=True)
    allRows = {grb: loadRuns(grb) for grb in GRBS}
    print("Plotted values (check against the --post tables)")
    print(f"{'GRB':11s}{'run':4s}{'set':6s}{'BW ms':>7s} | {'ours median (+/-)':>22s}{'SNR-B':>8s}{'SNR':>8s} | "
          f"{'Bala median (+/-)':>20s}{'SNR':>8s}")
    for grb in GRBS:
        for r in allRows[grb]:
            print(f"{grb:11s}{r['run']:4s}{r['set']:6s}{r['bw_ms']:>7g} | "
                  f"{r['med']:>8.2f} (+{r['plus']:.2f}/-{r['minus']:.2f}){r['snrB']:>8.1f}{r['snrPhys']:>8.1f} | "
                  f"{r['bMed']:>8.1f} (+{r['bPlus']:.1f}/-{r['bMinus']:.1f}){r['bSnr']:>8.1f}")
    print()
    for name in todo:
        PLOTS[name](allRows)
