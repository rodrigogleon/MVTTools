"""
A4: are the MVT clusters tied to the timescale ladder used by haarPowerMod? No simulations are rerun.

The ladder. haarPowerMod measures the Haar power only in bands of timescale, 4 bands per doubling
(outputBinningRatio = 4 in config.yaml), built inside haarNDWT from block sizes in bins. Each band's timescale
("rung") is 0.5 x (lower + upper block size) x bin width. The rungs depend only on the number of bins, the bin
width and the settings (binFactor 4, oversampling 32, totalSignalRepetitions 2), not on the counts, so they are
rebuilt here with the same steps as haarNDWT, without computing any power (fast for millions of bins). At start
the script checks this against haarNDWT itself for four light-curve lengths and stops if they differ (also checked
locally: haarNDWT's rungs are identical to those used inside haarPowerMod).

The test. For every valid MVT value, its position between the two rungs around it, in log units:
    u = (log MVT - log rung_below) / (log rung_above - log rung_below)     (0 = on a rung, 0.5 = halfway)
If the ladder did not matter, u would be spread evenly between 0 and 1 (uniform).

A4a. Injection settings (all pulses, amplitudes, bin widths): u for each setting; Kolmogorov-Smirnov test
     against a uniform distribution; histograms of the values with the rungs drawn, and of u pooled by bin
     width and brightness group (faint: amplitude <= 3; moderate: 10-30; bright: >= 100).
A4b. Real bursts: the same for every saved Monte Carlo run (GRB 230307A and GRB 211211A rows, merged runs),
     using each run's saved light curve (counts.npy) and bin width (summary.json).

Decision rules (fixed before running)
  R4a. A setting with at least 30 valid values is "tied to the ladder" if the KS p-value is below 0.01.
       Over those settings: the effect is general if more than half are tied; absent if fewer than 5%; partial
       otherwise.
  R4b. The same rule, run by run, for the real bursts (reported as a count).
  Also reported, not part of the rules: the share of u values within 0.20-0.35 and 0.65-0.80 (the two positions
  seen in a local preview of G3 at amplitude 3000, 0.01 ms; uniform would give 30% in total).

Output: results/injection_recovery/analysis_A4/summary.json and, in results/plots/: IR_rungs_<pulse>.png,
IR_rung_position.png, REAL_rungs.png, REAL_rung_position.png (each also .pdf)

Usage (from the scripts folder, next to run_injection_recovery.py)
  python analyse_rung_positions.py
"""
import os, sys, json, glob, logging
import numpy as np
from scipy.stats import kstest
import run_injection_recovery as ir

MIN_VALID = 30
P_CUT = 0.01
HIST_AMPS = [3.0, 30.0, 300.0, ir.BRIGHT_AMPLITUDE]
GROUPS = [("faint (amp <= 3)", lambda a: a <= 3), ("moderate (amp 10-30)", lambda a: 10 <= a <= 30),
          ("bright (amp >= 100)", lambda a: a >= 100)]
REAL_ROOTS = [os.path.join(ir.RESULTS, "bala_rows_8-900keV_rs_haarfix"),
              os.path.join(ir.RESULTS, "bala_rows_211211A"),
              os.path.join(ir.RESULTS, "merged_mc")]
OUT = os.path.join(ir.OUT_DIR, "analysis_A4")

_cache = {}


def loadHaar():
    sys.path.insert(0, ir.REPO)
    here = os.getcwd()
    os.chdir(ir.REPO)
    from loggerSetup import initialiseLogging
    initialiseLogging()
    logging.disable(logging.CRITICAL)
    import analysisTools.haarMethods as hm
    os.chdir(here)
    return hm


def rungsFast(nBins, bw, binFactor=4, oversample=32.0, reps=2):
    """The rung timescales (ms) exactly as haarNDWT builds them, without computing any power: the scale lists of
    generateScales and the band loop, including the shared band edges (lowerBoundScales and upperBoundScales are
    views of the same array in haarNDWT, so tightening one band also moves its neighbour's edge)."""
    lengthOfData = (nBins + 1) + (reps - 1) * (nBins - 1) - 1
    maxIndex = (lengthOfData - 1).bit_length()

    def generateScales(perOctave):
        dyadic = 1 << np.arange(maxIndex, dtype="int32")
        inter = 2.0 * np.round((2.0 ** (np.arange(maxIndex * perOctave, dtype="float64") / perOctave)) / 2.0)
        comb = np.unique(np.hstack((dyadic, inter)).astype("int32"))
        return comb[(comb > 0) & (2 * comb <= lengthOfData)]

    fine = generateScales(oversample)
    edges = generateScales(binFactor) if binFactor < oversample else fine.copy()
    lower, upper = edges[:-1], edges[1:]
    for k in range(len(lower)):
        m = (fine >= lower[k]) & (fine < upper[k])
        if np.count_nonzero(m) > 0:
            lower[k] = fine[m].min()
            upper[k] = fine[m].max()
    return np.unique(0.5 * (lower.astype("float64") + upper.astype("float64")) * bw * 1e3)


def rungsSlow(hm, nBins, bw):
    """Reference: the rungs from haarNDWT itself, run on a dummy light curve of the same length."""
    one = np.ones(nBins)
    lo, up, _, _, _ = hm.haarNDWT(one, one, one, binSizeInSeconds=bw, timeSeriesOversampleFactor=4 * 8.0,
                                  totalSignalRepetitions=2, binFactor=4)
    return np.unique(0.5 * (lo + up) * 1e3)


def rungs(hm, nBins, bw):
    """Rung timescales (ms) used by haarPowerMod for a light curve of nBins bins of width bw (s)."""
    k = (nBins, bw)
    if k not in _cache:
        _cache[k] = rungsFast(nBins, bw)
    return _cache[k]


def selfCheck(hm):
    """Stop if the fast rungs differ from haarNDWT's for a few light-curve lengths used here."""
    for n, bw in ((1500, 1e-4), (15000, 1e-5), (50000, 3e-6), (12345, 1e-5)):
        if not np.allclose(rungsFast(n, bw), rungsSlow(hm, n, bw), rtol=1e-12, atol=0):
            sys.exit(f"STOP: rung reconstruction differs from haarNDWT for {n} bins of {bw} s")
    print("self-check: rung reconstruction identical to haarNDWT (4 light-curve lengths)")


def positions(values_ms, R):
    """u of each value between its two neighbouring rungs; values outside the ladder are dropped."""
    v = np.asarray(values_ms, float)
    v = v[(v > R[0]) & (v < R[-1])]
    i = np.searchsorted(R, v) - 1
    return (np.log(v) - np.log(R[i])) / (np.log(R[i + 1]) - np.log(R[i]))


def stats(u):
    if u.size < 2:
        return dict(n=int(u.size), ksP=float("nan"), near=float("nan"))
    near = np.mean(((u >= 0.20) & (u <= 0.35)) | ((u >= 0.65) & (u <= 0.80)))
    return dict(n=int(u.size), ksP=float(kstest(u, "uniform").pvalue), near=float(near))


def valid(mvt, unc):
    mvt, unc = np.asarray(mvt, float), np.asarray(unc, float)
    return np.isfinite(mvt) & (mvt > 0) & (unc > 0)


def verdict(items):
    judged = [x for x in items if x["n"] >= MIN_VALID]
    tied = [x for x in judged if x["ksP"] < P_CUT]
    frac = len(tied) / len(judged) if judged else float("nan")
    v = "not judged" if not judged else ("general" if frac > 0.5 else ("absent" if frac < 0.05 else "partial"))
    return dict(judged=len(judged), tied=len(tied), fraction=frac, verdict=v)


def injection(hm):
    rows, pooled = [], {}
    print("\n===== A4a. POSITION BETWEEN RUNGS (injection settings) =====")
    print(f"{'pulse':8s}{'amp':>6s}{'BW ms':>7s}{'bins':>8s}{'valid':>6s}{'in ladder':>10s}{'KS p':>10s}{'near %':>8s}"
          f"{'rung step':>10s}")
    for key in ir.PULSES:
        tstart, tstop = ir.window(ir.PULSES[key])
        for amp in ir.AMPLITUDES + [ir.BRIGHT_AMPLITUDE]:
            path = os.path.join(ir.OUT_DIR, "settings", f"{key}_{ir.ampTag(amp)}.json")
            if not os.path.exists(path):
                print(f"WARNING: missing {path}")
                continue
            rec = json.load(open(path))
            for pb in rec["bins"]:
                bw = pb["binWidth_s"]
                nBins = ir.lightCurve(np.array([]), tstart, tstop, bw).size
                R = rungs(hm, nBins, bw)
                ok = valid(pb["mvt_ms"], pb["unc_ms"])
                vals = np.asarray(pb["mvt_ms"], float)[ok]
                u = positions(vals, R)
                s = stats(u)
                med = np.median(vals) if vals.size else np.nan
                step = float("nan")
                if np.isfinite(med) and R[0] < med < R[-1]:
                    j = np.searchsorted(R, med) - 1
                    step = R[j + 1] / R[j]
                row = dict(pulse=key, amplitude=amp, binWidth_ms=bw * 1e3, nBins=int(nBins), nValid=int(ok.sum()),
                           rungStepAtMedian=step, **s)
                rows.append(row)
                for name, test in GROUPS:
                    if test(amp):
                        pooled.setdefault((bw, name), []).append(u)
                print(f"{key:8s}{amp:>6g}{bw * 1e3:>7g}{nBins:>8d}{int(ok.sum()):>6d}{s['n']:>10d}{s['ksP']:>10.2g}"
                      f"{(100 * s['near'] if np.isfinite(s['near']) else float('nan')):>8.0f}{step:>10.3f}")
    v = verdict(rows)
    print(f"\nR4a: settings with >= {MIN_VALID} valid values inside the ladder: {v['judged']}; tied to the ladder "
          f"(KS p < {P_CUT}): {v['tied']} ({100 * v['fraction']:.0f}%) -> effect {v['verdict'].upper()}")
    for bw in ir.BIN_WIDTHS:
        sub = [x for x in rows if abs(x["binWidth_ms"] - bw * 1e3) < 1e-12 and x["n"] >= MIN_VALID]
        if sub:
            print(f"    bin width {bw * 1e3:g} ms: {sum(x['ksP'] < P_CUT for x in sub)}/{len(sub)} tied")
    for name, test in GROUPS:
        sub = [x for x in rows if test(x["amplitude"]) and x["n"] >= MIN_VALID]
        if sub:
            print(f"    {name}: {sum(x['ksP'] < P_CUT for x in sub)}/{len(sub)} tied, median share near 0.2-0.35 "
                  f"or 0.65-0.8: {100 * np.median([x['near'] for x in sub]):.0f}% (uniform: 30%)")
    return rows, pooled, v


def real(hm):
    rows, data = [], []
    print("\n===== A4b. POSITION BETWEEN RUNGS (real bursts, saved Monte Carlo runs) =====")
    print(f"{'run folder':45s}{'BW ms':>7s}{'bins':>9s}{'valid':>6s}{'in ladder':>10s}{'KS p':>10s}{'near %':>8s}")
    for root in REAL_ROOTS:
        for f in sorted(glob.glob(os.path.join(root, "*", "mvt_values_seconds.npy"))):
            d = os.path.dirname(f)
            need = [os.path.join(d, x) for x in ("mvt_uncertainties_seconds.npy", "counts.npy", "summary.json")]
            name = os.path.relpath(d, ir.RESULTS)
            if not all(os.path.exists(x) for x in need):
                print(f"{name:45s}  skipped (missing {[os.path.basename(x) for x in need if not os.path.exists(x)]})")
                continue
            mvt, unc = np.load(f) * 1e3, np.load(need[0]) * 1e3
            nBins = int(np.load(need[1]).size)
            bw = float(json.load(open(need[2]))["binSize_s"])
            R = rungs(hm, nBins, bw)
            vals = mvt[valid(mvt, unc)]
            u = positions(vals, R)
            s = stats(u)
            rows.append(dict(run=name, binWidth_ms=bw * 1e3, nBins=nBins, nValid=int(vals.size), **s))
            data.append((name, vals, R, u))
            print(f"{name:45s}{bw * 1e3:>7g}{nBins:>9d}{int(vals.size):>6d}{s['n']:>10d}{s['ksP']:>10.2g}"
                  f"{(100 * s['near'] if np.isfinite(s['near']) else float('nan')):>8.0f}")
    v = verdict(rows)
    print(f"\nR4b: real runs with >= {MIN_VALID} valid values inside the ladder: {v['judged']}; tied to the ladder: "
          f"{v['tied']} ({100 * v['fraction']:.0f}%) -> effect {v['verdict'].upper()}")
    return rows, data, v


def plots(hm, pooled, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(ir.PLOT_DIR, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(ir.PLOT_DIR, f"{name}.{ext}"), dpi=130)
        plt.close(fig)

    def valuePanel(ax, vals, R, title):
        if vals.size >= 2:
            ax.hist(vals, bins=np.linspace(vals.min() * 0.97, vals.max() * 1.03, 120), color="C0")
            for r in R[(R > vals.min() * 0.9) & (R < vals.max() * 1.1)]:
                ax.axvline(r, color="0.5", lw=0.7, ls=":")
            ax.axvline(np.median(vals), color="k", lw=1.2)
        ax.set_title(title, fontsize=8)

    # values with the rungs, per pulse
    for key in ir.PULSES:
        tstart, tstop = ir.window(ir.PULSES[key])
        fig, axes = plt.subplots(len(ir.BIN_WIDTHS), len(HIST_AMPS), figsize=(4 * len(HIST_AMPS), 3 * len(ir.BIN_WIDTHS)))
        for j, amp in enumerate(HIST_AMPS):
            path = os.path.join(ir.OUT_DIR, "settings", f"{key}_{ir.ampTag(amp)}.json")
            rec = json.load(open(path)) if os.path.exists(path) else None
            for i, bw in enumerate(ir.BIN_WIDTHS):
                ax = axes[i, j]
                pb = next((p for p in rec["bins"] if abs(p["binWidth_s"] - bw) < 1e-15), None) if rec else None
                if pb is None:
                    ax.set_axis_off()
                    continue
                R = rungs(hm, ir.lightCurve(np.array([]), tstart, tstop, bw).size, bw)
                vals = np.asarray(pb["mvt_ms"], float)[valid(pb["mvt_ms"], pb["unc_ms"])]
                valuePanel(ax, vals, R, f"{key}, amp {amp:g}, {bw * 1e3:g} ms")
                if i == len(ir.BIN_WIDTHS) - 1:
                    ax.set_xlabel("MVT (ms); dotted = rungs; black = median")
        save(fig, f"IR_rungs_{key}")

    # pooled positions between rungs, injection
    fig, axes = plt.subplots(len(ir.BIN_WIDTHS), len(GROUPS), figsize=(4 * len(GROUPS), 3 * len(ir.BIN_WIDTHS)),
                             sharex=True)
    for i, bw in enumerate(ir.BIN_WIDTHS):
        for j, (name, _) in enumerate(GROUPS):
            ax = axes[i, j]
            u = np.concatenate(pooled.get((bw, name), [np.array([])]))
            ax.hist(u, bins=np.linspace(0, 1, 41), color="C0")
            if u.size:
                ax.axhline(u.size / 40, color="C3", lw=1, label="uniform")
            ax.set_title(f"{bw * 1e3:g} ms, {name}, n = {u.size}", fontsize=8)
            if i == len(ir.BIN_WIDTHS) - 1:
                ax.set_xlabel("position between rungs u (0 = on a rung)")
    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
            break
    save(fig, "IR_rung_position")

    # real bursts: values with rungs, and positions
    if data:
        n = len(data)
        cols = 4
        nr = int(np.ceil(n / cols))
        fig, axes = plt.subplots(nr, cols, figsize=(4 * cols, 2.8 * nr), squeeze=False)
        fig2, axes2 = plt.subplots(nr, cols, figsize=(4 * cols, 2.8 * nr), squeeze=False)
        for k, (name, vals, R, u) in enumerate(data):
            valuePanel(axes[k // cols, k % cols], vals, R, name)
            ax = axes2[k // cols, k % cols]
            ax.hist(u, bins=np.linspace(0, 1, 41), color="C0")
            if u.size:
                ax.axhline(u.size / 40, color="C3", lw=1)
            ax.set_title(f"{name}, n = {u.size}", fontsize=8)
        for k in range(n, nr * cols):
            axes[k // cols, k % cols].set_axis_off()
            axes2[k // cols, k % cols].set_axis_off()
        save(fig, "REAL_rungs")
        save(fig2, "REAL_rung_position")
    print(f"\nSaved plots in {ir.PLOT_DIR}: IR_rungs_<pulse>, IR_rung_position, REAL_rungs, REAL_rung_position (.png/.pdf)")


if __name__ == "__main__":
    hm = loadHaar()
    selfCheck(hm)
    rowsI, pooled, vI = injection(hm)
    rowsR, data, vR = real(hm)
    plots(hm, pooled, data)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(dict(R4a=vI, R4b=vR, injection=rowsI, real=rowsR), f, indent=1, default=float)
    print(f"Saved {OUT}/summary.json")
