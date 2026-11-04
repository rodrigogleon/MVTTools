"""
MVT of the precursor of GRB 230307A (sprint step 8), with the same workflow as run_grb_mvt.py, and three checks
that use what the injection-recovery runs taught us.

Interval and data
  Precursor interval -0.064 to 0.32 s (Dichiara et al. 2023, ApJL 954, L29), 8-900 keV (as all our runs and
  Bala et al.; Dichiara et al. used 8-1000 keV and 15-350 keV), Fermi GBM NaI TTE data. The interval is well
  before the 3-7 s data loss. Background: first-order polynomial on the off-burst intervals [-68.8, -18.8] and
  [69.4, 119.4] s, 1.024 s bins (as run_grb_mvt.py). Because the precursor interval (0.384 s) is shorter than
  one 1.024 s background bin, the background rate over it is the overlap-weighted mean of the fitted bins that
  overlap the interval (run_grb_mvt.py uses only bins fully inside the interval, which here would be none).

Step A. Detector selection (Bala et al. 2026, Section 4.1, as run_detector_selection.py)
  Rank the 12 NaI detectors by SNR in the precursor interval, add them in that order, keep the set with the
  highest combined SNR. First with 16 ms bins (the interval is shorter than 1 s), then with bins equal to the
  MVT: before the Monte Carlo the MVT of Dichiara et al. (17 ms) is used; after it, the selection is repeated
  with our measured MVT and the script warns if the set changes (then rerun with --dets).
  Two detector sets are measured: the optimal set ("opt") and all 12 NaI detectors ("all"), as for the bursts.

Step B. MVT (as run_grb_mvt.py)
  For each set and bin width 0.1, 0.01 and 0.003 ms: light curve of all photons (detector counts added), single
  MVT, and 300 Monte Carlo realisations with resampled errors (the real-data scheme), seed 2026, 4 rungs per
  doubling. Median and 16-84% of the valid values; valid / upper limit / failed.

Step C. SNR and class (as run_grb_mvt.py), both SNR measures recorded
  Physical SNR_MVT: all photons, 8-900 keV background, at our median MVT (observed; min-max over 10 bin phases;
  mean of 300 Poisson resamples). Bala-compatible SNR_MVT: detectors combined with GbmTte.merge and the
  background over all channels; used for the class from Bala et al.'s validation curve (as before).

Checks and rules (fixed before running)
  R8a plateau: the medians at 0.01 and 0.003 ms agree within 10%. The quoted value is the one at the smallest
      bin width (Bala et al.'s workflow); if its class is UL or LUL, MVT84 is quoted as an upper limit.
  R8b Dichiara et al.: |ours - 17 ms| / sqrt(our error^2 + 2^2) <= 2 -> consistent (our error = the 16-84%
      half-width on the side facing 17 ms).
  R8c trust (from B5, precursor-like conditions): the physical SNR_MVT (observed) is at or above the highest B5
      precursor-condition threshold among the simulated pulses whose bright value R is within a factor 2 of our
      MVT -> "above the trust threshold"; otherwise "below: read the MVT as an upper estimate". B5 used 5000
      counts/s; the measured background is printed so the difference is visible (the made-to-measure check,
      next step, will use the real value).
  R8d background (C7, is it Poisson?): in the two off-burst intervals, for the optimal set, counts in bins of
      0.01, 0.1, 1, 10 and 100 ms; the expected count of each bin from a straight-line fit of the rate against
      time in each interval; Fano factor F = sum((n - mu)^2) / sum(mu) (1 for Poisson). Consistent with Poisson
      if |F - 1| <= 0.05 at every timescale up to 10 ms (the timescales that matter for this MVT); 100 ms is
      reported only (slow background changes can raise it without affecting the MVT).
  Detector set: the selection repeated at our measured MVT (optimal set, smallest bin width) gives the same set.

Output: results/precursor_mvt/GRB230307A/ (selection.json, <set>_<bin>/ with counts.npy, the Monte Carlo values
and summary.json, summary.json); plot results/plots/P11_precursor_230307A.png (.pdf too).

Usage (from the scripts folder)
  python run_precursor_mvt.py                    steps A, B, C and the checks (resumable: finished sets/bins are reused)
  python run_precursor_mvt.py --post             step C and the checks from saved results only
  python run_precursor_mvt.py --dets n0 n1 ...   force the optimal set (if the repeat selection asks for it)
"""
import os, sys, json, time, glob, argparse, subprocess
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
BALA_CODE = "/mnt/data/rgl36/FYP/external/GBM_MVT_paper/MVTfermi"
DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
RESULTS = "/mnt/data/rgl36/FYP/results"

GRB = "GRB230307A"
INTERVAL = (-0.064, 0.32)
BKG_INTERVALS = [(-68.8, -18.8), (69.4, 119.4)]
ENERGY_KEV = (8.0, 900.0)
BIN_WIDTHS = [1e-4, 1e-5, 3e-6]
N_REALISATIONS = 300
SEED = 2026
BKG_BIN_S = 1.024
BKG_ORDER = 1
N_PHASES = 10
FIRST_SELECTION_BIN_S = 0.016
DICHIARA_MS = (17.0, 2.0)
PLATEAU_TOL = 0.10
FANO_SCALES_S = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
FANO_TOL = 0.05
FANO_JUDGED_MAX_S = 1e-2
PLOT_WINDOW_S = (-1.0, 2.0)
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]
OUT = os.path.join(RESULTS, "precursor_mvt", GRB)
PLOT_DIR = os.path.join(RESULTS, "plots")
B5_SUMMARY = os.path.join(RESULTS, "injection_recovery", "b3_threshold", "summary.json")

parser = argparse.ArgumentParser()
parser.add_argument("--post", action="store_true", help="step C and checks from saved results only")
parser.add_argument("--dets", nargs="+", choices=ALL_NAI, help="force the optimal detector set")
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--npoisson", type=int, default=300, help="Poisson resamples for the mean SNR")
args = parser.parse_args()

sys.path.insert(0, REPO)
sys.path.insert(1, BALA_CODE)
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
import clasify_mvt_value as validation

validation.MODEL_FILE = os.path.join(BALA_CODE, "mvt_snr_fit_model.npz")
SHORT_NAMES = {"Below 95% CI (Upper Limit)": "UL", "Within 95% CI (Likely Upper Limit)": "LUL",
               "Above 95% CI (Robust Measurement)": "robust"}
ms = lambda x: float(x) * 1e3


def tteFile(det):
    files = glob.glob(f"{DATA_ROOT}/{GRB}/current/glg_tte_{det}_bn*_v0*.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    return files[0]


# ---------------------------------------------------------------- data and background
def loadDetector(det):
    """8-900 keV event times (only where they are used: the plot window -1 to 2 s, which contains the precursor
    interval, and the off-burst intervals), background fit, and the mean fitted rates over the interval."""
    tte = GbmTte.open(tteFile(det))
    t = np.array(tte.slice_energy(ENERGY_KEV).data.times)
    keep = (t >= PLOT_WINDOW_S[0]) & (t < PLOT_WINDOW_S[1])
    for a, b in BKG_INTERVALS:
        keep |= (t >= a) & (t < b)
    timesBand = np.sort(t[keep])
    del t, keep
    phaii = tte.to_phaii(bin_by_time, BKG_BIN_S, time_ref=0.0)
    fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=BKG_INTERVALS)
    fitter.fit(order=BKG_ORDER)
    model = fitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
    tte.close()
    tstart, tstop = np.asarray(model.tstart), np.asarray(model.tstop)
    ratesBand = np.asarray(model.integrate_energy(*ENERGY_KEV).rates).ravel()
    ratesAll = np.asarray(model.integrate_energy().rates).ravel()
    stat, dof = np.asarray(fitter.statistic), np.asarray(fitter.dof)
    return dict(timesBand=timesBand,
                bkgBand=overlapMean(tstart, tstop, ratesBand, INTERVAL),
                bkgAll=overlapMean(tstart, tstop, ratesAll, INTERVAL),
                redChi2=float(np.nansum(stat) / np.nansum(dof)) if np.nansum(dof) > 0 else np.nan)


def overlapMean(tstart, tstop, rates, interval):
    """Mean fitted rate over the interval, each background bin weighted by its overlap with the interval."""
    w = np.clip(np.minimum(tstop, interval[1]) - np.maximum(tstart, interval[0]), 0.0, None)
    if w.sum() <= 0:
        raise ValueError(f"no background bin overlaps {interval}")
    return float(np.sum(w * rates) / w.sum())


def inInterval(times):
    return times[(times >= INTERVAL[0]) & (times < INTERVAL[1])]


def lightCurve(times, binS, tmin, tmax, phase=0.0):
    edges = np.arange(tmin + phase * binS, tmax + binS, binS)
    return np.histogram(times, bins=edges)[0]


def snrMvt(maxCounts, bkgCps, mvtS):
    return maxCounts / np.sqrt(bkgCps * mvtS)


# ---------------------------------------------------------------- step A: detector selection
def selectDetectors(data, binS):
    """Bala et al.'s iterative SNR maximisation (as run_detector_selection.py)."""
    curves = {d: lightCurve(inInterval(data[d]["timesBand"]), binS, *INTERVAL) for d in ALL_NAI}
    single = {d: float(snrMvt(curves[d].max(), data[d]["bkgBand"], binS)) for d in ALL_NAI}
    ranked = sorted(single, key=single.get, reverse=True)
    rows, combined, bkgSum = [], np.zeros_like(curves[ranked[0]]), 0.0
    for k, det in enumerate(ranked, 1):
        combined = combined + curves[det]
        bkgSum += data[det]["bkgBand"]
        rows.append(dict(k=k, added=det, snr=float(snrMvt(combined.max(), bkgSum, binS)), peak=int(combined.max())))
    best = max(rows, key=lambda r: r["snr"])
    chosen = sorted(ranked[:best["k"]], key=ALL_NAI.index)
    print(f"\nselection, bin {binS * 1e3:g} ms: individual SNR " + "  ".join(f"{d} {single[d]:.1f}" for d in ranked))
    print("   cumulative: " + "  ".join(f"+{r['added']} {r['snr']:.1f}" for r in rows))
    print(f"   optimal set ({best['k']} detectors, SNR {best['snr']:.1f}): {chosen}")
    return dict(binWidth_s=binS, single=single, ranked=ranked, cumulative=rows, optimal=chosen, optimalSnr=best["snr"])


# ---------------------------------------------------------------- step B: Monte Carlo (as run_grb_mvt.py)
def classifyRealisation(mvt, unc):
    if not np.isfinite(mvt) or mvt <= 0.0:
        return "failed"
    if not unc > 0.0:
        return "upper limit"
    return "valid"


def monteCarlo(setName, dets, bw, data, haarMethods, commit):
    outDir = os.path.join(OUT, f"{setName}_{bw * 1e3:g}ms")
    path = os.path.join(outDir, "summary.json")
    if os.path.exists(path):
        s = json.load(open(path))
        if s["detectors"] == dets:
            print(f"{setName} {bw * 1e3:g} ms: already done, reused")
            return s
        sys.exit(f"{path} was made with detectors {s['detectors']}, not {dets}: move that folder away first")
    os.makedirs(outDir, exist_ok=True)
    times = np.sort(np.concatenate([inInterval(data[d]["timesBand"]) for d in dets]))
    counts = lightCurve(times, bw, *INTERVAL).astype(float)
    errors = np.sqrt(counts)
    np.save(os.path.join(outDir, "counts.npy"), counts)
    t0 = time.time()
    (_, _, singleMvt, singleUnc, _, _, _) = haarMethods.haarPowerMod(counts, errors, minimumBinSizeSeconds=bw)
    r = haarMethods.estimateMvtDistribution(counts, errors, haarMethods.haarPowerMod, binSizeSeconds=bw,
                                            nRealisations=N_REALISATIONS, nWorkers=args.workers, seed=SEED,
                                            resampleErrors=True)
    values, uncs, n = r["mvtValues"], r["mvtUncertainties"], r["nRealisations"]
    np.save(os.path.join(outDir, "mvt_values_seconds.npy"), values)
    np.save(os.path.join(outDir, "mvt_uncertainties_seconds.npy"), uncs)
    limits = values[r["limitMask"]]
    s = dict(grb=GRB, part="precursor", detectorSet=setName, detectors=dets, binSize_s=bw, timeRange_s=list(INTERVAL),
             energyRange_keV=list(ENERGY_KEV), resampleErrors=True, seed=SEED, mvttoolsCommit=commit,
             totalCounts=int(counts.sum()), nBins=int(counts.size), meanCountsPerBin=float(counts.mean()),
             singleMvt_ms=ms(singleMvt), singleUnc_ms=ms(singleUnc), singleType=classifyRealisation(singleMvt, singleUnc),
             nRealisations=n, nValid=r["nValid"], nLimit=r["nLimit"], nFailed=r["nFailed"],
             validPercent=100 * r["nValid"] / n, limitPercent=100 * r["nLimit"] / n, failedPercent=100 * r["nFailed"] / n,
             median_ms=ms(r["median"]), p16_ms=ms(r["p16"]), p84_ms=ms(r["p84"]),
             plus_err_ms=ms(r["p84"] - r["median"]), minus_err_ms=ms(r["median"] - r["p16"]),
             limits_median_ms=ms(np.median(limits)) if len(limits) else None,
             minutes_montecarlo=(time.time() - t0) / 60)
    with open(path, "w") as f:
        json.dump(s, f, indent=2, default=float)
    print(f"{setName:4s} {bw * 1e3:>6g} ms: {s['totalCounts']} counts, {s['meanCountsPerBin']:.3f}/bin | single "
          f"{s['singleMvt_ms']:.2f} ± {s['singleUnc_ms']:.2f} ms ({s['singleType']}) | median {s['median_ms']:.2f} "
          f"(+{s['plus_err_ms']:.2f}/-{s['minus_err_ms']:.2f}) ms, valid {s['validPercent']:.0f}%, "
          f"limit {s['limitPercent']:.0f}%, failed {s['failedPercent']:.0f}%  [{s['minutes_montecarlo']:.1f} min]", flush=True)
    return s


# ---------------------------------------------------------------- step C: SNR and class (as run_grb_mvt.py)
def mergedTimes(dets):
    ttes = []
    for d in dets:
        tte = GbmTte.open(tteFile(d))
        ttes.append(tte.slice_time(INTERVAL))
        tte.close()
    return inInterval(np.array(GbmTte.merge(ttes).slice_energy(ENERGY_KEV).data.times))


def snrFor(s, data, model, rng):
    dets, mvtMs = s["detectors"], s["median_ms"]
    if not (mvtMs and np.isfinite(mvtMs) and mvtMs > 0):
        return None
    mvtS = mvtMs / 1e3
    times = np.sort(np.concatenate([inInterval(data[d]["timesBand"]) for d in dets]))
    bkgBand = sum(data[d]["bkgBand"] for d in dets)
    bkgAll = sum(data[d]["bkgAll"] for d in dets)
    counts = lightCurve(times, mvtS, *INTERVAL)
    phases = [snrMvt(lightCurve(times, mvtS, *INTERVAL, phase=k / N_PHASES).max(), bkgBand, mvtS) for k in range(N_PHASES)]
    poissonMax = np.array([rng.poisson(counts).max() for _ in range(args.npoisson)])
    balaDef = float(snrMvt(lightCurve(mergedTimes(dets), mvtS, *INTERVAL).max(), bkgAll, mvtS))
    interpolators, rangeLog = model
    label, bounds = validation.classify_mvt_value(mvtMs, balaDef, interpolators, rangeLog)
    lo, mid, hi = (float(10 ** x) for x in bounds)
    return dict(bkgCps_band=bkgBand, bkgCps_allChannels=bkgAll, maxCounts=int(counts.max()),
                bkgCountsPerBin=bkgBand * mvtS, physical=float(snrMvt(counts.max(), bkgBand, mvtS)),
                phaseMin=float(min(phases)), phaseMax=float(max(phases)),
                poissonMean=float(np.mean(snrMvt(poissonMax, bkgBand, mvtS))),
                poissonStd=float(np.std(snrMvt(poissonMax, bkgBand, mvtS))),
                balaCompatible=balaDef, cls=SHORT_NAMES.get(label, label), band=[lo, mid, hi],
                outsideCurve=not (rangeLog[0] <= np.log10(mvtMs) <= rangeLog[1]))


# ---------------------------------------------------------------- checks
def fano(data, dets):
    """R8d: Fano factor of the summed background counts at several timescales, in both off-burst intervals."""
    out = {}
    for tau in FANO_SCALES_S:
        num = den = 0.0
        nb = 0
        for a, b in BKG_INTERVALS:
            times = np.concatenate([d["timesBand"][(d["timesBand"] >= a) & (d["timesBand"] < b)] for d in (data[x] for x in dets)])
            nBins = int(np.floor((b - a) / tau + 1e-9))
            edges = a + tau * np.arange(nBins + 1)
            n = np.histogram(times, bins=edges)[0].astype(float)
            centres = 0.5 * (edges[:-1] + edges[1:])
            coarse = int(np.floor((b - a) / BKG_BIN_S))
            cEdges = a + BKG_BIN_S * np.arange(coarse + 1)
            cRate = np.histogram(times, bins=cEdges)[0] / BKG_BIN_S
            slope, icpt = np.polyfit(0.5 * (cEdges[:-1] + cEdges[1:]), cRate, 1)
            mu = (slope * centres + icpt) * tau
            num += np.sum((n - mu) ** 2)
            den += np.sum(mu)
            nb += n.size
        F = num / den
        meanMu = den / nb
        out[tau] = dict(F=float(F), err=float(np.sqrt((1.0 / meanMu + 2.0) / nb)), meanCountsPerBin=float(meanMu), nBins=nb)
    return out


def b5Thresholds():
    if not os.path.exists(B5_SUMMARY):
        return None
    d = json.load(open(B5_SUMMARY))["pulses"]
    rows = []
    for key, v in d.items():
        p = v.get("precursor", {})
        if p.get("threshold"):
            rows.append(dict(pulse=key, R=p["reference"], snr=p["threshold"]["snr"], snrBelow=p["threshold"]["snrBelow"]))
    return rows


def plot(data, summaries, optDets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    bw, (lo, hi) = 0.004, PLOT_WINDOW_S
    times = np.sort(np.concatenate([data[d]["timesBand"] for d in optDets]))
    times = times[(times >= lo) & (times < hi)]
    edges = np.arange(lo, hi + bw, bw)
    a1.step(edges[:-1], np.histogram(times, bins=edges)[0], where="post", color="k", lw=0.7)
    a1.axhline(sum(data[d]["bkgBand"] for d in optDets) * bw, color="C3", ls="--", lw=1, label="fitted background")
    a1.axvspan(*INTERVAL, color="C0", alpha=0.15, label=f"precursor interval {list(INTERVAL)} s")
    a1.set_xlim(lo, hi)
    a1.set_xlabel("time since trigger (s)"); a1.set_ylabel(f"counts per {bw * 1e3:g} ms bin")
    a1.set_title(f"{GRB}, optimal set {optDets}, {ENERGY_KEV[0]:g}-{ENERGY_KEV[1]:g} keV", fontsize=10)
    a1.legend(fontsize=8)
    for k, (setName, c, dx) in enumerate((("opt", "C0", 0.97), ("all", "C1", 1.03))):
        pts = [s for s in summaries if s["detectorSet"] == setName and s["median_ms"] and np.isfinite(s["median_ms"])]
        if not pts:
            continue
        x = np.array([s["binSize_s"] * 1e3 for s in pts]) * dx
        y = np.array([s["median_ms"] for s in pts])
        a2.errorbar(x, y, yerr=[[s["minus_err_ms"] for s in pts], [s["plus_err_ms"] for s in pts]], fmt="o", color=c,
                    capsize=3, label=f"{setName} ({len(pts[0]['detectors'])} detectors)")
    a2.axhspan(DICHIARA_MS[0] - DICHIARA_MS[1], DICHIARA_MS[0] + DICHIARA_MS[1], color="0.85",
               label="Dichiara et al. 2023: 17 ± 2 ms (8-1000 keV)")
    a2.set_xscale("log")
    a2.set_xlabel("bin width (ms)"); a2.set_ylabel("precursor MVT (ms): median and 16-84%")
    a2.set_title(f"MVT against bin width (real-data Monte Carlo, {N_REALISATIONS} realisations)", fontsize=10)
    a2.grid(True, which="both", ls=":", alpha=0.5)
    a2.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"P11_precursor_230307A.{ext}"), dpi=140)
    plt.close(fig)
    print(f"\nSaved plot {PLOT_DIR}/P11_precursor_230307A.png (.pdf too)")


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
    print(f"MVTTools code in use: {commit}")
    print(f"{GRB} precursor {list(INTERVAL)} s, {ENERGY_KEV[0]:g}-{ENERGY_KEV[1]:g} keV, background order {BKG_ORDER} "
          f"on {BKG_INTERVALS} s")
    data = {d: loadDetector(d) for d in ALL_NAI}
    print("background over the interval (c/s, 8-900 keV): " + "  ".join(f"{d} {data[d]['bkgBand']:.0f}" for d in ALL_NAI))
    print("background fit reduced chi2: " + "  ".join(f"{d} {data[d]['redChi2']:.2f}" for d in ALL_NAI))
    print("photons in the interval (8-900 keV): " + "  ".join(f"{d} {inInterval(data[d]['timesBand']).size}" for d in ALL_NAI))

    os.makedirs(OUT, exist_ok=True)
    selPath = os.path.join(OUT, "selection.json")
    print("\n===== STEP A. DETECTOR SELECTION =====")
    first = selectDetectors(data, FIRST_SELECTION_BIN_S)
    provisional = selectDetectors(data, DICHIARA_MS[0] / 1e3)
    if args.dets:
        optDets = sorted(args.dets, key=ALL_NAI.index)
        print(f"optimal set forced with --dets: {optDets}")
    elif args.post and os.path.exists(selPath):
        optDets = json.load(open(selPath))["used"]
        print(f"optimal set used for the saved results: {optDets}")
    else:
        optDets = provisional["optimal"]
        if optDets != first["optimal"]:
            print(f"set changed from 16 ms to 17 ms bins: using the 17 ms one (Bala et al.: repeat with bin width = MVT)")
    sets = {"opt": optDets, "all": list(ALL_NAI)}

    print(f"\n===== STEP B. MVT, REAL-DATA MONTE CARLO ({N_REALISATIONS} realisations, resampled errors) =====")
    summaries = []
    if args.post:
        for setName in sets:
            for bw in BIN_WIDTHS:
                p = os.path.join(OUT, f"{setName}_{bw * 1e3:g}ms", "summary.json")
                if os.path.exists(p):
                    summaries.append(json.load(open(p)))
                else:
                    print(f"WARNING: no results for {setName} {bw * 1e3:g} ms")
    else:
        here = os.getcwd()
        os.chdir(REPO)
        from loggerSetup import initialiseLogging
        initialiseLogging()
        import analysisTools.haarMethods as haarMethods
        os.chdir(here)
        for setName, dets in sets.items():
            for bw in BIN_WIDTHS:
                summaries.append(monteCarlo(setName, dets, bw, data, haarMethods, commit))

    print("\n===== STEP C. SNR_MVT AND CLASS =====")
    model = validation.load_model()
    if model is None:
        sys.exit(f"Could not load the validation curve from {validation.MODEL_FILE}")
    print(f"{'set':4s}{'BW ms':>7s}{'median (16-84%) ms':>24s}{'valid%':>7s}{'UL%':>5s}{'fail%':>6s}{'bkg c/s':>9s}"
          f"{'SNR phys':>9s}{'phases':>13s}{'Poisson':>13s}{'SNR Bala':>9s}{'class':>8s}{'band':>14s}")
    for i, s in enumerate(summaries):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 800, i]))
        r = snrFor(s, data, model, rng)
        s["snr"] = r
        med = f"{s['median_ms']:.2f} ({s['p16_ms']:.2f}-{s['p84_ms']:.2f})"
        if r is None:
            print(f"{s['detectorSet']:4s}{s['binSize_s'] * 1e3:>7g}{med:>24s}  no valid median")
            continue
        print(f"{s['detectorSet']:4s}{s['binSize_s'] * 1e3:>7g}{med:>24s}{s['validPercent']:>7.0f}{s['limitPercent']:>5.0f}"
              f"{s['failedPercent']:>6.0f}{r['bkgCps_band']:>9.0f}{r['physical']:>9.1f}"
              f"{r['phaseMin']:>6.1f}-{r['phaseMax']:<6.1f}{r['poissonMean']:>7.1f}±{r['poissonStd']:<5.1f}"
              f"{r['balaCompatible']:>9.1f}{r['cls']:>8s}{r['band'][0]:>7.1f}-{r['band'][2]:<6.1f}"
              + ("  (outside the curve's MVT range)" if r["outsideCurve"] else ""))
    print("SNR phys = all photons, 8-900 keV background (observed); phases = min-max over 10 bin phases; Poisson = mean ± std"
          f" of {args.npoisson} resamples; SNR Bala = merged detectors, all-channel background (used for the class);"
          " band = Bala et al.'s 95% band at that MVT (above it = robust)")

    print("\n===== CHECKS =====")
    verdict = {}
    for setName in sets:
        rows = {s["binSize_s"]: s for s in summaries if s["detectorSet"] == setName}
        if 1e-5 not in rows or 3e-6 not in rows or rows[3e-6]["snr"] is None:
            print(f"{setName}: results missing, not judged")
            continue
        a, b = rows[1e-5], rows[3e-6]
        change = b["median_ms"] / a["median_ms"] - 1
        plateau = abs(change) <= PLATEAU_TOL
        cls = b["snr"]["cls"]
        quote = (f"MVT84 = {b['p84_ms']:.2f} ms (upper limit, class {cls})" if cls in ("UL", "LUL")
                 else f"MVT = {b['median_ms']:.2f} (+{b['plus_err_ms']:.2f}/-{b['minus_err_ms']:.2f}) ms (class {cls})")
        err = b["plus_err_ms"] if b["median_ms"] < DICHIARA_MS[0] else b["minus_err_ms"]
        gap = (b["median_ms"] - DICHIARA_MS[0]) / np.hypot(err, DICHIARA_MS[1])
        print(f"\n{setName} ({len(b['detectors'])} detectors {b['detectors']})")
        print(f"  R8a plateau: 0.01 -> 0.003 ms median change {100 * change:+.1f}% -> {'yes' if plateau else 'NO'};"
              f" quoted: {quote}")
        print(f"  R8b Dichiara et al. 17 ± 2 ms: difference {b['median_ms'] - DICHIARA_MS[0]:+.2f} ms = {gap:+.1f} sigma -> "
              f"{'consistent' if abs(gap) <= 2 else 'NOT consistent'}")
        th = b5Thresholds()
        trust = None
        if th:
            near = [t for t in th if 0.5 <= t["R"] / b["median_ms"] <= 2.0]
            if near:
                need = max(t["snr"] for t in near)
                trust = b["snr"]["physical"] >= need
                print(f"  R8c trust (B5 precursor-like thresholds of pulses with R within x2 of our MVT: "
                      + ", ".join(f"{t['pulse']} R {t['R']:.1f} ms SNR {t['snr']:.1f}" for t in near)
                      + f"): physical SNR {b['snr']['physical']:.1f} vs {need:.1f} -> "
                      + ("ABOVE the trust threshold" if trust else "BELOW: read the MVT as an upper estimate"))
            else:
                print("  R8c trust: no B5 pulse with R within x2 of our MVT -> not judged")
            print(f"      background here {b['snr']['bkgCps_band']:.0f} c/s (8-900 keV) against 5000 c/s in B5")
        else:
            print(f"  R8c trust: {B5_SUMMARY} not found -> not judged")
        verdict[setName] = dict(plateauChange=change, plateau=plateau, quote=quote, cls=cls, dichiaraGapSigma=gap,
                                trustAboveB5=trust)

    print("\n  Detector set repeated at our measured MVT (optimal set, 0.003 ms):")
    opt3 = next((s for s in summaries if s["detectorSet"] == "opt" and s["binSize_s"] == 3e-6), None)
    repeat = None
    if opt3 and opt3["median_ms"] and np.isfinite(opt3["median_ms"]):
        repeat = selectDetectors(data, opt3["median_ms"] / 1e3)
        same = repeat["optimal"] == optDets
        print("  -> same set" if same else
              f"  -> SET CHANGED: rerun with  python run_precursor_mvt.py --dets {' '.join(repeat['optimal'])}"
              "  (after moving the opt_* result folders away)")

    print(f"\n  R8d background Poisson check (off-burst intervals, optimal set, 8-900 keV):")
    fz = fano(data, optDets)
    ok = True
    for tau, f in fz.items():
        judged = tau <= FANO_JUDGED_MAX_S
        good = abs(f["F"] - 1) <= FANO_TOL
        ok &= good or not judged
        print(f"    {tau * 1e3:>7g} ms: F = {f['F']:.4f} ± {f['err']:.4f}  ({f['meanCountsPerBin']:.3g} counts/bin, "
              f"{f['nBins']} bins){'' if judged else '  [reported only]'}{'' if good else '  <- outside 1 ± 0.05'}")
    print(f"  -> {'consistent with Poisson' if ok else 'NOT consistent with Poisson'} up to {FANO_JUDGED_MAX_S * 1e3:g} ms")

    with open(selPath, "w") as f:
        json.dump(dict(first16ms=first, provisional17ms=provisional, used=optDets,
                       repeatAtMeasuredMvt=repeat), f, indent=2, default=float)
    report = dict(grb=GRB, interval_s=list(INTERVAL), energyRange_keV=list(ENERGY_KEV), bkgIntervals_s=BKG_INTERVALS,
                  mvttoolsCommit=commit, sets=sets,
                  background={d: dict(bkgBand=data[d]["bkgBand"], bkgAll=data[d]["bkgAll"], redChi2=data[d]["redChi2"])
                              for d in ALL_NAI},
                  runs=summaries, verdict=verdict, fano={str(k): v for k, v in fz.items()}, fanoConsistent=ok)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved {OUT}/summary.json and selection.json")
    plot(data, summaries, optDets)
    print(f"Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
