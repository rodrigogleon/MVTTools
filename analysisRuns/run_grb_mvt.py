"""
MVT of a Fermi GBM burst with the full Bala et al. (2026) workflow, one row of their Table 3 per run:
  1. light curve (8-900 keV, chosen detectors, bin width, source interval)
  2. single MVT and 300 Monte Carlo realisations (errors sqrt(counts) per realisation, seed 2026)
  3. background (first-order polynomial on the off-burst intervals, 1.024 s bins)
  4. SNR_MVT = max(counts in bins of width MVT) / sqrt(background rate * MVT)
  5. reliability class from Bala et al.'s validation curve (UL / LUL / robust), using a Bala-compatible SNR_MVT:
     detectors combined with GbmTte.merge (default force_unique=True) and the background over all energy
     channels, as in Bala et al.'s code, because the validation curve was calibrated with that definition.
     The physical SNR_MVT (all photons, 8-900 keV background) is reported as well.
and, over all finished runs, the full Table 3 comparison and the workflow verdict at the plateau
(smallest bin width; if UL or LUL the quoted value is MVT84).

This replaces run_bala_rows.py, run_bala_rows_211211A.py and run_snr_classify.py with one script.
Steps 1-2 are copied unchanged from run_bala_rows.py, steps 3-5 unchanged from run_snr_classify.py.
Each burst's settings are in GRBS below. Results folders are the ones already used, so finished
Monte Carlo runs are reused.

Usage
  python run_grb_mvt.py --grb GRB211211A --run S        Monte Carlo + SNR_MVT + class for one row
  python run_grb_mvt.py --grb GRB211211A --post         SNR_MVT + class + full table for all finished rows (no Monte Carlo)
  python run_grb_mvt.py --grb GRB211211A --table        quick MVT-only table
"""
import os, sys, json, time, glob, argparse, subprocess
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
BALA_CODE = "/mnt/data/rgl36/FYP/external/GBM_MVT_paper/MVTfermi"
DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
RESULTS = "/mnt/data/rgl36/FYP/results"

ENERGY_KEV = (8.0, 900.0)
N_REALISATIONS = 300
SEED = 2026
BKG_BIN_S = 1.024
BKG_ORDER = 1
N_PHASES = 10
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]


def row(dets, bw, interval, single, median, plus, minus, snr, success, failed):
    return (dets, bw, interval, dict(single=single, median=median, plus=plus, minus=minus,
                                     snr=snr, success=success, failed=failed))


# Bala et al. (2026), Table 3 (values checked against the PDF)
GRBS = {
    "GRB230307A": dict(
        mcRoot=f"{RESULTS}/bala_rows_8-900keV_rs_haarfix",
        bkgIntervals=[(-68.8, -18.8), (69.4, 119.4)],
        detectorSets={"5det": ["n0", "n1", "n6", "n7", "na"],
                      "7det": ["n0", "n1", "n6", "n7", "n9", "na", "nb"],
                      "all": ALL_NAI},
        plateau={"7det": "G", "all": "H"},
        runs={
            "B": row("7det", 1e-5, (-1.64, 18.0), "1.3 ± 0.6", 1.9, 0.7, 0.4, 94.2, 44.3, 55.7),
            "C": row("7det", 1e-4, (-1.64, 18.0), "4.2 ± 0.6", 4.2, 0.4, 0.5, 135.4, 98.0, 2.0),
            "D": row("5det", 1e-4, (-1.6, 45.3), "4.3 ± 0.8", 4.4, 0.7, 0.5, 137.0, 93.0, 7.0),
            "Ds": row("5det", 1e-4, (-1.64, 18.0), "4.3 ± 0.8", 4.4, 0.7, 0.5, 137.0, 93.0, 7.0),
            "E": row("all", 1e-4, (-1.64, 18.0), "3.8 ± 0.6", 3.9, 0.5, 0.3, 116.2, 100., 0.0),
            "F": row("all", 1e-5, (-1.64, 18.0), "1.5 ± 0.5", 1.8, 0.4, 0.3, 82.1, 62.3, 37.7),
            "G": row("7det", 3e-6, (-1.64, 18.0), "1.3 ± 0.5", 1.7, 0.7, 0.4, 90.9, 36.3, 63.7),
            "H": row("all", 3e-6, (-1.64, 18.0), "1.4 ± 0.4", 1.8, 0.5, 0.3, 80.7, 57.7, 42.3),
        }),
    "GRB211211A": dict(
        mcRoot=f"{RESULTS}/bala_rows_211211A",
        bkgIntervals=[(-68.0, -18.0), (69.7, 119.7)],
        detectorSets={"1det": ["n2"], "3det": ["n1", "n2", "na"], "all": ALL_NAI},
        plateau={"3det": "S", "all": "V"},
        runs={
            "P": row("1det", 1e-4, (-0.9, 15.0), "7.7 ± 1.6", 8.8, 2.0, 1.7, 134.7, 88.3, 11.7),
            "Q": row("3det", 1e-4, (-0.9, 15.0), "4.2 ± 1.3", 4.7, 0.9, 0.7, 121.9, 58.7, 41.3),
            "R": row("3det", 1e-5, (-0.9, 15.0), "2.6 ± 1.4", 3.6, 1.6, 0.8, 113.5, 46.3, 53.7),
            "S": row("3det", 3e-6, (-0.9, 15.0), "2.5 ± 1.3", 3.9, 1.0, 1.0, 116.4, 42.7, 57.3),
            "T": row("all", 1e-4, (-0.9, 15.0), "4.3 ± 0.9", 4.4, 0.8, 0.6, 94.1, 96.7, 3.3),
            "U": row("all", 1e-5, (-0.9, 15.0), "2.5 ± 0.9", 3.4, 0.9, 0.9, 85.0, 58.3, 41.7),
            "V": row("all", 3e-6, (-0.9, 15.0), "2.5 ± 0.9", 3.1, 1.1, 0.7, 82.7, 54.7, 45.3),
        }),
}

parser = argparse.ArgumentParser()
parser.add_argument("--grb", choices=list(GRBS), required=True)
parser.add_argument("--run", help="one row: Monte Carlo + SNR_MVT + class")
parser.add_argument("--post", action="store_true", help="SNR_MVT + class + full table for all finished rows")
parser.add_argument("--table", action="store_true", help="quick MVT-only table")
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--npoisson", type=int, default=300, help="Poisson resamples for the mean SNR")
args = parser.parse_args()

CFG = GRBS[args.grb]
RUNS = CFG["runs"]
if args.run is not None and args.run not in RUNS:
    parser.error(f"--run must be one of {list(RUNS)} for {args.grb}")
if not (args.run or args.post or args.table):
    parser.error("give --run X, --post or --table")
SNR_DIR = os.path.join(RESULTS, "snr_classify", args.grb)
ms = lambda x: float(x) * 1e3


def gapSigma(s, b):
    """Difference in medians divided by the two facing error bars combined."""
    if s["median_ms"] < b["median"]:
        err = np.hypot(s["plus_err_ms"], b["minus"])
    else:
        err = np.hypot(s["minus_err_ms"], b["plus"])
    return (s["median_ms"] - b["median"]) / err


def loadSummary(run):
    path = os.path.join(CFG["mcRoot"], f"Run{run}", "summary.json")
    return json.load(open(path)) if os.path.exists(path) else None


def printQuickTable():
    print(f"\n{args.grb}, {ENERGY_KEV[0]:g}-{ENERGY_KEV[1]:g} keV, resampled errors, seed {SEED}")
    print(f"{'run':4s}{'dets':6s}{'BW ms':>7s}{'interval':>15s} | {'ours: median (+/-)':>22s}{'valid%':>8s}{'UL%':>6s}{'fail%':>7s}"
          f" | {'Bala: median (+/-)':>20s}{'succ%':>7s} | {'gap':>6s}")
    for run, (dets, bw, interval, b) in RUNS.items():
        s = loadSummary(run)
        left = f"{run:4s}{dets:6s}{bw * 1e3:>7g}{str(list(interval)):>15s} | "
        bala = f"{b['median']:>8.1f} (+{b['plus']:.1f}/-{b['minus']:.1f}){b['success']:>7.1f}"
        if s is None:
            print(left + f"{'not run yet':>43s} | " + bala + " |")
            continue
        ours = f"{s['median_ms']:>8.2f} (+{s['plus_err_ms']:.2f}/-{s['minus_err_ms']:.2f})"
        print(left + f"{ours:>22s}{s['validPercent']:>8.1f}{s['limitPercent']:>6.1f}{s['failedPercent']:>7.1f} | "
              + bala + f" | {gapSigma(s, b):>+5.1f}σ")
    print("gap = (ours - Bala) / combined facing error bars.  Bala 'Failed' = our UL + Failed.")


if args.table:
    printQuickTable()
    sys.exit(0)

sys.path.insert(0, REPO)
sys.path.insert(1, BALA_CODE)
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
import clasify_mvt_value as validation

validation.MODEL_FILE = os.path.join(BALA_CODE, "mvt_snr_fit_model.npz")
SHORT_NAMES = {
    "Below 95% CI (Upper Limit)": "UL",
    "Within 95% CI (Likely Upper Limit)": "LUL",
    "Above 95% CI (Robust Measurement)": "robust",
}


def tteFile(det):
    files = glob.glob(f"{DATA_ROOT}/{args.grb}/current/glg_tte_{det}_bn*_v0*.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    return files[0]


# ---------------------------------------------------------------- steps 1-2 (as run_bala_rows.py)
def binDetectors(detectors, binSize, tmin, tmax):
    """Sum the event counts of the chosen detectors between ENERGY_KEV, in bins of binSize."""
    binEdges = np.arange(tmin, tmax + binSize, binSize)
    counts = np.zeros(len(binEdges) - 1, dtype=float)
    perDetector = {}
    for det in detectors:
        tte = GbmTte.open(tteFile(det))
        times = np.array(tte.slice_energy(ENERGY_KEV).data.times)
        tte.close()
        detCounts, _ = np.histogram(times, bins=binEdges)
        counts += detCounts
        perDetector[det] = int(detCounts.sum())
    return counts, perDetector


def classifyRealisation(mvt, unc):
    if not np.isfinite(mvt) or mvt <= 0.0:
        return "failed"
    if not unc > 0.0:
        return "upper limit"
    return "valid"


def monteCarlo(run):
    """Single MVT and 300 realisations for one row; writes the same files as run_bala_rows.py."""
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from loggerSetup import initialiseLogging
    initialiseLogging()
    import analysisTools.haarMethods as haarMethods

    dets, bw, (tmin, tmax), b = RUNS[run]
    commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"],
                            capture_output=True, text=True).stdout.strip()
    print(f"MVTTools code in use: {commit}")
    detectors = CFG["detectorSets"][dets]
    outDir = os.path.join(CFG["mcRoot"], f"Run{run}")
    os.makedirs(outDir, exist_ok=True)

    t0 = time.time()
    counts, perDetector = binDetectors(detectors, bw, tmin, tmax)
    errors = np.sqrt(counts)
    np.save(os.path.join(outDir, "counts.npy"), counts)
    print(f"\n===== RUN {run}: {args.grb}, {dets}, bin {bw * 1e3:g} ms, [{tmin}, {tmax}] s, {ENERGY_KEV} keV =====")
    print(f"detectors        {detectors}")
    print(f"counts/detector  {perDetector}")
    print(f"number of bins   {len(counts)}")
    print(f"total counts     {int(counts.sum())}")
    print(f"mean counts/bin  {counts.mean():.3f}")
    print(f"empty bins       {100 * np.mean(counts == 0):.1f}%")
    print(f"binning took     {time.time() - t0:.1f} s")

    t0 = time.time()
    (_, _, singleMvt, singleUnc, _, _, _) = haarMethods.haarPowerMod(counts, errors, minimumBinSizeSeconds=bw)
    singleType = classifyRealisation(singleMvt, singleUnc)
    print(f"\nsingle MVT       {ms(singleMvt):.3f} ± {ms(singleUnc):.3f} ms  ({singleType}), took {time.time() - t0:.1f} s")

    t0 = time.time()
    r = haarMethods.estimateMvtDistribution(
        counts, errors, haarMethods.haarPowerMod,
        binSizeSeconds=bw, nRealisations=N_REALISATIONS,
        nWorkers=args.workers, seed=SEED, resampleErrors=True)
    minutes = (time.time() - t0) / 60

    values, uncs = r["mvtValues"], r["mvtUncertainties"]
    np.save(os.path.join(outDir, "mvt_values_seconds.npy"), values)
    np.save(os.path.join(outDir, "mvt_uncertainties_seconds.npy"), uncs)
    limits = values[r["limitMask"]]
    n = r["nRealisations"]
    summary = {
        "run": run, "mvttoolsCommit": commit, "grb": args.grb, "detectorSet": dets, "detectors": detectors,
        "binSize_s": bw, "timeRange_s": [tmin, tmax], "energyRange_keV": list(ENERGY_KEV),
        "resampleErrors": True, "seed": SEED, "totalCounts": int(counts.sum()),
        "singleMvt_ms": ms(singleMvt), "singleUnc_ms": ms(singleUnc), "singleType": singleType,
        "nRealisations": n, "nUnique": int(len(np.unique(values))),
        "nValid": r["nValid"], "nLimit": r["nLimit"], "nFailed": r["nFailed"],
        "validPercent": 100 * r["nValid"] / n, "limitPercent": 100 * r["nLimit"] / n,
        "failedPercent": 100 * r["nFailed"] / n,
        "median_ms": ms(r["median"]), "p16_ms": ms(r["p16"]), "p84_ms": ms(r["p84"]),
        "plus_err_ms": ms(r["p84"] - r["median"]), "minus_err_ms": ms(r["median"] - r["p16"]),
        "mean_valid_ms": ms(r["mean"]), "std_valid_ms": ms(r["std"]),
        "limits_median_ms": ms(np.median(limits)) if len(limits) else None,
        "limits_p16_p84_ms": [ms(x) for x in np.percentile(limits, [16, 84])] if len(limits) else None,
        "minutes_montecarlo": minutes, "bala": b,
    }
    with open(os.path.join(outDir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== RUN {run}: RESULT vs BALA et al. (2026), TABLE 3 =====")
    print(f"{'':24s}{'ours':>22s}{'Bala':>14s}")
    print(f"{'single MVT (ms)':24s}{f'{ms(singleMvt):.2f} ± {ms(singleUnc):.2f}':>22s}{b['single']:>14s}")
    print(f"{'median MVT (ms)':24s}{summary['median_ms']:>22.2f}{b['median']:>14.1f}")
    print(f"{'  + error (ms)':24s}{summary['plus_err_ms']:>22.2f}{b['plus']:>14.1f}")
    print(f"{'  - error (ms)':24s}{summary['minus_err_ms']:>22.2f}{b['minus']:>14.1f}")
    print(f"{'valid / success (%)':24s}{summary['validPercent']:>22.1f}{b['success']:>14.1f}")
    print(f"{'UL (%)':24s}{summary['limitPercent']:>22.1f}{'':>14s}")
    print(f"{'failed (%)':24s}{summary['failedPercent']:>22.1f}{'':>14s}")
    print(f"{'UL + failed (%)':24s}{summary['limitPercent'] + summary['failedPercent']:>22.1f}{b['failed']:>14.1f}")
    print(f"{'distinct values':24s}{summary['nUnique']:>22d}")
    print(f"{'Monte Carlo time (min)':24s}{minutes:>22.1f}")
    print(f"\ngap to Bala: {summary['median_ms'] - b['median']:+.2f} ms = {gapSigma(summary, b):+.1f} sigma")
    print(f"Saved in {outDir}")
    return summary


# ---------------------------------------------------------------- steps 3-5 (as run_snr_classify.py)
def loadDetector(det):
    """Event times (8-900 keV and all channels) and the fitted background model of one detector."""
    tte = GbmTte.open(tteFile(det))
    timesBand = np.sort(np.array(tte.slice_energy(ENERGY_KEV).data.times))
    timesAll = np.sort(np.array(tte.data.times))
    phaii = tte.to_phaii(bin_by_time, BKG_BIN_S, time_ref=0.0)
    fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=CFG["bkgIntervals"])
    fitter.fit(order=BKG_ORDER)
    model = fitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
    tte.close()
    stat, dof = np.asarray(fitter.statistic), np.asarray(fitter.dof)
    return dict(
        timesBand=timesBand, timesAll=timesAll,
        tstart=np.asarray(model.tstart), tstop=np.asarray(model.tstop),
        ratesBand=np.asarray(model.integrate_energy(*ENERGY_KEV).rates).ravel(),
        ratesAll=np.asarray(model.integrate_energy().rates).ravel(),
        redChi2=float(np.nansum(stat) / np.nansum(dof)) if np.nansum(dof) > 0 else np.nan,
    )


def meanRate(d, interval, key):
    """Average fitted background rate (counts/s) over the 1.024 s bins inside the interval."""
    inside = (d["tstart"] >= interval[0]) & (d["tstop"] <= interval[1])
    return float(np.mean(d[key][inside]))


def offBurstRate(times):
    n = sum(np.count_nonzero((times >= a) & (times < b)) for a, b in CFG["bkgIntervals"])
    return n / sum(b - a for a, b in CFG["bkgIntervals"])


def lightCurve(times, binS, tmin, tmax, phase=0.0):
    edges = np.arange(tmin + phase * binS, tmax + binS, binS)
    return np.histogram(times, bins=edges)[0]


def snrMvt(maxCounts, bkgCps, mvtS):
    return maxCounts / np.sqrt(bkgCps * mvtS)


def snrSet(times, bkgBand, bkgAll, mvtMs, interval, rng):
    mvtS = mvtMs / 1e3
    counts = lightCurve(times, mvtS, *interval)
    phases = [snrMvt(lightCurve(times, mvtS, *interval, phase=k / N_PHASES).max(), bkgBand, mvtS)
              for k in range(N_PHASES)]
    poissonMax = np.array([rng.poisson(counts).max() for _ in range(args.npoisson)])
    return dict(
        mvt_ms=mvtMs, maxCounts=int(counts.max()), bkgCountsPerBin=bkgBand * mvtS,
        observed=snrMvt(counts.max(), bkgBand, mvtS),
        phaseMin=float(min(phases)), phaseMax=float(max(phases)),
        poissonMean=float(np.mean(snrMvt(poissonMax, bkgBand, mvtS))),
        poissonStd=float(np.std(snrMvt(poissonMax, bkgBand, mvtS))),
        observedAllChannelBkg=snrMvt(counts.max(), bkgAll, mvtS),
    )


def classify(mvtMs, snr, model):
    if mvtMs is None or snr is None or not (mvtMs > 0 and snr > 0):
        return "n/a", None
    interpolators, mvtRangeLog = model
    label, bounds = validation.classify_mvt_value(mvtMs, snr, interpolators, mvtRangeLog)
    low, mid, high = (float(10 ** x) for x in bounds)
    return SHORT_NAMES.get(label, label), dict(lower=low, median=mid, upper=high)


def loadModel():
    loaded = validation.load_model()
    if loaded is None:
        sys.exit(f"Could not load the validation curve from {validation.MODEL_FILE}")
    return loaded


def snrForRun(run, s, data, model):
    """Background, SNR_MVT (4 versions, at our and Bala's median) and class for one finished row."""
    detSet, bw, interval, b = RUNS[run]
    dets = CFG["detectorSets"][detSet]
    times = np.sort(np.concatenate([data[d]["timesBand"] for d in dets]))
    times = times[(times >= interval[0]) & (times < interval[1])]
    bkgBand = sum(meanRate(data[d], interval, "ratesBand") for d in dets)
    bkgAll = sum(meanRate(data[d], interval, "ratesAll") for d in dets)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, list(RUNS).index(run)]))
    ours = snrSet(times, bkgBand, bkgAll, s["median_ms"], interval, rng)
    atBala = snrSet(times, bkgBand, bkgAll, b["median"], interval, rng)

    print(f"\nRun {run}: {detSet}, bin {bw * 1e3:g} ms, {list(interval)} s, "
          f"background {bkgBand:.0f} c/s (8-900 keV), {bkgAll:.0f} c/s (all channels)")
    for name, x in (("at our median", ours), ("at Bala median", atBala)):
        print(f"  {name:15s} {x['mvt_ms']:6.2f} ms: max {x['maxCounts']:5d} counts/bin, bkg {x['bkgCountsPerBin']:6.2f}/bin | "
              f"SNR obs {x['observed']:6.1f}  phases {x['phaseMin']:6.1f}-{x['phaseMax']:6.1f}  "
              f"Poisson {x['poissonMean']:6.1f} ± {x['poissonStd']:4.1f}  allE {x['observedAllChannelBkg']:6.1f}")
    print(f"  Bala Table 3 SNR {b['snr']:.1f}")

    # Bala-compatible SNR: detectors combined with GbmTte.merge (default force_unique=True, as in Bala et al.'s
    # code) and the background counted over all energy channels; used only for the class, because the
    # validation curve was calibrated with this definition
    merged = mergedTimes(dets, interval)
    balaDefOurs = balaDefSnr(merged, bkgAll, s["median_ms"], interval)
    balaDefAtBala = balaDefSnr(merged, bkgAll, b["median"], interval)
    print(f"  Bala-compatible SNR (merged detectors, all-channel background): {balaDefOurs:6.1f} at our median, "
          f"{balaDefAtBala:6.1f} at Bala median (Table 3 {b['snr']:.1f}, ratio {balaDefAtBala / b['snr']:.3f})")

    classOurs, boundsOurs = classify(s["median_ms"], ours["observed"], model)
    classOursPoisson, _ = classify(s["median_ms"], ours["poissonMean"], model)
    classOursBalaDef, boundsOursBalaDef = classify(s["median_ms"], balaDefOurs, model)
    classBala, boundsBala = classify(b["median"], b["snr"], model)
    return dict(detectorSet=detSet, binSize_s=bw, interval_s=list(interval),
                bkgCps_band=bkgBand, bkgCps_allChannels=bkgAll,
                snrAtOurMedian=ours, snrAtBalaMedian=atBala,
                classOurs=classOurs, classOursPoissonSnr=classOursPoisson,
                boundsOurs=boundsOurs, classBala=classBala, boundsBala=boundsBala,
                snrBalaDefAtOurMedian=balaDefOurs, snrBalaDefAtBalaMedian=balaDefAtBala,
                classOursBalaDef=classOursBalaDef, boundsOursBalaDef=boundsOursBalaDef)


def mergedTimes(dets, interval):
    """8-900 keV event times after GbmTte.merge of the detectors (each sliced to the interval first)."""
    ttes = []
    for d in dets:
        tte = GbmTte.open(tteFile(d))
        ttes.append(tte.slice_time(interval))
        tte.close()
    times = np.array(GbmTte.merge(ttes).slice_energy(ENERGY_KEV).data.times)
    return times[(times >= interval[0]) & (times < interval[1])]


def balaDefSnr(times, bkgAll, mvtMs, interval):
    mvtS = mvtMs / 1e3
    return float(snrMvt(lightCurve(times, mvtS, *interval).max(), bkgAll, mvtS))


def printClassLine(run, s, r):
    b = RUNS[run][3]
    needO = f"{r['boundsOurs']['upper']:.1f}" if r["boundsOurs"] else "n/a"
    needB = f"{r['boundsBala']['upper']:.1f}" if r["boundsBala"] else "n/a"
    print(f"{run:4s}{s['median_ms']:>13.2f}{r['snrAtOurMedian']['observed']:>7.1f}{needO:>13s}{r['classOurs']:>8s}"
          f"{r['classOursPoissonSnr']:>15s}{r['snrBalaDefAtOurMedian']:>9.1f}{r['classOursBalaDef']:>10s}"
          f" | {b['median']:>13.1f}{b['snr']:>7.1f}{needB:>13s}{r['classBala']:>8s}")


CLASS_HEADER = (f"{'':4s}{'':>13s}{'physical':>7s}{'':>13s}{'':>8s}{'':>15s}{'Bala-compatible':>19s}\n"
                f"{'run':4s}{'ours: MVT ms':>13s}{'SNR':>7s}{'needs SNR >':>13s}{'class':>8s}{'(Poisson SNR)':>15s}"
                f"{'SNR':>9s}{'CLASS':>10s} | {'Bala: MVT ms':>13s}{'SNR':>7s}{'needs SNR >':>13s}{'class':>8s}")


def post():
    """Steps 3-5 for every finished row, the full Table 3 comparison and the plateau verdict."""
    t0 = time.time()
    os.makedirs(SNR_DIR, exist_ok=True)
    report = {"grb": args.grb, "energyRange_keV": list(ENERGY_KEV), "bkgIntervals_s": CFG["bkgIntervals"],
              "bkgBin_s": BKG_BIN_S, "bkgOrder": BKG_ORDER, "seed": SEED, "nPoisson": args.npoisson}

    summaries = {}
    for run in RUNS:
        s = loadSummary(run)
        if s is None:
            print(f"WARNING: no Monte Carlo results for run {run}")
        else:
            summaries[run] = s
    commits = sorted({s.get("mvttoolsCommit", "?") for s in summaries.values()})
    print(f"MVT results read from {CFG['mcRoot']}")
    print(f"MVTTools commit(s) used for those results: {commits}")
    report["mvttoolsCommits"] = commits

    print(f"\n===== PART 1: background per detector (order-{BKG_ORDER} polynomial, {BKG_BIN_S} s bins, "
          f"off-burst {CFG['bkgIntervals']} s) =====")
    neededDets = sorted({d for dets, *_ in RUNS.values() for d in CFG["detectorSets"][dets]}, key=ALL_NAI.index)
    intervals = sorted({iv for _, _, iv, _ in RUNS.values()})
    data, bkgTable = {}, {}
    print(f"{'det':4s}{'red chi2':>9s} | {'off-burst raw':>13s}"
          + "".join(f"{'fit ' + str(list(iv)):>22s}" for iv in intervals)
          + f" | {'fit, all channels':>18s}   (counts/s, 8-900 keV unless stated)")
    for det in neededDets:
        d = loadDetector(det)
        data[det] = d
        rowOut = dict(redChi2=d["redChi2"], offBurstRaw=offBurstRate(d["timesBand"]))
        for iv in intervals:
            rowOut[f"fit_{iv[0]:g}_{iv[1]:g}"] = meanRate(d, iv, "ratesBand")
            rowOut[f"fitAll_{iv[0]:g}_{iv[1]:g}"] = meanRate(d, iv, "ratesAll")
        bkgTable[det] = rowOut
        print(f"{det:4s}{rowOut['redChi2']:>9.2f} | {rowOut['offBurstRaw']:>13.1f}"
              + "".join(f"{rowOut[f'fit_{iv[0]:g}_{iv[1]:g}']:>22.1f}" for iv in intervals)
              + f" | {rowOut[f'fitAll_{intervals[0][0]:g}_{intervals[0][1]:g}']:>18.1f}")
    report["background"] = bkgTable

    model = loadModel()
    lo, hi = (10 ** x for x in model[1])
    print(f"\nValidation curve loaded: MVT range {lo:.2f}-{hi:.0f} ms")

    print("\n===== PART 2: SNR_MVT per run =====")
    print("obs = observed light curve; phases = min-max over 10 bin phases; Poisson = mean ± std of "
          f"{args.npoisson} resamples; allE = observed with all-channel background")
    results = {run: snrForRun(run, s, data, model) for run, s in summaries.items()}

    print("\n===== PART 3: reliability class (Bala et al.'s validation curve) =====")
    print(CLASS_HEADER)
    for run, r in results.items():
        printClassLine(run, summaries[run], r)
    print("'needs SNR >' = upper edge of the 95% band at that MVT (above it = robust).")
    print("CLASS = class from the Bala-compatible SNR (same SNR definition as the validation curve); this is the one used.")

    def quoteFor(cls, s):
        return (f"MVT84 = {s['p84_ms']:.2f} ms (upper limit)" if cls in ("UL", "LUL")
                else f"MVT = {s['median_ms']:.2f} (+{s['plus_err_ms']:.2f}/-{s['minus_err_ms']:.2f}) ms")

    print("\n----- Workflow verdict (plateau = smallest bin width; class from the Bala-compatible SNR) -----")
    verdict = {}
    for detSet, run in CFG["plateau"].items():
        if run not in results:
            continue
        s, r, b = summaries[run], results[run], RUNS[run][3]
        quote = quoteFor(r["classOurs"], s)
        quoteBalaDef = quoteFor(r["classOursBalaDef"], s)
        balaQuote = (f"MVT84 = {b['median'] + b['plus']:.1f} ms" if r["classBala"] in ("UL", "LUL")
                     else f"MVT = {b['median']:.1f} ms")
        print(f"{detSet:5s} run {run}: ours {r['classOursBalaDef']:6s} -> {quoteBalaDef:38s} | Bala {r['classBala']:6s} -> {balaQuote}"
              f"   (physical SNR would give {r['classOurs']})")
        verdict[detSet] = dict(run=run, classOurs=r["classOurs"], quoteOurs=quote,
                               classBala=r["classBala"], quoteBala=balaQuote,
                               classOursBalaDef=r["classOursBalaDef"], quoteOursBalaDef=quoteBalaDef)
    report["verdict"] = verdict

    print("\n===== FULL TABLE 3 COMPARISON (MVT columns read from the saved runs, unchanged) =====")
    print(f"{'':16s} | {'OURS':^87s} | {'BALA ET AL. TABLE 3':^56s} |")
    print(f"{'run':4s}{'dets':6s}{'BW ms':>6s} | {'single MVT ms':>15s}{'median (+/-) ms':>22s}{'valid':>7s}{'UL':>6s}{'fail':>6s}"
          f"{'UL+fail':>8s}{'SNR':>7s}{'SNR-B':>8s}{'class':>8s} | {'single':>10s}{'median (+/-)':>17s}{'succ':>7s}{'failed':>7s}"
          f"{'SNR':>7s}{'class':>8s} | {'gap':>6s}")
    for run, r in results.items():
        s, (detSet, bw, interval, b) = summaries[run], RUNS[run]
        single = f"{s['singleMvt_ms']:.2f} ± {s['singleUnc_ms']:.2f}"
        med = f"{s['median_ms']:.2f} (+{s['plus_err_ms']:.2f}/-{s['minus_err_ms']:.2f})"
        bmed = f"{b['median']:.1f} (+{b['plus']:.1f}/-{b['minus']:.1f})"
        ulFail = s["limitPercent"] + s["failedPercent"]
        print(f"{run:4s}{detSet:6s}{bw * 1e3:>6g} | {single:>15s}{med:>22s}{s['validPercent']:>7.1f}{s['limitPercent']:>6.1f}"
              f"{s['failedPercent']:>6.1f}{ulFail:>8.1f}{r['snrAtOurMedian']['observed']:>7.1f}"
              f"{r['snrBalaDefAtOurMedian']:>8.1f}{r['classOursBalaDef']:>8s} | "
              f"{b['single']:>10s}{bmed:>17s}{b['success']:>7.1f}{b['failed']:>7.1f}{b['snr']:>7.1f}{r['classBala']:>8s} | "
              f"{gapSigma(s, b):>+5.1f}σ")
        r["table"] = dict(ours=dict(single_ms=[s["singleMvt_ms"], s["singleUnc_ms"]], median_ms=s["median_ms"],
                                    plus_ms=s["plus_err_ms"], minus_ms=s["minus_err_ms"], p84_ms=s["p84_ms"],
                                    validPct=s["validPercent"], limitPct=s["limitPercent"],
                                    failedPct=s["failedPercent"], snr=r["snrAtOurMedian"]["observed"]),
                          bala=b, gapSigma=gapSigma(s, b))
    print("valid / UL / fail = % of realisations.  Bala 'succ' = our valid; Bala 'failed' = our UL + fail.")
    print("SNR = physical SNR_MVT (all photons, 8-900 keV background); SNR-B = Bala-compatible SNR_MVT;"
          " class from SNR-B.")
    print("gap = (our median - Bala median) / combined facing error bars.")

    report["runs"] = results
    with open(os.path.join(SNR_DIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved in {SNR_DIR}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    if args.run:
        s = monteCarlo(args.run)
        print(f"\n===== RUN {args.run}: SNR_MVT and reliability class =====")
        dets = CFG["detectorSets"][RUNS[args.run][0]]
        data = {d: loadDetector(d) for d in dets}
        r = snrForRun(args.run, s, data, loadModel())
        print("\n" + CLASS_HEADER)
        printClassLine(args.run, s, r)
        s["snrClassify"] = r
        with open(os.path.join(CFG["mcRoot"], f"Run{args.run}", "summary.json"), "w") as f:
            json.dump(s, f, indent=2, default=float)
        print("(saved in the run's summary.json; use --post for the full table and the plateau verdict)")
    if args.post:
        post()
