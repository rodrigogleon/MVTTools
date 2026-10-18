"""
GRB 230307A: background, SNR_MVT and reliability class for every row of Bala et al. (2026), Table 3.

This is post-processing only. It reads the MVT results already saved by run_bala_rows.py
(haar-fix runs) and does NOT rerun the Monte Carlo.

Part 1  Background rate per NaI detector, with Bala et al.'s method:
          TTE -> 1.024 s bins (time_ref 0) -> first-order polynomial fitted to the off-burst
          intervals -> interpolated over the burst -> averaged over the run interval.
        Given in 8-900 keV (the light-curve band) and over all energy channels (in Bala et al.'s
        simulation code the background event list is not cut in energy).
        Sanity check: raw 8-900 keV count rate in the off-burst intervals.

Part 2  SNR_MVT for each run, as in Bala et al.'s code:
          SNR_MVT = max(counts in bins of width MVT) / sqrt(background rate * MVT)
        computed at our median MVT and at Bala et al.'s median MVT, on
          - the observed light curve (bins starting at the start of the interval)
          - 10 bin phases (bins shifted by 0, 0.1, ..., 0.9 MVT): min and max
          - 300 Poisson resamples of the light curve (seed 2026): mean
          - the same as the first, but with the all-channel background

Part 3  Reliability class with Bala et al.'s validation curve (clasify_mvt_value.py and
        mvt_snr_fit_model.npz from their repository):
          below the 95% band   -> upper limit (UL)
          inside the band      -> likely upper limit (LUL)
          above the band       -> robust
        For our values (our median, our SNR) and for Bala et al.'s values (their median, their SNR).
        Workflow verdict: the class of the plateau value (smallest bin width, runs G and H);
        if it is UL or LUL, the quoted value is MVT84 (the 84th percentile).

At the end: the full Table 3 comparison, every column, ours next to Bala et al.

Usage
  python run_snr_classify.py
  python run_snr_classify.py --npoisson 1000
"""
import os, sys, json, time, glob, argparse, subprocess
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
BALA_CODE = "/mnt/data/rgl36/FYP/external/GBM_MVT_paper/MVTfermi"
GRB = "GRB230307A"
DATA_DIR = f"/mnt/data/rgl36/FYP/data/fermiGBM/{GRB}/current"
RUNS_ROOT = "/mnt/data/rgl36/FYP/results/bala_rows_8-900keV_rs_haarfix"
OUT_DIR = "/mnt/data/rgl36/FYP/results/snr_classify"

ENERGY_KEV = (8.0, 900.0)
BKG_INTERVALS = [(-68.8, -18.8), (69.4, 119.4)]   # off-burst intervals (Bala et al.)
BKG_BIN_S = 1.024
BKG_ORDER = 1
SEED = 2026
N_PHASES = 10

DETECTOR_SETS = {
    "5det": ["n0", "n1", "n6", "n7", "na"],
    "7det": ["n0", "n1", "n6", "n7", "n9", "na", "nb"],
    "all": ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"],
}
SHORT = (-1.64, 18.0)
FULL = (-1.6, 45.3)

# run: detector set, bin width (s), interval (s), Bala et al. (2026) Table 3 row
RUNS = {
    "B": ("7det", 1e-5, SHORT, dict(single="1.3 ± 0.6", median=1.9, plus=0.7, minus=0.4, snr=94.2,  success=44.3, failed=55.7)),
    "C": ("7det", 1e-4, SHORT, dict(single="4.2 ± 0.6", median=4.2, plus=0.4, minus=0.5, snr=135.4, success=98.0, failed=2.0)),
    "D": ("5det", 1e-4, FULL,  dict(single="4.3 ± 0.8", median=4.4, plus=0.7, minus=0.5, snr=137.0, success=93.0, failed=7.0)),
    "Ds": ("5det", 1e-4, SHORT, dict(single="4.3 ± 0.8", median=4.4, plus=0.7, minus=0.5, snr=137.0, success=93.0, failed=7.0)),
    "E": ("all",  1e-4, SHORT, dict(single="3.8 ± 0.6", median=3.9, plus=0.5, minus=0.3, snr=116.2, success=100., failed=0.0)),
    "F": ("all",  1e-5, SHORT, dict(single="1.5 ± 0.5", median=1.8, plus=0.4, minus=0.3, snr=82.1,  success=62.3, failed=37.7)),
    "G": ("7det", 3e-6, SHORT, dict(single="1.3 ± 0.5", median=1.7, plus=0.7, minus=0.4, snr=90.9,  success=36.3, failed=63.7)),
    "H": ("all",  3e-6, SHORT, dict(single="1.4 ± 0.4", median=1.8, plus=0.5, minus=0.3, snr=80.7,  success=57.7, failed=42.3)),
}
PLATEAU_RUNS = {"7det": "G", "all": "H"}   # smallest bin width for each detector set

parser = argparse.ArgumentParser()
parser.add_argument("--npoisson", type=int, default=300, help="Poisson resamples for the mean SNR")
args = parser.parse_args()

sys.path.insert(0, BALA_CODE)
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


# ---------------------------------------------------------------- Part 1
def loadDetector(det):
    """Event times (8-900 keV and all channels) and the fitted background model of one detector."""
    files = glob.glob(f"{DATA_DIR}/glg_tte_{det}_bn*_v00.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    tte = GbmTte.open(files[0])
    timesBand = np.sort(np.array(tte.slice_energy(ENERGY_KEV).data.times))
    timesAll = np.sort(np.array(tte.data.times))

    phaii = tte.to_phaii(bin_by_time, BKG_BIN_S, time_ref=0.0)
    fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=BKG_INTERVALS)
    fitter.fit(order=BKG_ORDER)
    model = fitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
    tte.close()

    stat, dof = np.asarray(fitter.statistic), np.asarray(fitter.dof)
    return dict(
        timesBand=timesBand,
        timesAll=timesAll,
        tstart=np.asarray(model.tstart),
        tstop=np.asarray(model.tstop),
        ratesBand=np.asarray(model.integrate_energy(*ENERGY_KEV).rates).ravel(),
        ratesAll=np.asarray(model.integrate_energy().rates).ravel(),
        redChi2=float(np.nansum(stat) / np.nansum(dof)) if np.nansum(dof) > 0 else np.nan,
    )


def meanRate(d, interval, key):
    """Average fitted background rate (counts/s) over the 1.024 s bins inside the interval."""
    inside = (d["tstart"] >= interval[0]) & (d["tstop"] <= interval[1])
    return float(np.mean(d[key][inside]))


def offBurstRate(times):
    n = sum(np.count_nonzero((times >= a) & (times < b)) for a, b in BKG_INTERVALS)
    return n / sum(b - a for a, b in BKG_INTERVALS)


# ---------------------------------------------------------------- Part 2
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
        mvt_ms=mvtMs,
        maxCounts=int(counts.max()),
        bkgCountsPerBin=bkgBand * mvtS,
        observed=snrMvt(counts.max(), bkgBand, mvtS),
        phaseMin=float(min(phases)),
        phaseMax=float(max(phases)),
        poissonMean=float(np.mean(snrMvt(poissonMax, bkgBand, mvtS))),
        poissonStd=float(np.std(snrMvt(poissonMax, bkgBand, mvtS))),
        observedAllChannelBkg=snrMvt(counts.max(), bkgAll, mvtS),
    )


# ---------------------------------------------------------------- Part 3
def classify(mvtMs, snr, model):
    if mvtMs is None or snr is None or not (mvtMs > 0 and snr > 0):
        return "n/a", None
    interpolators, mvtRangeLog = model
    label, bounds = validation.classify_mvt_value(mvtMs, snr, interpolators, mvtRangeLog)
    low, mid, high = (float(10 ** x) for x in bounds)
    return SHORT_NAMES.get(label, label), dict(lower=low, median=mid, upper=high)


def gapSigma(s, b):
    if s["median_ms"] < b["median"]:
        err = np.hypot(s["plus_err_ms"], b["minus"])
    else:
        err = np.hypot(s["minus_err_ms"], b["plus"])
    return (s["median_ms"] - b["median"]) / err


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"energyRange_keV": list(ENERGY_KEV), "bkgIntervals_s": BKG_INTERVALS,
              "bkgBin_s": BKG_BIN_S, "bkgOrder": BKG_ORDER, "seed": SEED, "nPoisson": args.npoisson}

    summaries = {}
    for run in RUNS:
        path = os.path.join(RUNS_ROOT, f"Run{run}", "summary.json")
        if os.path.exists(path):
            summaries[run] = json.load(open(path))
        else:
            print(f"WARNING: no results for run {run} ({path})")
    commits = sorted({s.get("mvttoolsCommit", "?") for s in summaries.values()})
    print(f"MVT results read from {RUNS_ROOT}")
    print(f"MVTTools commit(s) used for those results: {commits}")
    report["mvttoolsCommits"] = commits

    # ---------------- Part 1 ----------------
    print(f"\n===== PART 1: background per detector (order-{BKG_ORDER} polynomial, {BKG_BIN_S} s bins, "
          f"off-burst {BKG_INTERVALS} s) =====")
    neededDets = sorted({d for dets, *_ in RUNS.values() for d in DETECTOR_SETS[dets]},
                        key=DETECTOR_SETS["all"].index)
    data, bkgTable = {}, {}
    print(f"{'det':4s}{'red chi2':>9s} | {'off-burst raw':>13s}{'fit, short':>12s}{'fit, full':>11s}"
          f" | {'fit, all channels, short':>25s}   (counts/s, 8-900 keV unless stated)")
    for det in neededDets:
        d = loadDetector(det)
        data[det] = d
        row = dict(redChi2=d["redChi2"], offBurstRaw=offBurstRate(d["timesBand"]),
                   short=meanRate(d, SHORT, "ratesBand"), full=meanRate(d, FULL, "ratesBand"),
                   shortAll=meanRate(d, SHORT, "ratesAll"), fullAll=meanRate(d, FULL, "ratesAll"))
        bkgTable[det] = row
        print(f"{det:4s}{row['redChi2']:>9.2f} | {row['offBurstRaw']:>13.1f}{row['short']:>12.1f}{row['full']:>11.1f}"
              f" | {row['shortAll']:>25.1f}")
    report["background"] = bkgTable

    # ---------------- Part 2 and 3 ----------------
    loaded = validation.load_model()
    if loaded is None:
        sys.exit(f"Could not load the validation curve from {validation.MODEL_FILE}")
    lo, hi = (10 ** x for x in loaded[1])
    print(f"\nValidation curve loaded: MVT range {lo:.2f}-{hi:.0f} ms")

    print("\n===== PART 2: SNR_MVT per run =====")
    print("obs = observed light curve; phases = min-max over 10 bin phases; Poisson = mean ± std of "
          f"{args.npoisson} resamples; allE = observed with all-channel background")
    results = {}
    for run, (detSet, bw, interval, b) in RUNS.items():
        if run not in summaries:
            continue
        s = summaries[run]
        dets = DETECTOR_SETS[detSet]
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

        classOurs, boundsOurs = classify(s["median_ms"], ours["observed"], loaded)
        classOursPoisson, _ = classify(s["median_ms"], ours["poissonMean"], loaded)
        classBala, boundsBala = classify(b["median"], b["snr"], loaded)
        results[run] = dict(detectorSet=detSet, binSize_s=bw, interval_s=list(interval),
                            bkgCps_band=bkgBand, bkgCps_allChannels=bkgAll,
                            snrAtOurMedian=ours, snrAtBalaMedian=atBala,
                            classOurs=classOurs, classOursPoissonSnr=classOursPoisson,
                            boundsOurs=boundsOurs, classBala=classBala, boundsBala=boundsBala)

    print("\n===== PART 3: reliability class (Bala et al.'s validation curve) =====")
    print(f"{'run':4s}{'ours: MVT ms':>13s}{'SNR':>7s}{'needs SNR >':>13s}{'class':>8s}{'(Poisson SNR)':>15s}"
          f" | {'Bala: MVT ms':>13s}{'SNR':>7s}{'needs SNR >':>13s}{'class':>8s}")
    for run, r in results.items():
        s, b = summaries[run], RUNS[run][3]
        needO = f"{r['boundsOurs']['upper']:.1f}" if r["boundsOurs"] else "n/a"
        needB = f"{r['boundsBala']['upper']:.1f}" if r["boundsBala"] else "n/a"
        print(f"{run:4s}{s['median_ms']:>13.2f}{r['snrAtOurMedian']['observed']:>7.1f}{needO:>13s}{r['classOurs']:>8s}"
              f"{r['classOursPoissonSnr']:>15s} | {b['median']:>13.1f}{b['snr']:>7.1f}{needB:>13s}{r['classBala']:>8s}")
    print("'needs SNR >' = upper edge of the 95% band at that MVT (above it = robust).")

    print("\n----- Workflow verdict (plateau = smallest bin width) -----")
    verdict = {}
    for detSet, run in PLATEAU_RUNS.items():
        if run not in results:
            continue
        s, r, b = summaries[run], results[run], RUNS[run][3]
        quote = (f"MVT84 = {s['p84_ms']:.2f} ms (upper limit)" if r["classOurs"] in ("UL", "LUL")
                 else f"MVT = {s['median_ms']:.2f} (+{s['plus_err_ms']:.2f}/-{s['minus_err_ms']:.2f}) ms")
        balaQuote = (f"MVT84 = {b['median'] + b['plus']:.1f} ms" if r["classBala"] in ("UL", "LUL")
                     else f"MVT = {b['median']:.1f} ms")
        print(f"{detSet:5s} run {run}: ours {r['classOurs']:6s} -> {quote:38s} | Bala {r['classBala']:6s} -> {balaQuote}")
        verdict[detSet] = dict(run=run, classOurs=r["classOurs"], quoteOurs=quote,
                               classBala=r["classBala"], quoteBala=balaQuote)
    report["verdict"] = verdict

    # ---------------- Full table ----------------
    print("\n===== FULL TABLE 3 COMPARISON (MVT columns read from the saved runs, unchanged) =====")
    hdr = (f"{'run':4s}{'dets':6s}{'BW ms':>6s} | {'single MVT ms':>15s}{'median (+/-) ms':>22s}{'valid':>7s}{'UL':>6s}{'fail':>6s}"
           f"{'UL+fail':>8s}{'SNR':>7s}{'class':>8s} | {'single':>10s}{'median (+/-)':>17s}{'succ':>7s}{'failed':>7s}"
           f"{'SNR':>7s}{'class':>8s} | {'gap':>6s}")
    print(f"{'':16s} | {'OURS':^79s} | {'BALA ET AL. TABLE 3':^56s} |")
    print(hdr)
    for run, r in results.items():
        s, (detSet, bw, interval, b) = summaries[run], RUNS[run]
        single = f"{s['singleMvt_ms']:.2f} ± {s['singleUnc_ms']:.2f}"
        med = f"{s['median_ms']:.2f} (+{s['plus_err_ms']:.2f}/-{s['minus_err_ms']:.2f})"
        bmed = f"{b['median']:.1f} (+{b['plus']:.1f}/-{b['minus']:.1f})"
        ulFail = s["limitPercent"] + s["failedPercent"]
        print(f"{run:4s}{detSet:6s}{bw * 1e3:>6g} | {single:>15s}{med:>22s}{s['validPercent']:>7.1f}{s['limitPercent']:>6.1f}"
              f"{s['failedPercent']:>6.1f}{ulFail:>8.1f}{r['snrAtOurMedian']['observed']:>7.1f}{r['classOurs']:>8s} | "
              f"{b['single']:>10s}{bmed:>17s}{b['success']:>7.1f}{b['failed']:>7.1f}{b['snr']:>7.1f}{r['classBala']:>8s} | "
              f"{gapSigma(s, b):>+5.1f}σ")
        r["table"] = dict(ours=dict(single_ms=[s["singleMvt_ms"], s["singleUnc_ms"]], median_ms=s["median_ms"],
                                    plus_ms=s["plus_err_ms"], minus_ms=s["minus_err_ms"], p84_ms=s["p84_ms"],
                                    validPct=s["validPercent"], limitPct=s["limitPercent"],
                                    failedPct=s["failedPercent"], snr=r["snrAtOurMedian"]["observed"]),
                          bala=b, gapSigma=gapSigma(s, b))
    print("valid / UL / fail = % of realisations.  Bala 'succ' = our valid; Bala 'failed' = our UL + fail.")
    print("gap = (our median - Bala median) / combined facing error bars.")

    report["runs"] = results
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved in {OUT_DIR}  ({(time.time() - t0) / 60:.1f} min)")
