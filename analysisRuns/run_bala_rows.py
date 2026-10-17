"""
GRB 230307A: reproduce each row of Bala et al. (2026), Table 3, one run per row.

Fixed settings for every run (to match Bala et al. where already possible):
  energy range      8-900 keV (as in Bala et al.'s public code)
  errors            recomputed as sqrt(counts) for every realisation
  realisations      300, seed 2026
Kept as in our pipeline for now (to change later, one at a time):
  detectors         fixed list per row (no selection algorithm)
  background        not used (no off-burst intervals)
  SNR_MVT           not computed

Usage
  python run_bala_rows.py --run C          one run (B, C, D, E, F, G or H)
  python run_bala_rows.py --table          table of all finished runs vs Bala
"""
import os, sys, json, time, glob, argparse, subprocess
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
GRB = "GRB230307A"
OUT_ROOT = "/mnt/data/rgl36/FYP/results/bala_rows_8-900keV_rs_haarfix"
ENERGY_KEV = (8.0, 900.0)
N_REALISATIONS = 300
SEED = 2026

DETECTOR_SETS = {
    "5det": ["n0", "n1", "n6", "n7", "na"],
    "7det": ["n0", "n1", "n6", "n7", "n9", "na", "nb"],
    "all": ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"],
}
SHORT = (-1.64, 18.0)
FULL = (-1.6, 45.3)

# run: detector set, bin width (s), time interval (s), Bala et al. (2026) Table 3 row
RUNS = {
    "B": ("7det", 1e-5, SHORT, dict(single="1.3 ± 0.6", median=1.9, plus=0.7, minus=0.4, snr=94.2,  success=44.3, failed=55.7)),
    "C": ("7det", 1e-4, SHORT, dict(single="4.2 ± 0.6", median=4.2, plus=0.4, minus=0.5, snr=135.4, success=98.0, failed=2.0)),
    "D": ("5det", 1e-4, FULL,  dict(single="4.3 ± 0.8", median=4.4, plus=0.7, minus=0.5, snr=137.0, success=93.0, failed=7.0)),
    # extra: same detectors as D on the short interval, compared with the same Bala row (interval test)
    "Ds": ("5det", 1e-4, SHORT, dict(single="4.3 ± 0.8", median=4.4, plus=0.7, minus=0.5, snr=137.0, success=93.0, failed=7.0)),
    "E": ("all",  1e-4, SHORT, dict(single="3.8 ± 0.6", median=3.9, plus=0.5, minus=0.3, snr=116.2, success=100., failed=0.0)),
    "F": ("all",  1e-5, SHORT, dict(single="1.5 ± 0.5", median=1.8, plus=0.4, minus=0.3, snr=82.1,  success=62.3, failed=37.7)),
    "G": ("7det", 3e-6, SHORT, dict(single="1.3 ± 0.5", median=1.7, plus=0.7, minus=0.4, snr=90.9,  success=36.3, failed=63.7)),
    "H": ("all",  3e-6, SHORT, dict(single="1.4 ± 0.4", median=1.8, plus=0.5, minus=0.3, snr=80.7,  success=57.7, failed=42.3)),
}

parser = argparse.ArgumentParser()
parser.add_argument("--run", choices=list(RUNS), help="which row of Table 3 to reproduce")
parser.add_argument("--table", action="store_true", help="print all finished runs next to Bala et al.")
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()


def gapSigma(ours, b):
    """Difference in medians divided by the two facing error bars combined."""
    if ours["median_ms"] < b["median"]:
        err = np.hypot(ours["plus_err_ms"], b["minus"])
    else:
        err = np.hypot(ours["minus_err_ms"], b["plus"])
    return (ours["median_ms"] - b["median"]) / err


def printTable():
    print(f"\nGRB 230307A, {ENERGY_KEV[0]:g}-{ENERGY_KEV[1]:g} keV, resampled errors, seed {SEED}")
    print(f"{'run':4s}{'dets':6s}{'BW ms':>7s}{'interval':>15s} | {'ours: median (+/-)':>22s}{'valid%':>8s}{'UL%':>6s}{'fail%':>7s}"
          f" | {'Bala: median (+/-)':>20s}{'succ%':>7s} | {'gap':>6s}")
    for run, (dets, bw, interval, b) in RUNS.items():
        path = os.path.join(OUT_ROOT, f"Run{run}", "summary.json")
        left = f"{run:4s}{dets:6s}{bw * 1e3:>7g}{str(list(interval)):>15s} | "
        bala = f"{b['median']:>8.1f} (+{b['plus']:.1f}/-{b['minus']:.1f}){b['success']:>7.1f}"
        if not os.path.exists(path):
            print(left + f"{'not run yet':>43s} | " + bala + " |")
            continue
        s = json.load(open(path))
        ours = f"{s['median_ms']:>8.2f} (+{s['plus_err_ms']:.2f}/-{s['minus_err_ms']:.2f})"
        print(left + f"{ours:>22s}{s['validPercent']:>8.1f}{s['limitPercent']:>6.1f}{s['failedPercent']:>7.1f} | "
              + bala + f" | {gapSigma(s, b):>+5.1f}σ")
    print("gap = (ours - Bala) / combined facing error bars.  Bala 'Failed' = our UL + Failed.")


if args.table:
    printTable()
    sys.exit(0)
if args.run is None:
    parser.error("give --run X or --table")

sys.path.insert(0, REPO)
os.chdir(REPO)

from loggerSetup import initialiseLogging
initialiseLogging()
from loadConfig import config
import analysisTools.haarMethods as haarMethods
from gdt.missions.fermi.gbm.tte import GbmTte


def binDetectors(detectors, binSize, tmin, tmax):
    """Sum the event counts of the chosen detectors between ENERGY_KEV, in bins of binSize."""
    dataPath = config.generalSettings.directories.dataPath
    binEdges = np.arange(tmin, tmax + binSize, binSize)
    counts = np.zeros(len(binEdges) - 1, dtype=float)
    perDetector = {}
    for det in detectors:
        files = glob.glob(f"{dataPath}/fermiGBM/{GRB}/current/glg_tte_{det}_bn*_v00.fit")
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
        tte = GbmTte.open(files[0])
        times = np.array(tte.slice_energy(ENERGY_KEV).data.times)
        tte.close()
        detCounts, _ = np.histogram(times, bins=binEdges)
        counts += detCounts
        perDetector[det] = int(detCounts.sum())
    return counts, perDetector


def classify(mvt, unc):
    if not np.isfinite(mvt) or mvt <= 0.0:
        return "failed"
    if not unc > 0.0:
        return "upper limit"
    return "valid"


if __name__ == "__main__":
    dets, bw, (tmin, tmax), b = RUNS[args.run]
    commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"],
                            capture_output=True, text=True).stdout.strip()
    print(f"MVTTools code in use: {commit}")
    detectors = DETECTOR_SETS[dets]
    outDir = os.path.join(OUT_ROOT, f"Run{args.run}")
    os.makedirs(outDir, exist_ok=True)
    ms = lambda x: float(x) * 1e3

    # 1. light curve
    t0 = time.time()
    counts, perDetector = binDetectors(detectors, bw, tmin, tmax)
    errors = np.sqrt(counts)
    np.save(os.path.join(outDir, "counts.npy"), counts)
    print(f"\n===== RUN {args.run}: {GRB}, {dets}, bin {bw * 1e3:g} ms, [{tmin}, {tmax}] s, {ENERGY_KEV} keV =====")
    print(f"detectors        {detectors}")
    print(f"counts/detector  {perDetector}")
    print(f"number of bins   {len(counts)}")
    print(f"total counts     {int(counts.sum())}")
    print(f"mean counts/bin  {counts.mean():.3f}")
    print(f"empty bins       {100 * np.mean(counts == 0):.1f}%")
    print(f"binning took     {time.time() - t0:.1f} s")

    # 2. single MVT on the observed light curve
    t0 = time.time()
    (_, _, singleMvt, singleUnc, _, _, _) = haarMethods.haarPowerMod(counts, errors, minimumBinSizeSeconds=bw)
    singleType = classify(singleMvt, singleUnc)
    print(f"\nsingle MVT       {ms(singleMvt):.3f} ± {ms(singleUnc):.3f} ms  ({singleType}), took {time.time() - t0:.1f} s")

    # 3. Monte Carlo, errors recomputed for every realisation
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
        "run": args.run,
        "mvttoolsCommit": commit,
        "grb": GRB,
        "detectorSet": dets,
        "detectors": detectors,
        "binSize_s": bw,
        "timeRange_s": [tmin, tmax],
        "energyRange_keV": list(ENERGY_KEV),
        "resampleErrors": True,
        "seed": SEED,
        "totalCounts": int(counts.sum()),
        "singleMvt_ms": ms(singleMvt),
        "singleUnc_ms": ms(singleUnc),
        "singleType": singleType,
        "nRealisations": n,
        "nUnique": int(len(np.unique(values))),
        "nValid": r["nValid"],
        "nLimit": r["nLimit"],
        "nFailed": r["nFailed"],
        "validPercent": 100 * r["nValid"] / n,
        "limitPercent": 100 * r["nLimit"] / n,
        "failedPercent": 100 * r["nFailed"] / n,
        "median_ms": ms(r["median"]),
        "p16_ms": ms(r["p16"]),
        "p84_ms": ms(r["p84"]),
        "plus_err_ms": ms(r["p84"] - r["median"]),
        "minus_err_ms": ms(r["median"] - r["p16"]),
        "mean_valid_ms": ms(r["mean"]),
        "std_valid_ms": ms(r["std"]),
        "limits_median_ms": ms(np.median(limits)) if len(limits) else None,
        "limits_p16_p84_ms": [ms(x) for x in np.percentile(limits, [16, 84])] if len(limits) else None,
        "minutes_montecarlo": minutes,
        "bala": b,
    }
    with open(os.path.join(outDir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== RUN {args.run}: RESULT vs BALA et al. (2026), TABLE 3 =====")
    print(f"{'':24s}{'ours':>22s}{'Bala':>14s}")
    print(f"{'single MVT (ms)':24s}{f'{ms(singleMvt):.2f} ± {ms(singleUnc):.2f}':>22s}{b['single']:>14s}")
    print(f"{'median MVT (ms)':24s}{summary['median_ms']:>22.2f}{b['median']:>14.1f}")
    print(f"{'  + error (ms)':24s}{summary['plus_err_ms']:>22.2f}{b['plus']:>14.1f}")
    print(f"{'  - error (ms)':24s}{summary['minus_err_ms']:>22.2f}{b['minus']:>14.1f}")
    print(f"{'valid / success (%)':24s}{summary['validPercent']:>22.1f}{b['success']:>14.1f}")
    print(f"{'UL (%)':24s}{summary['limitPercent']:>22.1f}{'':>14s}")
    print(f"{'failed (%)':24s}{summary['failedPercent']:>22.1f}{'':>14s}")
    print(f"{'UL + failed (%)':24s}{summary['limitPercent'] + summary['failedPercent']:>22.1f}{b['failed']:>14.1f}")
    print(f"{'SNR_MVT':24s}{'not computed':>22s}{b['snr']:>14.1f}")
    print(f"{'distinct values':24s}{summary['nUnique']:>22d}")
    print(f"{'Monte Carlo time (min)':24s}{minutes:>22.1f}")
    print(f"\ngap to Bala: {summary['median_ms'] - b['median']:+.2f} ms = {gapSigma(summary, b):+.1f} sigma")
    print(f"Saved in {outDir}")
