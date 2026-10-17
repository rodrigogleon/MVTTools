"""
GRB 230307A: reproduce the rows of Bala et al. (2026), Table 3.

Bins the saved Fermi GBM TTE events for a chosen detector set, bin width and
time interval, measures a single MVT on the observed light curve, runs the
Monte Carlo (valid / upper limit / failed kept separate), and prints the
result next to the matching row of Table 3.

Examples
  Run C  7 det, 0.1 ms, [-1.64, 18.0] s:
      python run_bala_compare.py --bin 1e-4 --dets 7det
  Run D  5 det, 0.1 ms, full interval [-1.6, 45.3] s:
      python run_bala_compare.py --bin 1e-4 --dets 5det --tmin -1.6 --tmax 45.3
  Run E  all 12 NaI, 0.1 ms, [-1.64, 18.0] s:
      python run_bala_compare.py --bin 1e-4 --dets all
"""
import os, sys, json, time, glob, argparse
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
GRB = "GRB230307A"
RESULTS = "/mnt/data/rgl36/FYP/results"

DETECTOR_SETS = {
    "5det": ["n0", "n1", "n6", "n7", "na"],
    "7det": ["n0", "n1", "n6", "n7", "n9", "na", "nb"],
    "all": ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"],
}

# Bala et al. (2026), Table 3, GRB 230307A.
# Interval: the table places "Source: [-1.64, 18.0] s" under the 7-det and All rows,
# and the GRB line "Source: [-1.6, 45.3] s" under the 5-det row (our reading).
BALA_TABLE3 = {
    ("7det", 1e-4):  dict(single="4.2 ± 0.6", median=4.2, plus=0.4, minus=0.5, snr=135.4, success=98.0, failed=2.0,  interval="[-1.64, 18.0]"),
    ("7det", 1e-5):  dict(single="1.3 ± 0.6", median=1.9, plus=0.7, minus=0.4, snr=94.2,  success=44.3, failed=55.7, interval="[-1.64, 18.0]"),
    ("7det", 3e-6):  dict(single="1.3 ± 0.5", median=1.7, plus=0.7, minus=0.4, snr=90.9,  success=36.3, failed=63.7, interval="[-1.64, 18.0]"),
    ("all", 1e-4):   dict(single="3.8 ± 0.6", median=3.9, plus=0.5, minus=0.3, snr=116.2, success=100., failed=0.0,  interval="[-1.64, 18.0]"),
    ("all", 1e-5):   dict(single="1.5 ± 0.5", median=1.8, plus=0.4, minus=0.3, snr=82.1,  success=62.3, failed=37.7, interval="[-1.64, 18.0]"),
    ("all", 3e-6):   dict(single="1.4 ± 0.4", median=1.8, plus=0.5, minus=0.3, snr=80.7,  success=57.7, failed=42.3, interval="[-1.64, 18.0]"),
    ("5det", 1e-4):  dict(single="4.3 ± 0.8", median=4.4, plus=0.7, minus=0.5, snr=137.0, success=93.0, failed=7.0,  interval="[-1.6, 45.3]"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--bin", type=float, required=True, help="bin size in seconds: 1e-4, 1e-5 or 3e-6")
parser.add_argument("--dets", required=True, choices=list(DETECTOR_SETS), help="5det, 7det or all")
parser.add_argument("--tmin", type=float, default=-1.64)
parser.add_argument("--tmax", type=float, default=18.0)
parser.add_argument("--nreal", type=int, default=300)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()

sys.path.insert(0, REPO)
os.chdir(REPO)

from loggerSetup import initialiseLogging
initialiseLogging()
from loadConfig import config
import analysisTools.haarMethods as haarMethods
from gdt.missions.fermi.gbm.tte import GbmTte


def binDetectors(detectors, binSize, tmin, tmax):
    """Sum the event counts of the chosen detectors (same energy cut and bin edges as the pipeline)."""
    dataPath = config.generalSettings.directories.dataPath
    eLow, eHigh = (float(x) for x in config.preProcessingConfig.fermiGMBConfig.download.energyRange.split("-"))
    binEdges = np.arange(tmin, tmax + binSize, binSize)
    counts = np.zeros(len(binEdges) - 1, dtype=float)
    perDetector = {}
    for det in detectors:
        files = glob.glob(f"{dataPath}/fermiGBM/{GRB}/current/glg_tte_{det}_bn*_v00.fit")
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
        tte = GbmTte.open(files[0])
        times = np.array(tte.slice_energy((eLow, eHigh)).data.times)
        tte.close()
        detCounts, _ = np.histogram(times, bins=binEdges)
        counts += detCounts
        perDetector[det] = int(detCounts.sum())
    return counts, perDetector, (eLow, eHigh)


def classify(mvt, unc):
    if not np.isfinite(mvt) or mvt <= 0.0:
        return "failed"
    if not unc > 0.0:
        return "upper limit"
    return "valid"


if __name__ == "__main__":
    detectors = DETECTOR_SETS[args.dets]
    tag = f"{args.bin * 1e3:g}ms_{args.dets}_t{args.tmin:g}to{args.tmax:g}"
    outDir = f"{RESULTS}/{GRB}_bala_{tag}"
    os.makedirs(outDir, exist_ok=True)
    ms = lambda x: float(x) * 1e3

    # 1. bin the photons
    t0 = time.time()
    counts, perDetector, energyRange = binDetectors(detectors, args.bin, args.tmin, args.tmax)
    errors = np.sqrt(counts)
    np.save(os.path.join(outDir, "counts.npy"), counts)

    print(f"\n===== BINNING: {GRB}, {args.dets}, bin {args.bin * 1e3:g} ms, [{args.tmin}, {args.tmax}] s, {energyRange} keV =====")
    print(f"detectors        {detectors}")
    print(f"counts/detector  {perDetector}")
    print(f"number of bins   {len(counts)}")
    print(f"total counts     {int(counts.sum())}")
    print(f"mean counts/bin  {counts.mean():.3f}")
    print(f"empty bins       {100 * np.mean(counts == 0):.1f}%")
    print(f"binning took     {time.time() - t0:.1f} s")

    # 2. single MVT on the observed light curve (no Monte Carlo)
    t0 = time.time()
    (_, _, singleMvt, singleUnc, _, _, _) = haarMethods.haarPowerMod(counts, errors, minimumBinSizeSeconds=args.bin)
    singleSeconds = time.time() - t0
    singleType = classify(singleMvt, singleUnc)
    print(f"\n===== SINGLE MVT (observed light curve) =====")
    print(f"single MVT       {ms(singleMvt):.3f} ± {ms(singleUnc):.3f} ms  ({singleType})")
    print(f"one calculation  {singleSeconds:.1f} s")

    # 3. Monte Carlo
    t0 = time.time()
    r = haarMethods.estimateMvtDistribution(
        counts, errors, haarMethods.haarPowerMod,
        binSizeSeconds=args.bin, nRealisations=args.nreal,
        nWorkers=args.workers, seed=args.seed)
    minutes = (time.time() - t0) / 60

    values, uncs = r["mvtValues"], r["mvtUncertainties"]
    np.save(os.path.join(outDir, "mvt_values_seconds.npy"), values)
    np.save(os.path.join(outDir, "mvt_uncertainties_seconds.npy"), uncs)
    limits = values[r["limitMask"]]

    summary = {
        "grb": GRB,
        "detectorSet": args.dets,
        "detectors": detectors,
        "binSize_s": args.bin,
        "timeRange_s": [args.tmin, args.tmax],
        "energyRange_keV": list(energyRange),
        "totalCounts": int(counts.sum()),
        "seed": args.seed,
        "singleMvt_ms": ms(singleMvt),
        "singleUnc_ms": ms(singleUnc),
        "singleType": singleType,
        "nRealisations": r["nRealisations"],
        "nUnique": int(len(np.unique(values))),
        "nValid": r["nValid"],
        "nLimit": r["nLimit"],
        "nFailed": r["nFailed"],
        "validPercent": 100 * r["nValid"] / r["nRealisations"],
        "limitPercent": 100 * r["nLimit"] / r["nRealisations"],
        "failedPercent": 100 * r["nFailed"] / r["nRealisations"],
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
    }
    with open(os.path.join(outDir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== MONTE CARLO: {args.dets}, bin {args.bin * 1e3:g} ms, [{args.tmin}, {args.tmax}] s =====")
    for k, v in summary.items():
        print(f"{k:20s} {v}")

    # 4. side by side with Bala et al. (2026), Table 3
    b = next((row for (dets, bw), row in BALA_TABLE3.items()
              if dets == args.dets and np.isclose(bw, args.bin)), None)
    print("\n===== COMPARISON WITH BALA et al. (2026), TABLE 3 =====")
    if b is None:
        print("No matching row in Table 3 for this detector set and bin width.")
    else:
        print(f"{'':26s}{'ours':>26s}{'Bala':>26s}")
        print(f"{'time interval (s)':26s}{f'[{args.tmin}, {args.tmax}]':>26s}{b['interval']:>26s}")
        print(f"{'single MVT (ms)':26s}{f'{ms(singleMvt):.2f} ± {ms(singleUnc):.2f}':>26s}{b['single']:>26s}")
        print(f"{'median MVT (ms)':26s}{summary['median_ms']:>26.2f}{b['median']:>26.1f}")
        print(f"{'  + error (ms)':26s}{summary['plus_err_ms']:>26.2f}{b['plus']:>26.1f}")
        print(f"{'  - error (ms)':26s}{summary['minus_err_ms']:>26.2f}{b['minus']:>26.1f}")
        print(f"{'valid / success (%)':26s}{summary['validPercent']:>26.1f}{b['success']:>26.1f}")
        print(f"{'limit + failed (%)':26s}{summary['limitPercent'] + summary['failedPercent']:>26.1f}{b['failed']:>26.1f}")
        print(f"{'SNR_MVT':26s}{'not computed yet':>26s}{b['snr']:>26.1f}")
        gapSigma = abs(summary["median_ms"] - b["median"]) / np.hypot(
            summary["plus_err_ms"] if summary["median_ms"] < b["median"] else summary["minus_err_ms"],
            b["minus"] if summary["median_ms"] < b["median"] else b["plus"])
        print(f"\ndifference in median: {summary['median_ms'] - b['median']:+.2f} ms  ({gapSigma:.1f} sigma, both error bars)")
    print(f"\nSaved in {outDir}")
