"""
Bin-width check (Bala et al. 2026, Table 3): re-bin the saved Fermi GBM TTE
events of GRB 230307A at a finer bin size and rerun the Monte Carlo MVT.

Examples
  check binning only (no Monte Carlo), must reproduce photonCounts.csv:
      python run_binwidth.py --bin 1e-4 --nreal 0
  0.01 ms, same 5 detectors as the 0.1 ms run:
      python run_binwidth.py --bin 1e-5
  0.01 ms, the 7 detectors listed by Bala et al. for this bin width:
      python run_binwidth.py --bin 1e-5 --dets n0,n1,n6,n7,n9,na,nb
"""
import os, sys, json, time, glob, argparse
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
GRB = "GRB230307A"
TIME_RANGE = (-1.64, 18.0)
RESULTS = "/mnt/data/rgl36/FYP/results"
CSV_01MS = f"/mnt/data/rgl36/FYP/processed/{GRB}/photonCounts.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--bin", type=float, required=True, help="bin size in seconds, e.g. 1e-5")
parser.add_argument("--dets", default="n0,n1,n6,n7,na", help="comma-separated detector list")
parser.add_argument("--nreal", type=int, default=300, help="Monte Carlo realisations (0 = binning check only)")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()

sys.path.insert(0, REPO)
os.chdir(REPO)

from loggerSetup import initialiseLogging
initialiseLogging()
from loadConfig import config
import analysisTools.haarMethods as haarMethods
from gdt.missions.fermi.gbm.tte import GbmTte


def binDetectors(detectors, binSize):
    """Sum the event counts of the chosen detectors, using the same energy cut and bin edges as the pipeline."""
    dataPath = config.generalSettings.directories.dataPath
    eLow, eHigh = (float(x) for x in config.preProcessingConfig.fermiGMBConfig.download.energyRange.split("-"))
    binEdges = np.arange(TIME_RANGE[0], TIME_RANGE[1] + binSize, binSize)
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


if __name__ == "__main__":
    detectors = [d.strip() for d in args.dets.split(",")]
    tag = f"{args.bin * 1e3:g}ms_{len(detectors)}det"

    t0 = time.time()
    counts, perDetector, energyRange = binDetectors(detectors, args.bin)
    errors = np.sqrt(counts)

    print(f"\n===== BINNING: {GRB}, bin {args.bin * 1e3:g} ms, {TIME_RANGE} s, {energyRange} keV =====")
    print(f"detectors        {detectors}")
    print(f"counts/detector  {perDetector}")
    print(f"number of bins   {len(counts)}")
    print(f"total counts     {int(counts.sum())}")
    print(f"max counts/bin   {int(counts.max())}")
    print(f"empty bins       {100 * np.mean(counts == 0):.1f}%")
    print(f"binning took     {time.time() - t0:.1f} s")

    if args.nreal == 0:
        if np.isclose(args.bin, 1e-4) and os.path.exists(CSV_01MS):
            saved = np.genfromtxt(CSV_01MS, delimiter=",")[:, 0]
            same = len(saved) == len(counts) and np.array_equal(saved, counts)
            print(f"\nmatches saved photonCounts.csv: {same}  (saved: {len(saved)} bins, {int(saved.sum())} counts)")
        sys.exit(0)

    outDir = f"{RESULTS}/{GRB}_bin_{tag}"
    os.makedirs(outDir, exist_ok=True)
    np.save(os.path.join(outDir, "counts.npy"), counts)

    t0 = time.time()
    r = haarMethods.estimateMvtDistribution(
        counts, errors, haarMethods.haarPowerMod,
        binSizeSeconds=args.bin, nRealisations=args.nreal,
        nWorkers=args.workers, seed=args.seed)
    minutes = (time.time() - t0) / 60

    np.save(os.path.join(outDir, "mvt_values_seconds.npy"), r["mvtValues"])
    np.save(os.path.join(outDir, "mvt_uncertainties_seconds.npy"), r["mvtUncertainties"])

    ms = lambda x: float(x) * 1e3
    summary = {
        "grb": GRB,
        "binSize_s": args.bin,
        "timeRange_s": list(TIME_RANGE),
        "energyRange_keV": list(energyRange),
        "detectors": detectors,
        "totalCounts": int(counts.sum()),
        "seed": args.seed,
        "nRealisations": r["nRealisations"],
        "nUnique": int(len(np.unique(r["mvtValues"]))),
        "nValid": r["nValid"],
        "nLimit": r["nLimit"],
        "nFailed": r["nFailed"],
        "successRate": r["successRate"],
        "median_ms": ms(r["median"]),
        "p16_ms": ms(r["p16"]),
        "p84_ms": ms(r["p84"]),
        "minus_err_ms": ms(r["median"] - r["p16"]),
        "plus_err_ms": ms(r["p84"] - r["median"]),
        "mean_valid_ms": ms(r["mean"]),
        "std_valid_ms": ms(r["std"]),
        "minutes_montecarlo": minutes,
    }
    with open(os.path.join(outDir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== MONTE CARLO: bin {args.bin * 1e3:g} ms, {len(detectors)} detectors =====")
    for k, v in summary.items():
        print(f"{k:20s} {v}")
    print("\nBala et al. (2026), Table 3: 0.1 ms 4.4 ms | 0.01 ms 1.9 ms (7 dets) | 0.003 ms 1.7 ms (UL)")
    print(f"Saved in {outDir}")
