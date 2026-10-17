"""
Rerun only the Monte Carlo step on the saved GRB 230307A light curve,
using the fixed estimateMvtDistribution, and compare with the baseline run.
"""
import os, sys, json, time
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
GRB = "GRB230307A"
CSV = f"/mnt/data/rgl36/FYP/processed/{GRB}/photonCounts.csv"
BASELINE = f"/mnt/data/rgl36/FYP/results/{GRB}_baseline/summary.json"
OUT_DIR = f"/mnt/data/rgl36/FYP/results/{GRB}_mcfix"
N_WORKERS = 16

sys.path.insert(0, REPO)
os.chdir(REPO)

from loggerSetup import initialiseLogging
initialiseLogging()
from loadConfig import config
import analysisTools.haarMethods as haarMethods
from fermiDataTools.fermiGRBObservation import EXPORT_BIN_SIZE_SECONDS


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    nReal = int(config.mvtAnalysisConfig.numberOfRealisations)

    data = np.genfromtxt(CSV, delimiter=",")
    counts, errors = data[:, 0], data[:, 1]

    t0 = time.time()
    r = haarMethods.estimateMvtDistribution(
        counts, errors, haarMethods.haarPowerMod,
        binSizeSeconds=EXPORT_BIN_SIZE_SECONDS, nRealisations=nReal, nWorkers=N_WORKERS)
    minutes = (time.time() - t0) / 60

    np.save(os.path.join(OUT_DIR, "mvt_values_seconds.npy"), r["mvtValues"])
    np.save(os.path.join(OUT_DIR, "mvt_uncertainties_seconds.npy"), r["mvtUncertainties"])

    ms = lambda x: float(x) * 1e3
    summary = {
        "grb": GRB,
        "binSize_s": EXPORT_BIN_SIZE_SECONDS,
        "nRealisations": r["nRealisations"],
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
        "meanAll_ms": ms(r["meanAll"]),
        "stdAll_ms": ms(r["stdAll"]),
        "minutes_montecarlo": minutes,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== FIXED MONTE CARLO =====")
    for k, v in summary.items():
        print(f"{k:20s} {v}")

    if os.path.exists(BASELINE):
        b = json.load(open(BASELINE))
        print("\n===== COMPARISON (ms) =====")
        print(f"{'':28s}{'baseline':>12s}{'fixed':>12s}")
        print(f"{'Derek mean (all runs)':28s}{b['derek_mean_ms']:12.3f}{summary['meanAll_ms']:12.3f}")
        print(f"{'Derek std (all runs)':28s}{b['derek_std_ms']:12.3f}{summary['stdAll_ms']:12.3f}")
        print(f"{'Median':28s}{b['all_median_ms']:12.3f}{summary['median_ms']:12.3f}")
        print(f"{'16th percentile':28s}{b['all_p16_ms']:12.3f}{summary['p16_ms']:12.3f}")
        print(f"{'84th percentile':28s}{b['all_p84_ms']:12.3f}{summary['p84_ms']:12.3f}")
        print("Bala et al. (2026), 0.1 ms, same detectors: median 4.4 ms, success 93%")
    print(f"\nSaved in {OUT_DIR}")
