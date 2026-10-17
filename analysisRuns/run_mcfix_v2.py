"""
Rerun only the Monte Carlo step on the saved GRB 230307A light curve,
now with an independent random seed per realisation, and compare with
the baseline and the first (v1) mc-fix run.
"""
import os, sys, json, time
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
GRB = "GRB230307A"
CSV = f"/mnt/data/rgl36/FYP/processed/{GRB}/photonCounts.csv"
RESULTS = "/mnt/data/rgl36/FYP/results"
OUT_DIR = f"{RESULTS}/{GRB}_mcfix_v2"
N_WORKERS = 16
SEED = 2026

sys.path.insert(0, REPO)
os.chdir(REPO)

from loggerSetup import initialiseLogging
initialiseLogging()
from loadConfig import config
import analysisTools.haarMethods as haarMethods
from fermiDataTools.fermiGRBObservation import EXPORT_BIN_SIZE_SECONDS


def nUnique(path):
    return int(len(np.unique(np.load(path)))) if os.path.exists(path) else None


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    nReal = int(config.mvtAnalysisConfig.numberOfRealisations)

    data = np.genfromtxt(CSV, delimiter=",")
    counts, errors = data[:, 0], data[:, 1]

    t0 = time.time()
    r = haarMethods.estimateMvtDistribution(
        counts, errors, haarMethods.haarPowerMod,
        binSizeSeconds=EXPORT_BIN_SIZE_SECONDS, nRealisations=nReal,
        nWorkers=N_WORKERS, seed=SEED)
    minutes = (time.time() - t0) / 60

    np.save(os.path.join(OUT_DIR, "mvt_values_seconds.npy"), r["mvtValues"])
    np.save(os.path.join(OUT_DIR, "mvt_uncertainties_seconds.npy"), r["mvtUncertainties"])

    ms = lambda x: float(x) * 1e3
    summary = {
        "grb": GRB,
        "binSize_s": EXPORT_BIN_SIZE_SECONDS,
        "seed": SEED,
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
        "meanAll_ms": ms(r["meanAll"]),
        "stdAll_ms": ms(r["stdAll"]),
        "minutes_montecarlo": minutes,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== MC-FIX v2 (one seed per realisation) =====")
    for k, v in summary.items():
        print(f"{k:20s} {v}")

    b = json.load(open(f"{RESULTS}/{GRB}_baseline/summary.json"))
    v1 = json.load(open(f"{RESULTS}/{GRB}_mcfix/summary.json"))
    u_b = nUnique(f"{RESULTS}/{GRB}_baseline/mvt_values_seconds.npy")
    u_v1 = nUnique(f"{RESULTS}/{GRB}_mcfix/mvt_values_seconds.npy")

    print("\n===== COMPARISON (ms) =====")
    print(f"{'':24s}{'baseline':>10s}{'v1':>10s}{'v2':>10s}")
    print(f"{'different values /300':24s}{u_b:>10d}{u_v1:>10d}{summary['nUnique']:>10d}")
    print(f"{'mean of all runs':24s}{b['derek_mean_ms']:10.3f}{v1['meanAll_ms']:10.3f}{summary['meanAll_ms']:10.3f}")
    print(f"{'std of all runs':24s}{b['derek_std_ms']:10.3f}{v1['stdAll_ms']:10.3f}{summary['stdAll_ms']:10.3f}")
    print(f"{'median':24s}{b['all_median_ms']:10.3f}{v1['median_ms']:10.3f}{summary['median_ms']:10.3f}")
    print(f"{'16th percentile':24s}{b['all_p16_ms']:10.3f}{v1['p16_ms']:10.3f}{summary['p16_ms']:10.3f}")
    print(f"{'84th percentile':24s}{b['all_p84_ms']:10.3f}{v1['p84_ms']:10.3f}{summary['p84_ms']:10.3f}")
    print("Bala et al. (2026), 0.1 ms, same detectors: 4.4 (+0.7/-0.5) ms, success 93%")
    print(f"\nSaved in {OUT_DIR}")
