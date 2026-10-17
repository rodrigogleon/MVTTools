"""
Baseline run of Derek's unchanged MVT code on one Fermi burst.
Saves all Monte Carlo MVT values so they can be inspected afterwards.
"""
import os, sys, json, time, functools
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
GRB = "GRB230307A"
TIME_RANGE = (-1.64, 18.0)      # Bala et al. (2026) Table 3
N_WORKERS = 16                  # Chandra is shared: do not use all 128 cores
OUT_DIR = f"/mnt/data/rgl36/FYP/results/{GRB}_baseline"

sys.path.insert(0, REPO)
os.chdir(REPO)

from loggerSetup import initialiseLogging
initialiseLogging()
import logging
logger = logging.getLogger("run_baseline")

from loadConfig import config
import analysisTools.haarMethods as haarMethods
import fermiDataTools.fermiGRBObservation as fermiObs
from fermiDataTools.fermiGBMCatalogueGRB import getFermiGRBData
from fermiDataTools.fermiGBMPipe import processFermiGBMData

# limit the number of cores used by every Monte Carlo call (no change to the repo code)
limitedMC = functools.partial(haarMethods.estimateMvtUncertainty, nWorkers=N_WORKERS)
fermiObs.estimateMvtUncertainty = limitedMC


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    nReal = int(config.mvtAnalysisConfig.numberOfRealisations)
    logger.info(f"Baseline run: {GRB}, time range {TIME_RANGE}, {N_WORKERS} workers, {nReal} realisations")

    t0 = time.time()
    grb = getFermiGRBData(GRB)
    csvPath = processFermiGBMData(grb, timeRange=TIME_RANGE)
    t1 = time.time()
    logger.info(f"Light curve saved to {csvPath} ({(t1 - t0) / 60:.1f} min)")

    data = np.genfromtxt(csvPath, delimiter=",")
    counts, errors = data[:, 0], data[:, 1]
    binSize = fermiObs.EXPORT_BIN_SIZE_SECONDS

    mean, std, values = limitedMC(counts, errors, haarMethods.haarPowerMod,
                                  binSizeSeconds=binSize, nRealisations=nReal)
    t2 = time.time()
    np.save(os.path.join(OUT_DIR, "mvt_values_seconds.npy"), values)

    good = values[values > 0]
    def pct(a, q):
        return float(np.percentile(a, q)) * 1e3 if a.size else None
    summary = {
        "grb": GRB,
        "timeRange_s": TIME_RANGE,
        "binSize_s": binSize,
        "nRealisations": nReal,
        "nWorkers": N_WORKERS,
        "lightCurvePath": str(csvPath),
        "totalPhotons": float(counts.sum()),
        "derek_mean_ms": float(mean) * 1e3,
        "derek_std_ms": float(std) * 1e3,
        "nZero": int((values <= 0).sum()),
        "fractionZero": float((values <= 0).mean()),
        "all_median_ms": pct(values, 50),
        "all_p16_ms": pct(values, 16),
        "all_p84_ms": pct(values, 84),
        "nonzero_median_ms": pct(good, 50),
        "nonzero_p16_ms": pct(good, 16),
        "nonzero_p84_ms": pct(good, 84),
        "minutes_lightcurve": (t1 - t0) / 60,
        "minutes_montecarlo": (t2 - t1) / 60,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== BASELINE RESULT =====")
    for k, v in summary.items():
        print(f"{k:22s} {v}")
    print(f"\nSaved in {OUT_DIR}")
