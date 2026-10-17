"""
Two checks on GRB 230307A (8-900 keV):

Part 1  Table 3 interval question.
        Single MVT for the 5-detector set at 0.1 ms on the SHORT [-1.64, 18.0] s and FULL
        [-1.6, 45.3] s intervals, with both codes, to compare with Bala et al.'s 5-detector
        row (4.3 +/- 0.8 ms).

Part 2  Our Haar code (as currently checked out in MVTTools) vs Bala et al.'s original Haar
        code on IDENTICAL inputs. 7 detectors, 0.01 ms, [-1.64, 18.0] s (the Run B setup):
          a) the observed light curve
          b) the first N Monte Carlo realisations of Run B (same seeds, errors = sqrt(counts)),
             also compared with the values saved by Run B (computed with the code at that time)

Both codes are called with the same settings Bala et al. use (afactor=-1, weight=True,
snr=3, nrepl=2, bin_fac=4, tau_bg_max=0.01, max_dt=100).

Usage
  python run_code_compare.py                  (N = 32 realisations, 16 workers)
  python run_code_compare.py --nreal 64
"""
import os, sys, json, time, glob, argparse, subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = "/mnt/data/rgl36/FYP/MVTTools"
BALA_CODE = "/mnt/data/rgl36/FYP/external/GBM_MVT_paper/MVTfermi"
GRB = "GRB230307A"
OUT_DIR = "/mnt/data/rgl36/FYP/results/code_compare"
RUN_B_VALUES = "/mnt/data/rgl36/FYP/results/bala_rows_8-900keV_rs/RunB/mvt_values_seconds.npy"
ENERGY_KEV = (8.0, 900.0)
SHORT = (-1.64, 18.0)
FULL = (-1.6, 45.3)
RUN_B_UNCS = "/mnt/data/rgl36/FYP/results/bala_rows_8-900keV_rs/RunB/mvt_uncertainties_seconds.npy"
SEED = 2026
N_RUN_B = 300

DETECTOR_SETS = {
    "5det": ["n0", "n1", "n6", "n7", "na"],
    "7det": ["n0", "n1", "n6", "n7", "n9", "na", "nb"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--nreal", type=int, default=32, help="number of Run B realisations to compare")
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

sys.path.insert(0, REPO)
sys.path.insert(1, BALA_CODE)
os.chdir(REPO)

from loggerSetup import initialiseLogging
initialiseLogging()
from loadConfig import config
from analysisTools.haarMethods import haarPowerMod as ourHaar
from haar_power_mod import haar_power_mod as balaHaar
from gdt.missions.fermi.gbm.tte import GbmTte


def binDetectors(detectors, binSize, tmin, tmax):
    dataPath = config.generalSettings.directories.dataPath
    binEdges = np.arange(tmin, tmax + binSize, binSize)
    counts = np.zeros(len(binEdges) - 1, dtype=float)
    for det in detectors:
        files = glob.glob(f"{dataPath}/fermiGBM/{GRB}/current/glg_tte_{det}_bn*_v00.fit")
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
        tte = GbmTte.open(files[0])
        times = np.array(tte.slice_energy(ENERGY_KEV).data.times)
        tte.close()
        detCounts, _ = np.histogram(times, bins=binEdges)
        counts += detCounts
    return counts


def classify(mvt, unc):
    if not np.isfinite(mvt) or mvt <= 0.0:
        return "failed"
    if not unc > 0.0:
        return "limit"
    return "valid"


def runBoth(counts, binSize):
    """MVT and uncertainty (seconds) from both codes on the same light curve."""
    # our code gets exactly what the Monte Carlo gives it (as in Run B)
    d = ourHaar(counts, np.sqrt(counts), minimumBinSizeSeconds=binSize)
    # Bala's code gets what their wrapper gives it: float counts, errors = sqrt(|counts|)
    c = np.asarray(counts, dtype=float)
    b = balaHaar(c, np.sqrt(np.abs(c)), min_dt=binSize, doplot=False, afactor=-1.0,
                 verbose=False, weight=True, file="compare")
    return float(d[2]), float(d[3]), float(b[2]), float(b[3])


def oneRealisation(task):
    i, seed, counts, binSize = task
    sampled = np.random.default_rng(seed).poisson(lam=counts)   # identical to Run B
    return (i,) + runBoth(sampled, binSize)


def fmt(mvt, unc):
    return f"{mvt * 1e3:8.3f} ± {unc * 1e3:6.3f} ms ({classify(mvt, unc)})"


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"],
                            capture_output=True, text=True).stdout.strip()
    print(f"MVTTools code in use: {commit}")
    report = {"mvttoolsCommit": commit}

    # ---------------- Part 1 ----------------
    print("\n===== PART 1: 5 det, 0.1 ms, single MVT, short vs full interval =====")
    print("Bala Table 3, 5-det row:  4.3 ± 0.8 ms")
    for name, interval in (("short", SHORT), ("full", FULL)):
        t0 = time.time()
        counts5 = binDetectors(DETECTOR_SETS["5det"], 1e-4, *interval)
        d5, du5, b5, bu5 = runBoth(counts5, 1e-4)
        print(f"\n{name} {list(interval)} s, total counts {int(counts5.sum())}  ({time.time() - t0:.0f} s)")
        print(f"  our code               {fmt(d5, du5)}")
        print(f"  Bala's code            {fmt(b5, bu5)}")
        report[f"part1_5det_{name}_0.1ms"] = dict(interval=list(interval), totalCounts=int(counts5.sum()),
                                                 ours_ms=[d5 * 1e3, du5 * 1e3], bala_ms=[b5 * 1e3, bu5 * 1e3])

    # ---------------- Part 2a ----------------
    print("\n===== PART 2a: 7 det, 0.01 ms, observed light curve =====")
    t0 = time.time()
    counts7 = binDetectors(DETECTOR_SETS["7det"], 1e-5, *SHORT)
    d7, du7, b7, bu7 = runBoth(counts7, 1e-5)
    print(f"total counts             {int(counts7.sum())}")
    print(f"our code                 {fmt(d7, du7)}")
    print(f"Bala's code              {fmt(b7, bu7)}")
    print(f"Run B single MVT (saved)     1.282 ±  0.434 ms")
    print(f"took {time.time() - t0:.0f} s")
    report["part2a_7det_0.01ms_observed"] = dict(totalCounts=int(counts7.sum()),
                                                 ours_ms=[d7 * 1e3, du7 * 1e3], bala_ms=[b7 * 1e3, bu7 * 1e3])

    # ---------------- Part 2b ----------------
    n = args.nreal
    print(f"\n===== PART 2b: first {n} realisations of Run B, both codes =====")
    seeds = np.random.SeedSequence(SEED).spawn(N_RUN_B)[:n]
    tasks = [(i, seeds[i], counts7, 1e-5) for i in range(n)]
    rows = [None] * n
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(oneRealisation, t) for t in tasks]
        for k, f in enumerate(as_completed(futures), 1):
            i, dm, du, bm, bu = f.result()
            rows[i] = (dm, du, bm, bu)
            print(f"  done {k:3d}/{n}  (realisation {i})", flush=True)
    minutes = (time.time() - t0) / 60
    rows = np.array(rows)
    dm, du, bm, bu = rows.T

    haveRunB = os.path.exists(RUN_B_VALUES) and os.path.exists(RUN_B_UNCS)
    if haveRunB:
        rbm, rbu = np.load(RUN_B_VALUES)[:n], np.load(RUN_B_UNCS)[:n]
    else:
        rbm = rbu = np.full(n, np.nan)

    print(f"\n{'i':>3s} | {'our code now (ms)':>28s} | {'Bala code (ms)':>28s} | {'diff (ms)':>9s} | {'Run B saved (ms)':>28s}")
    for i in range(n):
        print(f"{i:3d} | {fmt(dm[i], du[i]):>28s} | {fmt(bm[i], bu[i]):>28s} | {(bm[i] - dm[i]) * 1e3:+9.4f} | {fmt(rbm[i], rbu[i]):>28s}")

    typeD = np.array([classify(a, b) for a, b in zip(dm, du)])
    typeB = np.array([classify(a, b) for a, b in zip(bm, bu)])
    same = np.isclose(dm, bm, rtol=1e-6, atol=1e-12) & np.isclose(du, bu, rtol=1e-6, atol=1e-12)
    vD, vB = dm[typeD == "valid"], bm[typeB == "valid"]

    print("\n===== PART 2b SUMMARY =====")
    print(f"realisations compared         {n}   ({minutes:.1f} min)")
    print(f"our code now == Bala code      {int(same.sum())}/{n}")
    print(f"same classification            {int((typeD == typeB).sum())}/{n}")
    print(f"max |MVT difference|           {np.max(np.abs(bm - dm)) * 1e3:.4f} ms")
    print(f"valid: our code now {len(vD)}/{n}, Bala code {len(vB)}/{n}")
    if len(vD):
        print(f"median of valid: our code now {np.median(vD) * 1e3:.3f} ms")
    if len(vB):
        print(f"median of valid: Bala code    {np.median(vB) * 1e3:.3f} ms")
    if haveRunB:
        typeR = np.array([classify(a, b) for a, b in zip(rbm, rbu)])
        vR = rbm[typeR == "valid"]
        match = np.isclose(dm, rbm, rtol=1e-6, atol=1e-12)
        print(f"\nour code now == saved Run B values: {int(match.sum())}/{n}")
        print(f"saved Run B == Bala code:           {int(np.isclose(bm, rbm, rtol=1e-6, atol=1e-12).sum())}/{n}")
        print(f"saved Run B valid: {len(vR)}/{n}" + (f", median of valid {np.median(vR) * 1e3:.3f} ms" if len(vR) else ""))
        changed = int((typeR != typeD).sum())
        print(f"classification changed (saved Run B -> code now): {changed}/{n}")
        report["part2b_realisations_runB_saved"] = dict(validRunB=int(len(vR)),
            medianValidRunB_ms=float(np.median(vR) * 1e3) if len(vR) else None,
            runB_ms=(rbm * 1e3).tolist(), runBUnc_ms=(rbu * 1e3).tolist())

    report["part2b_realisations"] = dict(
        n=n, minutes=minutes, identical=int(same.sum()), sameType=int((typeD == typeB).sum()),
        maxAbsDiff_ms=float(np.max(np.abs(bm - dm)) * 1e3),
        validOurs=int(len(vD)), validBala=int(len(vB)),
        medianValidOurs_ms=float(np.median(vD) * 1e3) if len(vD) else None,
        medianValidBala_ms=float(np.median(vB) * 1e3) if len(vB) else None,
        ours_ms=(dm * 1e3).tolist(), oursUnc_ms=(du * 1e3).tolist(),
        bala_ms=(bm * 1e3).tolist(), balaUnc_ms=(bu * 1e3).tolist())
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved in {OUT_DIR}")
