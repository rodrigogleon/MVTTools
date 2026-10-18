"""
GRB 230307A: "cleaner light curve" test.

Hypothesis: our light curves have more signal relative to noise than Bala et al.'s. This would explain,
with one cause, the three remaining differences with their Table 3:
  1. our SNR_MVT is 20-33% higher
  2. more of our realisations are valid at fine bins
  3. our error bars are 10-35% narrower

Test: make our light curve "noisier" by the exact amount needed for our SNR_MVT to equal Bala et al.'s
Table 3 value, rerun the same Monte Carlo, and check whether the success rate and the error bars also
move to their values. Nothing else changes (same detectors, bins, interval, energy range, seed).

Two ways of making it noisier (--mode):
  bkg    add a flat extra background of X counts/s.  SNR uses background b + X.
         Solved exactly: with A = peak - b*MVT (source counts in the peak bin) and y = sqrt((b+X)*MVT),
         SNR = A / y + y  ->  y = (S - sqrt(S^2 - 4A)) / 2
  scale  scale the source down by a factor f, background unchanged.
         SNR = (f*A + b*MVT) / sqrt(b*MVT)  ->  f = (S*sqrt(b*MVT) - b*MVT) / A
Each realisation is then Poisson(expected counts), errors sqrt(realised counts), as in the haar-fix runs.

Calibration is done at Bala et al.'s median MVT, against their Table 3 SNR, using our observed light curve
and the background from results/snr_classify/summary.json.

Usage
  python run_extra_bkg.py --run B --nreal 0          calibration only (seconds), no Monte Carlo
  python run_extra_bkg.py --run B                    calibration + 300 realisations
  python run_extra_bkg.py --run F --mode scale
"""
import os, sys, json, time, glob, argparse, subprocess
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
GRB = "GRB230307A"
DATA_DIR = f"/mnt/data/rgl36/FYP/data/fermiGBM/{GRB}/current"
CONTROL_ROOT = "/mnt/data/rgl36/FYP/results/bala_rows_8-900keV_rs_haarfix"
SNR_SUMMARY = "/mnt/data/rgl36/FYP/results/snr_classify/summary.json"
OUT_ROOT = "/mnt/data/rgl36/FYP/results/extra_bkg"

ENERGY_KEV = (8.0, 900.0)
SEED = 2026
N_SNR_CHECK = 100

DETECTOR_SETS = {
    "5det": ["n0", "n1", "n6", "n7", "na"],
    "7det": ["n0", "n1", "n6", "n7", "n9", "na", "nb"],
    "all": ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"],
}
SHORT = (-1.64, 18.0)

# fine-bin rows only: at 0.1 ms every realisation is already valid, so the test has no power there
RUNS = {
    "B": ("7det", 1e-5, SHORT, dict(median=1.9, plus=0.7, minus=0.4, snr=94.2, success=44.3)),
    "F": ("all",  1e-5, SHORT, dict(median=1.8, plus=0.4, minus=0.3, snr=82.1, success=62.3)),
    "G": ("7det", 3e-6, SHORT, dict(median=1.7, plus=0.7, minus=0.4, snr=90.9, success=36.3)),
    "H": ("all",  3e-6, SHORT, dict(median=1.8, plus=0.5, minus=0.3, snr=80.7, success=57.7)),
}

parser = argparse.ArgumentParser()
parser.add_argument("--run", choices=list(RUNS), required=True)
parser.add_argument("--mode", choices=["bkg", "scale"], default="bkg")
parser.add_argument("--nreal", type=int, default=300, help="0 = calibration only")
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

sys.path.insert(0, REPO)
os.chdir(REPO)
from loggerSetup import initialiseLogging
initialiseLogging()
import analysisTools.haarMethods as haarMethods
from gdt.missions.fermi.gbm.tte import GbmTte


def loadTimes(detectors, tmin, tmax):
    times = []
    for det in detectors:
        files = glob.glob(f"{DATA_DIR}/glg_tte_{det}_bn*_v00.fit")
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
        tte = GbmTte.open(files[0])
        times.append(np.array(tte.slice_energy(ENERGY_KEV).data.times))
        tte.close()
    times = np.sort(np.concatenate(times))
    return times[(times >= tmin) & (times < tmax)]


def percent(x, n):
    return 100.0 * x / n


if __name__ == "__main__":
    detSet, bw, (tmin, tmax), b = RUNS[args.run]
    detectors = DETECTOR_SETS[detSet]
    commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"],
                            capture_output=True, text=True).stdout.strip()
    print(f"MVTTools code in use: {commit}")
    print(f"\n===== RUN {args.run} ({detSet}, bin {bw * 1e3:g} ms, {[tmin, tmax]} s), mode '{args.mode}' =====")

    # ---------- 1. light curve and background (same as the haar-fix run and run_snr_classify) ----------
    times = loadTimes(detectors, tmin, tmax)
    edges = np.arange(tmin, tmax + bw, bw)
    counts = np.histogram(times, bins=edges)[0].astype(float)
    control = np.load(os.path.join(CONTROL_ROOT, f"Run{args.run}", "counts.npy"))
    sameCounts = len(control) == len(counts) and np.array_equal(control, counts)
    print(f"light curve: {len(counts)} bins, {int(counts.sum())} counts; identical to the haar-fix run: {sameCounts}")
    if not sameCounts:
        sys.exit("STOP: light curve differs from the haar-fix run, the comparison would not be like for like")

    bkgCps = json.load(open(SNR_SUMMARY))["runs"][args.run]["bkgCps_band"]

    # ---------- 2. calibration at Bala et al.'s median MVT ----------
    mvtS = b["median"] / 1e3
    k = int(round(mvtS / bw))                     # fine bins per MVT bin
    nGroups = len(counts) // k
    peak = counts[:nGroups * k].reshape(nGroups, k).sum(axis=1).max()
    bkgPerMvtBin = bkgCps * mvtS
    A = peak - bkgPerMvtBin                       # source counts in the peak bin
    S = b["snr"]
    snrNow = peak / np.sqrt(bkgPerMvtBin)

    if args.mode == "bkg":
        disc = S ** 2 - 4 * A
        if disc < 0:
            sys.exit(f"STOP: no extra background can bring SNR down to {S} (minimum possible {2 * np.sqrt(A):.1f})")
        y = (S - np.sqrt(disc)) / 2
        extraCps = y ** 2 / mvtS - bkgCps
        factor = 1.0
        expected = counts + extraCps * bw
        snrBkgCps = bkgCps + extraCps
        change = f"extra background X = {extraCps:.0f} c/s  (total background x{(bkgCps + extraCps) / bkgCps:.2f})"
    else:
        factor = (S * np.sqrt(bkgPerMvtBin) - bkgPerMvtBin) / A
        extraCps = 0.0
        bkgPerFineBin = bkgCps * bw
        expected = np.clip(bkgPerFineBin + factor * (counts - bkgPerFineBin), 0.0, None)
        snrBkgCps = bkgCps
        change = f"source scaled by f = {factor:.3f}  (background unchanged)"
    if extraCps < 0 or factor > 1:
        sys.exit("STOP: our SNR is already below Bala et al.'s, the test does not apply")

    # check: SNR of the modified light curve (expected counts, and mean over Poisson copies)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 99]))
    grouped = lambda c: c[:nGroups * k].reshape(nGroups, k).sum(axis=1)
    snrExpected = grouped(expected).max() / np.sqrt(snrBkgCps * mvtS)
    snrCopies = np.array([grouped(rng.poisson(expected)).max() for _ in range(N_SNR_CHECK)]) / np.sqrt(snrBkgCps * mvtS)

    print(f"\nbackground {bkgCps:.0f} c/s, MVT {b['median']} ms ({k} fine bins), peak {int(peak)} counts, "
          f"background {bkgPerMvtBin:.2f} counts per MVT bin")
    print(f"SNR now {snrNow:.1f}  ->  target (Bala Table 3) {S}")
    print(f"change needed: {change}")
    print(f"check, modified light curve: SNR {snrExpected:.1f} (expected counts), "
          f"{snrCopies.mean():.1f} ± {snrCopies.std():.1f} (mean of {N_SNR_CHECK} Poisson copies)")
    print(f"mean counts per fine bin: {counts.mean():.3f} -> {expected.mean():.3f}")

    report = dict(run=args.run, mode=args.mode, mvttoolsCommit=commit, detectors=detectors, binSize_s=bw,
                  timeRange_s=[tmin, tmax], energyRange_keV=list(ENERGY_KEV), seed=SEED,
                  bkgCps=bkgCps, calibrationMvt_ms=b["median"], peakCounts=int(peak),
                  snrBefore=snrNow, snrTarget=S, extraBkgCps=extraCps, sourceFactor=factor,
                  snrAfterExpected=snrExpected, snrAfterPoissonMean=float(snrCopies.mean()))
    if args.nreal == 0:
        print("\n--nreal 0: calibration only, no Monte Carlo.")
        sys.exit(0)

    # ---------- 3. Monte Carlo, identical settings to the haar-fix runs ----------
    outDir = os.path.join(OUT_ROOT, f"Run{args.run}_{args.mode}")
    os.makedirs(outDir, exist_ok=True)
    t0 = time.time()
    r = haarMethods.estimateMvtDistribution(
        expected, np.sqrt(expected), haarMethods.haarPowerMod,
        binSizeSeconds=bw, nRealisations=args.nreal,
        nWorkers=args.workers, seed=SEED, resampleErrors=True)
    minutes = (time.time() - t0) / 60
    np.save(os.path.join(outDir, "mvt_values_seconds.npy"), r["mvtValues"])
    np.save(os.path.join(outDir, "mvt_uncertainties_seconds.npy"), r["mvtUncertainties"])

    n = r["nRealisations"]
    ms = lambda x: float(x) * 1e3
    test = dict(median=ms(r["median"]), plus=ms(r["p84"] - r["median"]), minus=ms(r["median"] - r["p16"]),
                p84=ms(r["p84"]), valid=percent(r["nValid"], n), limit=percent(r["nLimit"], n),
                failed=percent(r["nFailed"], n))
    c = json.load(open(os.path.join(CONTROL_ROOT, f"Run{args.run}", "summary.json")))
    ctrl = dict(median=c["median_ms"], plus=c["plus_err_ms"], minus=c["minus_err_ms"], p84=c["p84_ms"],
                valid=c["validPercent"], limit=c["limitPercent"], failed=c["failedPercent"])
    bala = dict(median=b["median"], plus=b["plus"], minus=b["minus"], p84=b["median"] + b["plus"],
                valid=b["success"], limit=100 - b["success"], failed=float("nan"))

    print(f"\n===== RESULT, run {args.run}, mode '{args.mode}', {n} realisations ({minutes:.0f} min) =====")
    print(f"{'':22s}{'median':>8s}{'+err':>7s}{'-err':>7s}{'MVT84':>8s}{'valid%':>8s}{'UL%':>7s}{'fail%':>7s}{'SNR':>7s}")
    for name, x, snr in (("control (haar-fix)", ctrl, snrNow), ("this test", test, snrExpected), ("Bala Table 3", bala, S)):
        print(f"{name:22s}{x['median']:>8.2f}{x['plus']:>7.2f}{x['minus']:>7.2f}{x['p84']:>8.2f}{x['valid']:>8.1f}"
              f"{x['limit']:>7.1f}{x['failed']:>7.1f}{snr:>7.1f}")
    print("Bala UL% column = their 'failed' (our UL + fail).")

    # verdict, criteria fixed before running
    p = b["success"] / 100
    sigma = 100 * np.sqrt(p * (1 - p) / n)
    successOk = abs(test["valid"] - b["success"]) <= 2 * sigma
    plusCloser = abs(test["plus"] - b["plus"]) < abs(ctrl["plus"] - b["plus"])
    minusCloser = abs(test["minus"] - b["minus"]) < abs(ctrl["minus"] - b["minus"])
    medianOk = abs(test["median"] - b["median"]) <= 0.2
    print("\n----- criteria (set before the run) -----")
    print(f"1. valid% within 2 sigma of Bala ({b['success']} ± {2 * sigma:.1f}):  {test['valid']:.1f}  -> {'YES' if successOk else 'NO'}")
    print(f"2. +error closer to Bala than control:  {ctrl['plus']:.2f} -> {test['plus']:.2f} (Bala {b['plus']})  -> {'YES' if plusCloser else 'NO'}")
    print(f"3. -error closer to Bala than control:  {ctrl['minus']:.2f} -> {test['minus']:.2f} (Bala {b['minus']})  -> {'YES' if minusCloser else 'NO'}")
    print(f"4. median still within 0.2 ms of Bala:  {test['median']:.2f} (Bala {b['median']})  -> {'YES' if medianOk else 'NO'}")
    allOk = successOk and plusCloser and minusCloser and medianOk
    print(f"hypothesis {'SUPPORTED' if allOk else 'NOT fully supported'} for run {args.run} ({sum([successOk, plusCloser, minusCloser, medianOk])}/4)")

    report.update(nRealisations=n, minutes=minutes, test=test, control=ctrl, bala=bala,
                  criteria=dict(success=successOk, plusCloser=plusCloser, minusCloser=minusCloser, median=medianOk))
    with open(os.path.join(outDir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved in {outDir}")
