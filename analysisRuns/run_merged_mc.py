"""
Monte Carlo MVT on "merged" light curves: does the photon loss of GbmTte.merge also explain the remaining
MVT differences with Bala et al. (2026), Table 3 (higher success rates, narrower errors, lower 12-detector
medians)?

The light curve is built exactly as in our haar-fix runs, except that the detectors are combined with
GbmTte.merge(...) with its default force_unique=True (as in Bala et al.'s code), which keeps only one photon
per arrival-time tick. Everything else is identical: 8-900 keV, same bins and interval, 300 realisations,
seed 2026, errors sqrt(counts) per realisation, the same MVTTools code.

Criteria fixed before running (tolerances set by the precision of Table 3, whose errors are rounded to 0.1 ms):
  1. valid % within 2 sigma (binomial, 300 realisations) of Bala et al.'s success %
  2. median within 0.2 ms of Bala et al.
  3. + error within 0.1 ms of Bala et al.
  4. - error within 0.1 ms of Bala et al.
The merge explains a row if it passes all four; the comparison with our all-photon run (control) shows
whether it moved towards Bala et al.

Usage
  python run_merged_mc.py --grb GRB230307A --run B
  python run_merged_mc.py --grb GRB230307A --run F
"""
import os, sys, json, time, glob, argparse, subprocess
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
RESULTS = "/mnt/data/rgl36/FYP/results"
OUT_ROOT = f"{RESULTS}/merged_mc"
ENERGY_KEV = (8.0, 900.0)
N_REALISATIONS = 300
SEED = 2026
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]

# row: detectors, bin width (s), interval (s), Bala et al. Table 3 values, control folder
GRBS = {
    "GRB230307A": dict(control=f"{RESULTS}/bala_rows_8-900keV_rs_haarfix", scale=f"{RESULTS}/extra_bkg", runs={
        "B": (["n0", "n1", "n6", "n7", "n9", "na", "nb"], 1e-5, (-1.64, 18.0), dict(median=1.9, plus=0.7, minus=0.4, success=44.3)),
        "F": (ALL_NAI, 1e-5, (-1.64, 18.0), dict(median=1.8, plus=0.4, minus=0.3, success=62.3)),
        "G": (["n0", "n1", "n6", "n7", "n9", "na", "nb"], 3e-6, (-1.64, 18.0), dict(median=1.7, plus=0.7, minus=0.4, success=36.3)),
        "H": (ALL_NAI, 3e-6, (-1.64, 18.0), dict(median=1.8, plus=0.5, minus=0.3, success=57.7)),
    }),
    "GRB211211A": dict(control=f"{RESULTS}/bala_rows_211211A", scale=None, runs={
        "R": (["n1", "n2", "na"], 1e-5, (-0.9, 15.0), dict(median=3.6, plus=1.6, minus=0.8, success=46.3)),
        "S": (["n1", "n2", "na"], 3e-6, (-0.9, 15.0), dict(median=3.9, plus=1.0, minus=1.0, success=42.7)),
        "U": (ALL_NAI, 1e-5, (-0.9, 15.0), dict(median=3.4, plus=0.9, minus=0.9, success=58.3)),
        "V": (ALL_NAI, 3e-6, (-0.9, 15.0), dict(median=3.1, plus=1.1, minus=0.7, success=54.7)),
    }),
}

parser = argparse.ArgumentParser()
parser.add_argument("--grb", choices=list(GRBS), required=True)
parser.add_argument("--run", required=True)
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()
CFG = GRBS[args.grb]
if args.run not in CFG["runs"]:
    parser.error(f"--run must be one of {list(CFG['runs'])} for {args.grb}")

sys.path.insert(0, REPO)
os.chdir(REPO)
from loggerSetup import initialiseLogging
initialiseLogging()
import analysisTools.haarMethods as haarMethods
from gdt.missions.fermi.gbm.tte import GbmTte


def tteFile(det):
    files = glob.glob(f"{DATA_ROOT}/{args.grb}/current/glg_tte_{det}_bn*_v0*.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    return files[0]


if __name__ == "__main__":
    dets, bw, (tmin, tmax), b = CFG["runs"][args.run]
    commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"],
                            capture_output=True, text=True).stdout.strip()
    print(f"MVTTools code in use: {commit}")
    print(f"\n===== {args.grb} run {args.run}: {len(dets)} detectors, bin {bw * 1e3:g} ms, [{tmin}, {tmax}] s, "
          f"MERGED light curve =====")

    # 1. light curve from the merged event list
    ttes = []
    for d in dets:
        tte = GbmTte.open(tteFile(d))
        ttes.append(tte.slice_time((tmin, tmax)))
        tte.close()
    times = np.array(GbmTte.merge(ttes).slice_energy(ENERGY_KEV).data.times)
    edges = np.arange(tmin, tmax + bw, bw)
    counts = np.histogram(times, bins=edges)[0].astype(float)
    control = np.load(os.path.join(CFG["control"], f"Run{args.run}", "counts.npy"))
    if len(control) != len(counts):
        sys.exit("STOP: bins differ from the control run")
    k = int(round(b["median"] / 1e3 / bw))
    n = len(counts) // k
    peak = lambda c: c[:n * k].reshape(n, k).sum(axis=1).max()
    print(f"photons: all-photon (control) {int(control.sum())}, merged {int(counts.sum())}, "
          f"lost {100 * (1 - counts.sum() / control.sum()):.1f}%")
    print(f"peak in {b['median']} ms bins: control {int(peak(control))}, merged {int(peak(counts))}, "
          f"lost {100 * (1 - peak(counts) / peak(control)):.1f}%")
    print(f"bins with fewer counts than control: {100 * np.mean(counts < control):.2f}%; more: {int(np.sum(counts > control))}")

    outDir = os.path.join(OUT_ROOT, f"{args.grb}_Run{args.run}")
    os.makedirs(outDir, exist_ok=True)
    np.save(os.path.join(outDir, "counts.npy"), counts)

    # 2. Monte Carlo, identical settings to the haar-fix runs
    t0 = time.time()
    r = haarMethods.estimateMvtDistribution(
        counts, np.sqrt(counts), haarMethods.haarPowerMod,
        binSizeSeconds=bw, nRealisations=N_REALISATIONS,
        nWorkers=args.workers, seed=SEED, resampleErrors=True)
    minutes = (time.time() - t0) / 60
    np.save(os.path.join(outDir, "mvt_values_seconds.npy"), r["mvtValues"])
    np.save(os.path.join(outDir, "mvt_uncertainties_seconds.npy"), r["mvtUncertainties"])

    ms = lambda x: float(x) * 1e3
    nr = r["nRealisations"]
    test = dict(median=ms(r["median"]), plus=ms(r["p84"] - r["median"]), minus=ms(r["median"] - r["p16"]),
                p84=ms(r["p84"]), valid=100 * r["nValid"] / nr, limit=100 * r["nLimit"] / nr,
                failed=100 * r["nFailed"] / nr)
    c = json.load(open(os.path.join(CFG["control"], f"Run{args.run}", "summary.json")))
    ctrl = dict(median=c["median_ms"], plus=c["plus_err_ms"], minus=c["minus_err_ms"], p84=c["p84_ms"],
                valid=c["validPercent"], limit=c["limitPercent"], failed=c["failedPercent"])
    bala = dict(median=b["median"], plus=b["plus"], minus=b["minus"], p84=b["median"] + b["plus"],
                valid=b["success"], limit=100 - b["success"], failed=float("nan"))
    rows = [("control (all photons)", ctrl), ("merged", test)]
    if CFG["scale"]:
        sp = os.path.join(CFG["scale"], f"Run{args.run}_scale", "summary.json")
        if os.path.exists(sp):
            rows.append(("scale test (18 Oct)", json.load(open(sp))["test"]))
    rows.append(("Bala Table 3", bala))

    print(f"\n===== RESULT, {args.grb} run {args.run}, {nr} realisations ({minutes:.0f} min) =====")
    print(f"{'':24s}{'median':>8s}{'+err':>7s}{'-err':>7s}{'MVT84':>8s}{'valid%':>8s}{'UL%':>7s}{'fail%':>7s}")
    for name, x in rows:
        print(f"{name:24s}{x['median']:>8.2f}{x['plus']:>7.2f}{x['minus']:>7.2f}{x['p84']:>8.2f}{x['valid']:>8.1f}"
              f"{x['limit']:>7.1f}{x['failed']:>7.1f}")
    print("Bala UL% column = their 'failed' (our UL + fail).")

    p = b["success"] / 100
    sigma = 100 * np.sqrt(p * (1 - p) / nr)
    checks = [
        ("valid% within 2 sigma of Bala", f"{test['valid']:.1f} vs {b['success']} ± {2 * sigma:.1f}",
         abs(test["valid"] - b["success"]) <= 2 * sigma),
        ("median within 0.2 ms", f"{test['median']:.2f} vs {b['median']}", abs(test["median"] - b["median"]) <= 0.2),
        ("+error within 0.1 ms", f"{test['plus']:.2f} vs {b['plus']}", abs(test["plus"] - b["plus"]) <= 0.1),
        ("-error within 0.1 ms", f"{test['minus']:.2f} vs {b['minus']}", abs(test["minus"] - b["minus"]) <= 0.1),
    ]
    print("\n----- criteria (set before the run) -----")
    for i, (name, vals, ok) in enumerate(checks, 1):
        print(f"{i}. {name:32s} {vals:28s} -> {'YES' if ok else 'NO'}")
    nOk = sum(ok for *_, ok in checks)
    print(f"merge {'EXPLAINS' if nOk == 4 else 'does NOT fully explain'} run {args.run} ({nOk}/4)")
    print("moved towards Bala (|merged - Bala| < |control - Bala|): "
          + ", ".join(f"{key} {'yes' if abs(test[key] - bala[key]) < abs(ctrl[key] - bala[key]) else 'no'}"
                      for key in ("valid", "median", "plus", "minus", "p84")))

    json.dump(dict(grb=args.grb, run=args.run, mvttoolsCommit=commit, detectors=dets, binSize_s=bw,
                   timeRange_s=[tmin, tmax], photonsControl=int(control.sum()), photonsMerged=int(counts.sum()),
                   minutes=minutes, test=test, control=ctrl, bala=bala,
                   criteria={name: bool(ok) for name, _, ok in checks}),
              open(os.path.join(outDir, "summary.json"), "w"), indent=2, default=float)
    print(f"\nSaved in {outDir}")
