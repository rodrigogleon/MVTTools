"""
Detector selection by iterative SNR maximisation (Bala et al. 2026, Section 4.1, Figures 12-13).

Procedure, as described in the paper and in their public code (TTE_SIM_v2.py, find_optimal_detectors_for_run):
  1. For every NaI detector, build the light curve in the source interval (8-900 keV) with a bin width of
     64 ms (16 ms if T90 < 1 s) and compute its SNR = max(counts per bin) / sqrt(background rate * bin width).
  2. Rank the detectors from highest to lowest SNR.
  3. Add them one at a time in that order, summing their light curves (and their background rates), and
     compute the combined SNR at each step.
  4. The optimal combination is the one with the maximum combined SNR.
  5. Repeat steps 1-4 with the bin width equal to the measured MVT; if the optimal set changes, use the new one.

Background per detector: first-order polynomial fitted to the off-burst intervals on 1.024 s bins
(time_ref 0), interpolated over the burst and averaged over the source interval (same as run_snr_classify.py).

Usage
  python run_detector_selection.py                                   GRB 230307A, short interval, MVT 4.19 and 1.8 ms
  python run_detector_selection.py --interval -1.6 45.3              full interval
  python run_detector_selection.py --grb GRB211211A --interval A B --bkg a1 a2 b1 b2 --t90 X --mvt m1 m2
"""
import os, sys, json, glob, argparse
import numpy as np
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial

DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
OUT_DIR = "/mnt/data/rgl36/FYP/results/detector_selection"
NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]
ENERGY_KEV = (8.0, 900.0)
BKG_BIN_S = 1.024
BKG_ORDER = 1

parser = argparse.ArgumentParser()
parser.add_argument("--grb", default="GRB230307A")
parser.add_argument("--interval", type=float, nargs=2, default=[-1.64, 18.0], help="source interval (s)")
parser.add_argument("--bkg", type=float, nargs=4, default=[-68.8, -18.8, 69.4, 119.4],
                    help="off-burst intervals: start1 stop1 start2 stop2 (s)")
parser.add_argument("--t90", type=float, default=34.56, help="T90 (s); sets the first bin width (64 ms, or 16 ms if < 1 s)")
parser.add_argument("--mvt", type=float, nargs="*", default=[4.19, 1.8],
                    help="measured MVTs (ms) for the repeat step (bin width = MVT)")
args = parser.parse_args()

TMIN, TMAX = args.interval
BKG_INTERVALS = [(args.bkg[0], args.bkg[1]), (args.bkg[2], args.bkg[3])]


def loadDetector(det):
    """Event times in 8-900 keV inside the source interval, and the mean fitted background rate over it."""
    files = glob.glob(f"{DATA_ROOT}/{args.grb}/current/glg_tte_{det}_bn*_v00.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    tte = GbmTte.open(files[0])
    times = np.array(tte.slice_energy(ENERGY_KEV).data.times)
    phaii = tte.to_phaii(bin_by_time, BKG_BIN_S, time_ref=0.0)
    tte.close()
    fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=BKG_INTERVALS)
    fitter.fit(order=BKG_ORDER)
    model = fitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop).integrate_energy(*ENERGY_KEV)
    tstart, tstop = np.asarray(model.tstart), np.asarray(model.tstop)
    rates = np.asarray(model.rates).ravel()
    inside = (tstart >= TMIN) & (tstop <= TMAX)
    return times[(times >= TMIN) & (times < TMAX)], float(np.mean(rates[inside]))


def lightCurve(times, bwS):
    edges = np.arange(TMIN, TMAX + bwS, bwS)
    return np.histogram(times, bins=edges)[0]


def snr(counts, bkgCps, bwS):
    return counts.max() / np.sqrt(bkgCps * bwS)


def selectDetectors(curves, bkg, bwS):
    """Rank detectors by individual SNR, add them cumulatively, return the table and the optimal set."""
    single = {d: snr(curves[d], bkg[d], bwS) for d in curves}
    ranked = sorted(single, key=single.get, reverse=True)
    rows = []
    combined = np.zeros_like(curves[ranked[0]])
    bkgSum = 0.0
    for k, det in enumerate(ranked, 1):
        combined = combined + curves[det]
        bkgSum += bkg[det]
        rows.append(dict(k=k, added=det, dets=ranked[:k], snr=snr(combined, bkgSum, bwS),
                         peak=int(combined.max()), bkgPerBin=bkgSum * bwS))
    best = max(rows, key=lambda r: r["snr"])
    return single, ranked, rows, best


def printSelection(title, bwS, single, ranked, rows, best):
    print(f"\n===== {title}: bin width {bwS * 1e3:g} ms =====")
    print("individual SNR (ranked): " + "  ".join(f"{d} {single[d]:.1f}" for d in ranked))
    print(f"{'k':>3s} {'added':>6s} {'combined SNR':>13s} {'peak':>8s} {'bkg/bin':>9s}")
    for r in rows:
        mark = "  <- optimal" if r is best else ""
        print(f"{r['k']:>3d} {r['added']:>6s} {r['snr']:>13.1f} {r['peak']:>8d} {r['bkgPerBin']:>9.2f}{mark}")
    print(f"optimal set ({best['k']} detectors): {sorted(best['dets'], key=NAI.index)}")


if __name__ == "__main__":
    print(f"{args.grb}, source interval [{TMIN}, {TMAX}] s, {ENERGY_KEV[0]:g}-{ENERGY_KEV[1]:g} keV, "
          f"background order {BKG_ORDER} on {BKG_INTERVALS} s")
    events, bkg = {}, {}
    for det in NAI:
        events[det], bkg[det] = loadDetector(det)
    print("background (c/s): " + "  ".join(f"{d} {bkg[d]:.0f}" for d in NAI))

    report = dict(grb=args.grb, interval_s=[TMIN, TMAX], bkgIntervals_s=BKG_INTERVALS, energyRange_keV=list(ENERGY_KEV),
                  t90_s=args.t90, bkgCps=bkg, steps=[])

    firstBw = 0.016 if args.t90 < 1.0 else 0.064
    steps = [("Step 1 (initial)", firstBw)] + [(f"Repeat with bin width = MVT {m:g} ms", m / 1e3) for m in args.mvt]
    optimalSets = []
    for title, bwS in steps:
        curves = {d: lightCurve(events[d], bwS) for d in NAI}
        single, ranked, rows, best = selectDetectors(curves, bkg, bwS)
        printSelection(title, bwS, single, ranked, rows, best)
        optimalSets.append(sorted(best["dets"], key=NAI.index))
        report["steps"].append(dict(title=title, binWidth_s=bwS, singleSnr=single, ranked=ranked,
                                    cumulative=[dict(k=r["k"], added=r["added"], snr=r["snr"], peak=r["peak"],
                                                     bkgPerBin=r["bkgPerBin"]) for r in rows],
                                    optimal=optimalSets[-1], optimalSnr=best["snr"]))

    print("\n===== SUMMARY =====")
    for (title, bwS), dets in zip(steps, optimalSets):
        print(f"{title:38s} ({bwS * 1e3:g} ms): {len(dets)} detectors {dets}")
    if args.grb == "GRB230307A":
        print("\nBala et al. (2026) for GRB 230307A: initial optimal set = 5 detectors; Table 3 rows use")
        print("  5det ['n0', 'n1', 'n6', 'n7', 'na'], 7det ['n0', 'n1', 'n6', 'n7', 'n9', 'na', 'nb'], all 12")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.grb}_{TMIN:g}_{TMAX:g}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved in {out}")
