"""
Test of the "merge" hypothesis for the SNR_MVT difference with Bala et al. (2026), Table 3.

Hypothesis: Bala et al. combine detectors with GbmTte.merge(...), whose default force_unique=True keeps only
one event per distinct arrival time (np.unique on the times). Photons from different detectors that fall on
the same time tick are real, separate photons, but all except one are deleted. This removes many photons in
bright peaks and almost none in the background, so the SNR drops, more so with more detectors and brighter
bursts. Our scripts add the detectors' light curves (or concatenate their event times), so no photon is lost.

For every Table 3 row of both bursts, at Bala et al.'s own median MVT:
  ours    detectors' 8-900 keV event times concatenated (as run_grb_mvt.py)
  merged  each detector sliced to the source interval, GbmTte.merge(...) with its default settings, then
          8-900 keV (the order used in Bala et al.'s code: merge, then light curve in the energy range)
  SNR_MVT = max(counts in bins of width MVT) / sqrt(background rate * MVT), with the same fitted background
  for both. Also: events lost, peak counts lost, and the range over 10 bin phases.

Criterion fixed before running: the hypothesis is supported if, in every row, Bala et al.'s Table 3 SNR is
within 5% of our merged SNR or inside its bin-phase range; and not supported if the merged SNR stays as far
from Table 3 as ours.

Usage
  python run_merge_test.py
"""
import os, sys, json, glob, time
import numpy as np
import gdt
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial

DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
OUT_DIR = "/mnt/data/rgl36/FYP/results/merge_test"
ENERGY_KEV = (8.0, 900.0)
BKG_BIN_S = 1.024
N_PHASES = 10
TOLERANCE = 0.05
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]

# row: detector set, interval (s), Bala et al. median MVT (ms), Bala et al. SNR_MVT  (Table 3)
GRBS = {
    "GRB230307A": dict(
        bkgIntervals=[(-68.8, -18.8), (69.4, 119.4)],
        detectorSets={"5det": ["n0", "n1", "n6", "n7", "na"],
                      "7det": ["n0", "n1", "n6", "n7", "n9", "na", "nb"], "all": ALL_NAI},
        rows={"B": ("7det", (-1.64, 18.0), 1.9, 94.2), "C": ("7det", (-1.64, 18.0), 4.2, 135.4),
              "D": ("5det", (-1.6, 45.3), 4.4, 137.0), "E": ("all", (-1.64, 18.0), 3.9, 116.2),
              "F": ("all", (-1.64, 18.0), 1.8, 82.1), "G": ("7det", (-1.64, 18.0), 1.7, 90.9),
              "H": ("all", (-1.64, 18.0), 1.8, 80.7)}),
    "GRB211211A": dict(
        bkgIntervals=[(-68.0, -18.0), (69.7, 119.7)],
        detectorSets={"1det": ["n2"], "3det": ["n1", "n2", "na"], "all": ALL_NAI},
        rows={"P": ("1det", (-0.9, 15.0), 8.8, 134.7), "Q": ("3det", (-0.9, 15.0), 4.7, 121.9),
              "R": ("3det", (-0.9, 15.0), 3.6, 113.5), "S": ("3det", (-0.9, 15.0), 3.9, 116.4),
              "T": ("all", (-0.9, 15.0), 4.4, 94.1), "U": ("all", (-0.9, 15.0), 3.4, 85.0),
              "V": ("all", (-0.9, 15.0), 3.1, 82.7)}),
}


def tteFile(grb, det):
    files = glob.glob(f"{DATA_ROOT}/{grb}/current/glg_tte_{det}_bn*_v0*.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    return files[0]


def backgroundRate(tte, bkgIntervals, interval):
    """Mean fitted 8-900 keV background rate over the interval (same method as run_grb_mvt.py)."""
    phaii = tte.to_phaii(bin_by_time, BKG_BIN_S, time_ref=0.0)
    fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgIntervals)
    fitter.fit(order=1)
    model = fitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop).integrate_energy(*ENERGY_KEV)
    tstart, tstop, rates = np.asarray(model.tstart), np.asarray(model.tstop), np.asarray(model.rates).ravel()
    inside = (tstart >= interval[0]) & (tstop <= interval[1])
    return float(np.mean(rates[inside]))


def peaks(times, mvtS, interval):
    """Max counts per MVT bin starting at the interval start, and the min/max over 10 bin phases."""
    out = []
    for k in range(N_PHASES):
        edges = np.arange(interval[0] + k / N_PHASES * mvtS, interval[1] + mvtS, mvtS)
        out.append(np.histogram(times, bins=edges)[0].max())
    return out[0], min(out), max(out)


if __name__ == "__main__":
    t0 = time.time()
    print(f"gdt-core version {getattr(gdt, '__version__', '?')}; GbmTte.merge default force_unique=True")
    report, allRows = {}, []
    for grb, cfg in GRBS.items():
        print(f"\n===== {grb} =====")
        cache = {}
        for interval in sorted({iv for _, iv, _, _ in cfg["rows"].values()}):
            neededDets = sorted({d for s, iv, _, _ in cfg["rows"].values() if iv == interval
                                 for d in cfg["detectorSets"][s]}, key=ALL_NAI.index)
            for det in neededDets:
                tte = GbmTte.open(tteFile(grb, det))
                sliced = tte.slice_time(interval)
                times = np.array(sliced.slice_energy(ENERGY_KEV).data.times)
                times = times[(times >= interval[0]) & (times < interval[1])]
                allTimes = np.array(sliced.data.times)
                cache[(det, interval)] = dict(tte=sliced, times=times, bkg=backgroundRate(tte, cfg["bkgIntervals"], interval),
                                              dupInside=int(allTimes.size - np.unique(allTimes).size),
                                              tick=float(np.min(np.diff(np.unique(allTimes)))))
                tte.close()
        ticks = sorted({round(v["tick"] * 1e6, 3) for v in cache.values()})
        print(f"smallest time step between events in any detector: {ticks} microseconds")
        print(f"repeated times inside a single detector: "
              + ", ".join(f"{d} {v['dupInside']}" for (d, iv), v in sorted(cache.items(), key=lambda x: ALL_NAI.index(x[0][0]))
                          if iv == sorted({iv for _, iv, _, _ in cfg['rows'].values()})[0]))

        print(f"\n{'row':4s}{'dets':>5s}{'MVT':>6s} | {'events ours':>12s}{'merged':>10s}{'lost%':>7s} | "
              f"{'peak ours':>10s}{'merged':>8s}{'lost%':>7s} | {'SNR ours':>9s}{'SNR merged':>11s}{'(phases)':>16s}"
              f"{'Bala':>8s} | {'ours/Bala':>10s}{'merged/Bala':>12s}{'match':>7s}")
        rows = {}
        for row, (detSet, interval, mvtMs, balaSnr) in cfg["rows"].items():
            dets = cfg["detectorSets"][detSet]
            mvtS = mvtMs / 1e3
            ours = np.sort(np.concatenate([cache[(d, interval)]["times"] for d in dets]))
            merged = GbmTte.merge([cache[(d, interval)]["tte"] for d in dets])
            mergedTimes = np.array(merged.slice_energy(ENERGY_KEV).data.times)
            mergedTimes = mergedTimes[(mergedTimes >= interval[0]) & (mergedTimes < interval[1])]
            bkg = sum(cache[(d, interval)]["bkg"] for d in dets)
            noise = np.sqrt(bkg * mvtS)
            pOurs, _, _ = peaks(ours, mvtS, interval)
            pMerged, pMin, pMax = peaks(mergedTimes, mvtS, interval)
            snrOurs, snrMerged = pOurs / noise, pMerged / noise
            lowM, highM = pMin / noise, pMax / noise
            match = abs(snrMerged / balaSnr - 1) <= TOLERANCE or lowM <= balaSnr <= highM
            lostEv = 100 * (1 - mergedTimes.size / ours.size)
            lostPk = 100 * (1 - pMerged / pOurs)
            print(f"{row:4s}{len(dets):>5d}{mvtMs:>6.1f} | {ours.size:>12d}{mergedTimes.size:>10d}{lostEv:>7.1f} | "
                  f"{pOurs:>10d}{pMerged:>8d}{lostPk:>7.1f} | {snrOurs:>9.1f}{snrMerged:>11.1f}{f'{lowM:.1f}-{highM:.1f}':>16s}"
                  f"{balaSnr:>8.1f} | {snrOurs / balaSnr:>10.2f}{snrMerged / balaSnr:>12.2f}{'YES' if match else 'NO':>7s}")
            rows[row] = dict(detectors=dets, interval_s=list(interval), mvt_ms=mvtMs, bkgCps=bkg,
                             eventsOurs=int(ours.size), eventsMerged=int(mergedTimes.size), eventsLostPct=lostEv,
                             peakOurs=int(pOurs), peakMerged=int(pMerged), peakLostPct=lostPk,
                             snrOurs=snrOurs, snrMerged=snrMerged, snrMergedPhaseRange=[lowM, highM],
                             snrBala=balaSnr, ratioOurs=snrOurs / balaSnr, ratioMerged=snrMerged / balaSnr, match=bool(match))
            allRows.append(rows[row])
        report[grb] = dict(rows=rows, tickMicroseconds=ticks)

    nMatch = sum(r["match"] for r in allRows)
    meanOurs = np.mean([abs(r["ratioOurs"] - 1) for r in allRows]) * 100
    meanMerged = np.mean([abs(r["ratioMerged"] - 1) for r in allRows]) * 100
    print("\n===== VERDICT (criterion set before running) =====")
    print(f"rows where Table 3 SNR is within {TOLERANCE:.0%} of the merged SNR or inside its phase range: {nMatch}/{len(allRows)}")
    print(f"mean |SNR / Bala - 1|:  ours {meanOurs:.1f}%   merged {meanMerged:.1f}%")
    print("hypothesis " + ("SUPPORTED" if nMatch == len(allRows) else
                           "NOT supported" if meanMerged >= meanOurs else "PARTLY supported"))
    report["verdict"] = dict(matches=nMatch, rows=len(allRows), meanDevOursPct=meanOurs, meanDevMergedPct=meanMerged)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved in {OUT_DIR}  ({(time.time() - t0) / 60:.1f} min)")
