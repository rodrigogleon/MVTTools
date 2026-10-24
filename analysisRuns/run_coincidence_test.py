"""
Are the photons removed by GbmTte.merge explained by random coincidences plus a steady rate of genuine duplicates?

GbmTte.merge with its default force_unique=True keeps one event per distinct arrival time. Times are stored in
ticks of 2 microseconds (measured from the data; the stored times differ by 1.907 or 2.027 microseconds because of
floating-point rounding, averaging 2.000), and a detector never records two events in the same tick (no event
times repeat within one detector), so an event is removed only when another detector has an event in the same tick.

Prediction, two parts:
  1. random coincidences of separate photons: if detector i has probability p_i of an event in a given tick, the
     expected number of events per tick is sum(p_i) before the merge and 1 - prod(1 - p_i) after it, so the
     fraction removed is f = 1 - (1 - prod(1 - p_i)) / sum(p_i). p_i is taken from the detector's own counts in
     each 0.1 ms bin: p_i = counts_i * tick / 0.1 ms (bins much shorter than the burst variability, because
     coincidences grow with the square of the rate).
  2. genuine duplicates (one event recorded by several detectors; the pair-gap test showed they come mainly from
     the background and arrive at a roughly steady rate): measured for each detector set in a background-only
     window (-60 to -20 s) as observed deletions minus predicted random deletions, per second, and added to the
     burst prediction at that steady rate. Assumption: the duplicate rate during the burst equals the rate in the
     background window.
The test uses all energy channels (the merge acts before any energy cut) over the source interval, for each
detector set of Bala et al.'s Table 3. It also reports the fraction removed from the 8-900 keV light curve.

Changes from the first version: tick 2 microseconds instead of the smallest stored step (1.907 microseconds,
which made the random prediction about 5% too low); duplicates added to the prediction; the count rate used to
group the 10 ms bins is taken from the neighbouring bins (5 on each side), not the bin itself, so that a bin's own
fluctuation (which includes its duplicates) cannot decide its rate group.

Criteria fixed before running (same tolerances as the first version):
  1. one detector: no event removed
  2. the measured tick (mean of the smallest steps between events) is within 1% of 2 microseconds in every set
     with two or more detectors (1.907 microseconds, the value wrongly used before, is 4.6% away)
  3. for every detector set, the total fraction removed agrees with random + duplicates within 5% (relative)
  4. in every count-rate bin with at least 10^4 events, observed and random + duplicates agree within 3 sigma
     (binomial) or within 5% (relative), whichever is larger

Output: results/coincidence_test/summary.json and plot P7 in results/plots/

Usage
  python run_coincidence_test.py
"""
import os, json, glob, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gdt.missions.fermi.gbm.tte import GbmTte

DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
OUT_DIR = "/mnt/data/rgl36/FYP/results/coincidence_test"
PLOT_DIR = "/mnt/data/rgl36/FYP/results/plots"
ENERGY_KEV = (8.0, 900.0)
TICK_S = 2e-6                                       # GBM event-time tick, measured from the data
BKG_INTERVAL = (-60.0, -20.0)                       # background-only window for the duplicate rate
FINE_S = 1e-4
COARSE_S = 1e-2
RATIO = int(round(COARSE_S / FINE_S))
N_RATE_BINS = 12
N_NEIGHBOURS = 5                                    # rate label of a 10 ms bin = mean of 5 bins on each side
MIN_EVENTS = 1e4
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]

SETS = [
    ("GRB211211A", "1 det (n2)", ["n2"], (-0.9, 15.0)),
    ("GRB211211A", "3 det", ["n1", "n2", "na"], (-0.9, 15.0)),
    ("GRB211211A", "12 det", ALL_NAI, (-0.9, 15.0)),
    ("GRB230307A", "5 det", ["n0", "n1", "n6", "n7", "na"], (-1.6, 45.3)),
    ("GRB230307A", "7 det", ["n0", "n1", "n6", "n7", "n9", "na", "nb"], (-1.64, 18.0)),
    ("GRB230307A", "12 det", ALL_NAI, (-1.64, 18.0)),
]


def tteFile(grb, det):
    files = glob.glob(f"{DATA_ROOT}/{grb}/current/glg_tte_{det}_bn*_v0*.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    return files[0]


def inWindow(times, interval):
    return times[(times >= interval[0]) & (times < interval[1])]


def countLost(grb, dets, interval):
    """Events, observed deletions by the merge and predicted random deletions per 0.1 ms bin (all channels),
    the 8-900 keV fraction removed and the measured tick."""
    nCoarse = int((interval[1] - interval[0]) / COARSE_S)
    edges = interval[0] + FINE_S * np.arange(nCoarse * RATIO + 1)
    ttes, perDet, bandSum = [], [], 0
    for d in dets:
        tte = GbmTte.open(tteFile(grb, d))
        sliced = tte.slice_time(interval)
        tte.close()
        ttes.append(sliced)
        times = inWindow(np.array(sliced.data.times), interval)
        perDet.append(np.histogram(times, bins=edges)[0].astype(float))
        bandSum += inWindow(np.array(sliced.slice_energy(ENERGY_KEV).data.times), interval).size
    steps = np.diff(np.unique(np.concatenate([np.array(t.data.times) for t in ttes])))
    steps = steps[steps < 3e-6]                              # smallest steps (one tick)
    tickMeasured = float(np.mean(steps)) if steps.size else float("nan")
    merged = GbmTte.merge(ttes)
    mergedTimes = inWindow(np.array(merged.data.times), interval)
    mergedBand = inWindow(np.array(merged.slice_energy(ENERGY_KEV).data.times), interval).size
    mergedCounts = np.histogram(mergedTimes, bins=edges)[0].astype(float)

    perDet = np.array(perDet)                       # detectors x bins
    total = perDet.sum(axis=0)
    p = perDet * TICK_S / FINE_S                    # probability of an event per tick, per detector and fine bin
    sumP = p.sum(axis=0)
    keptFrac = np.where(sumP > 0, (1 - np.prod(1 - p, axis=0)) / np.where(sumP > 0, sumP, 1), 1.0)
    return dict(nCoarse=nCoarse, total=total, obsLost=total - mergedCounts, predRandom=total * (1 - keptFrac),
                fracBand=1 - mergedBand / bandSum, tick=tickMeasured)


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    report, plotData = {}, []
    allOk = True

    for grb, name, dets, interval in SETS:
        c = countLost(grb, dets, interval)
        total, obsLost, predRandom = c["total"], c["obsLost"], c["predRandom"]
        ticks = [c["tick"]]

        # duplicate deletions per second, measured in the background-only window for the same detectors
        if len(dets) > 1:
            b = countLost(grb, dets, BKG_INTERVAL)
            ticks.append(b["tick"])
            bDur = b["nCoarse"] * COARSE_S
            dupRate = (b["obsLost"].sum() - b["predRandom"].sum()) / bDur
            print(f"\n----- background window {list(BKG_INTERVAL)} s, {grb} {name}: events {int(b['total'].sum())}, "
                  f"deleted {int(b['obsLost'].sum())}, random prediction {b['predRandom'].sum():.0f}, "
                  f"duplicates {dupRate:.1f} deletions per second")
        else:
            dupRate = 0.0
        predDup = np.full(total.size, dupRate * FINE_S)
        predTotal = predRandom + predDup

        fracObs = obsLost.sum() / total.sum()
        fracRandom = predRandom.sum() / total.sum()
        fracPred = predTotal.sum() / total.sum()
        print(f"\n===== {grb}, {name}, {list(interval)} s =====")
        print(f"tick measured from the data: " + ", ".join(f"{t * 1e6:.4f}" for t in ticks)
              + f" microseconds (used: {TICK_S * 1e6:g})")
        print(f"events (all channels): {int(total.sum())}, after merge {int(total.sum() - obsLost.sum())}")
        print(f"fraction removed: observed {100 * fracObs:.3f}%, random only {100 * fracRandom:.3f}%, "
              f"random + duplicates {100 * fracPred:.3f}%   (8-900 keV light curve: {100 * c['fracBand']:.3f}%)")
        print("(events after the last full 10 ms bin of the interval are not used)")

        # observed and predicted fraction removed against total count rate (10 ms bins), in rate bins
        coarse = lambda a: a.reshape(c["nCoarse"], RATIO).sum(axis=1)
        total, obsLost, predRandom, predTotal = coarse(total), coarse(obsLost), coarse(predRandom), coarse(predTotal)
        # rate label of each 10 ms bin from its neighbours (N_NEIGHBOURS on each side, not the bin itself), so that a
        # bin's own Poisson fluctuation (which includes its duplicates) cannot decide which rate bin it falls in
        kernel = np.ones(2 * N_NEIGHBOURS + 1)
        kernel[N_NEIGHBOURS] = 0
        nSum = np.convolve(total, kernel, mode="same")
        nCount = np.convolve(np.ones(total.size), kernel, mode="same")
        rate = nSum / nCount / COARSE_S
        good = rate > 0
        rEdges = np.logspace(np.log10(rate[good].min()), np.log10(rate[good].max() * 1.0001), N_RATE_BINS + 1)
        rows = []
        print(f"{'rate (c/s)':>14s}{'events':>11s}{'obs %':>9s}{'random %':>10s}{'+dup %':>9s}{'obs/pred':>10s}"
              f"{'sigma':>8s}{'ok':>5s}")
        for lo, hi in zip(rEdges[:-1], rEdges[1:]):
            m = good & (rate >= lo) & (rate < hi)
            n = total[m].sum()
            if n < MIN_EVENTS:
                continue
            fo, fr, fp = obsLost[m].sum() / n, predRandom[m].sum() / n, predTotal[m].sum() / n
            sig = np.sqrt(max(fp * (1 - fp), 1e-12) / n)
            ok = abs(fo - fp) <= max(3 * sig, 0.05 * fp) if fp > 0 else obsLost[m].sum() == 0
            rows.append(dict(rateMid=float(np.sqrt(lo * hi)), events=int(n), obs=float(fo), predRandom=float(fr),
                             pred=float(fp), sigma=float(sig), ok=bool(ok)))
            print(f"{np.sqrt(lo * hi):>14.0f}{int(n):>11d}{100 * fo:>9.3f}{100 * fr:>10.3f}{100 * fp:>9.3f}"
                  f"{(fo / fp if fp > 0 else float('nan')):>10.3f}{((fo - fp) / sig if sig > 0 else 0):>8.1f}"
                  f"{'yes' if ok else 'NO':>5s}")

        if len(dets) == 1:
            # the tick criterion applies to sets of two or more detectors (as fixed before running)
            crit = {"one detector removes nothing": bool(obsLost.sum() == 0)}
        else:
            crit = {"tick within 1% of 2 microseconds": all(abs(t / TICK_S - 1) <= 1e-2 for t in ticks)}
            crit["total within 5% of random + duplicates"] = bool(abs(fracObs / fracPred - 1) <= 0.05)
            crit["every rate bin consistent"] = bool(all(rw["ok"] for rw in rows))
        for k, v in crit.items():
            print(f"  {k}: {'YES' if v else 'NO'}")
        allOk &= all(crit.values())
        report[f"{grb} {name}"] = dict(detectors=dets, interval_s=list(interval), tickUsed_us=TICK_S * 1e6,
                                      tickMeasured_us=[t * 1e6 for t in ticks], dupDeletionsPerSecond=dupRate,
                                      eventsAll=int(total.sum()), eventsMerged=int(total.sum() - obsLost.sum()),
                                      fracRemovedObs=float(fracObs), fracRemovedRandom=float(fracRandom),
                                      fracRemovedPred=float(fracPred), fracRemovedBand=float(c["fracBand"]),
                                      rateBins=rows, criteria=crit)
        plotData.append((grb, name, len(dets), rows))

    print(f"\n===== VERDICT (criteria set before running) =====")
    print("removed photons explained by random coincidences (2 microsecond tick) plus a steady rate of duplicates: "
          + ("YES" if allOk else "NOT in every case"))
    report["allCriteriaMet"] = bool(allOk)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Plot P7: fraction removed against total count rate; data points, random + duplicates (solid), random (dotted)
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"GRB230307A": "o", "GRB211211A": "s"}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (grb, name, k, rows) in enumerate(plotData):
        if not rows or k == 1:
            continue
        x = [rw["rateMid"] for rw in rows]
        ax.errorbar(x, [100 * rw["obs"] for rw in rows], yerr=[100 * rw["sigma"] for rw in rows],
                    fmt=markers[grb], color=colors[i % len(colors)], ms=5, label=f"{grb} {name} (observed)")
        ax.plot(x, [100 * rw["pred"] for rw in rows], "-", color=colors[i % len(colors)], lw=1)
        ax.plot(x, [100 * rw["predRandom"] for rw in rows], ":", color=colors[i % len(colors)], lw=1)
    ax.plot([], [], "k-", lw=1, label="random coincidences + duplicates")
    ax.plot([], [], "k:", lw=1, label="random coincidences only")
    ax.set_xscale("log")
    ax.set_xlabel("total count rate of the combined detectors (counts/s, all channels)")
    ax.set_ylabel("photons removed by GbmTte.merge (%)")
    ax.set_title("Photons removed by GbmTte.merge (force_unique=True)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"P7_merge_coincidences.{ext}"), dpi=150)
    print(f"\nSaved {OUT_DIR}/summary.json and {PLOT_DIR}/P7_merge_coincidences.png/.pdf  ({(time.time() - t0) / 60:.1f} min)")
