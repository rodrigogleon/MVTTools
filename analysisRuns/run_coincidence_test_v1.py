"""
Are the photons removed by GbmTte.merge random coincidences of separate photons in different detectors?

GbmTte.merge with its default force_unique=True keeps one event per distinct arrival time. Times are stored in
ticks of length tau (about 1.9 microseconds), and a detector cannot record two events in the same tick (its dead
time is longer than a tick), so an event is removed only when another detector has an event in the same tick.

Prediction for random (independent) arrivals: if detector i has probability p_i of an event in a given tick,
the expected number of events per tick is sum(p_i) before the merge and 1 - prod(1 - p_i) after it, so the
fraction removed is
    f = 1 - (1 - prod(1 - p_i)) / sum(p_i)
which is exactly 0 for one detector and about (k - 1) p / 2 for k detectors with equal small p.
p_i is taken from the detector's own counts in each 0.1 ms bin: p_i = counts_i * tau / 0.1 ms. The bins must be
much shorter than the burst variability, because coincidences grow with the square of the rate: averaging the
rate over longer bins would underestimate them. Given the counts in a short bin, the prediction is exact for
events spread uniformly inside it. The rate on the x axis of the plot is the total rate in 10 ms bins.

The test uses all energy channels (the merge acts before any energy cut) over the source interval,
for each detector set of Bala et al.'s Table 3. It also reports the fraction removed from the 8-900 keV light curve.

Criteria fixed before running:
  1. one detector: no event removed
  2. for every detector set, the total fraction removed agrees with the prediction within 5% (relative)
  3. in every count-rate bin with at least 10^4 events, observed and predicted agree within 3 sigma (binomial)
     or within 5% (relative), whichever is larger, with no systematic trend with rate

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
FINE_S = 1e-4
COARSE_S = 1e-2
RATIO = int(round(COARSE_S / FINE_S))
N_RATE_BINS = 12
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


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    report, plotData = {}, []
    allOk = True

    for grb, name, dets, interval in SETS:
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
        allTimes = np.sort(np.concatenate([np.array(t.data.times) for t in ttes]))
        tau = float(np.min(np.diff(np.unique(allTimes))))
        merged = GbmTte.merge(ttes)
        mergedTimes = inWindow(np.array(merged.data.times), interval)
        mergedBand = inWindow(np.array(merged.slice_energy(ENERGY_KEV).data.times), interval).size
        mergedCounts = np.histogram(mergedTimes, bins=edges)[0].astype(float)

        perDet = np.array(perDet)                       # detectors x bins
        total = perDet.sum(axis=0)
        p = perDet / (FINE_S / tau)                     # probability of an event per tick, per detector and fine bin
        sumP = p.sum(axis=0)
        keptFrac = np.where(sumP > 0, (1 - np.prod(1 - p, axis=0)) / np.where(sumP > 0, sumP, 1), 1.0)
        predLost = total * (1 - keptFrac)               # expected removed events per bin
        obsLost = total - mergedCounts

        fracObs = obsLost.sum() / total.sum()
        fracPred = predLost.sum() / total.sum()
        fracBand = 1 - mergedBand / bandSum
        print(f"\n===== {grb}, {name}, {list(interval)} s, tick {tau * 1e6:.3f} microseconds =====")
        print(f"events (all channels): {int(total.sum())}, after merge {int(mergedCounts.sum())}")
        print(f"fraction removed: observed {100 * fracObs:.3f}%, predicted {100 * fracPred:.3f}%   "
              f"(8-900 keV light curve: {100 * fracBand:.3f}%)")
        print("(events after the last full 10 ms bin of the interval are not used)")

        # observed and predicted fraction removed against total count rate (10 ms bins), in rate bins
        coarse = lambda a: a.reshape(nCoarse, RATIO).sum(axis=1)
        total, obsLost, predLost = coarse(total), coarse(obsLost), coarse(predLost)
        rate = total / COARSE_S
        good = rate > 0
        rEdges = np.logspace(np.log10(rate[good].min()), np.log10(rate[good].max() * 1.0001), N_RATE_BINS + 1)
        rows = []
        print(f"{'rate (c/s)':>14s}{'events':>11s}{'obs %':>9s}{'pred %':>9s}{'obs/pred':>10s}{'sigma':>8s}{'ok':>5s}")
        for lo, hi in zip(rEdges[:-1], rEdges[1:]):
            m = good & (rate >= lo) & (rate < hi)
            n = total[m].sum()
            if n < MIN_EVENTS:
                continue
            fo, fp = obsLost[m].sum() / n, predLost[m].sum() / n
            sig = np.sqrt(max(fp * (1 - fp), 1e-12) / n)
            ok = abs(fo - fp) <= max(3 * sig, 0.05 * fp) if fp > 0 else obsLost[m].sum() == 0
            rows.append(dict(rateMid=float(np.sqrt(lo * hi)), events=int(n), obs=float(fo), pred=float(fp),
                             sigma=float(sig), ok=bool(ok)))
            print(f"{np.sqrt(lo * hi):>14.0f}{int(n):>11d}{100 * fo:>9.3f}{100 * fp:>9.3f}"
                  f"{(fo / fp if fp > 0 else float('nan')):>10.3f}{((fo - fp) / sig if sig > 0 else 0):>8.1f}{'yes' if ok else 'NO':>5s}")

        if len(dets) == 1:
            c1 = obsLost.sum() == 0
            crit = {"one detector removes nothing": bool(c1)}
        else:
            crit = {"total within 5% of prediction": bool(abs(fracObs / fracPred - 1) <= 0.05),
                    "every rate bin consistent": bool(all(rw["ok"] for rw in rows))}
        for k, v in crit.items():
            print(f"  {k}: {'YES' if v else 'NO'}")
        allOk &= all(crit.values())
        report[f"{grb} {name}"] = dict(detectors=dets, interval_s=list(interval), tick_us=tau * 1e6,
                                      eventsAll=int(total.sum()), eventsMerged=int(mergedCounts.sum()),
                                      fracRemovedObs=float(fracObs), fracRemovedPred=float(fracPred),
                                      fracRemovedBand=float(fracBand), rateBins=rows, criteria=crit)
        plotData.append((grb, name, len(dets), rows))

    print(f"\n===== VERDICT (criteria set before running) =====")
    print("removed photons behave as random coincidences of separate photons: " + ("YES" if allOk else "NOT in every case"))
    report["allCriteriaMet"] = bool(allOk)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Plot P7: fraction removed against total count rate, data points and prediction
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
    ax.plot([], [], "k-", lw=1, label="random-coincidence prediction")
    ax.set_xscale("log")
    ax.set_xlabel("total count rate of the combined detectors (counts/s, all channels)")
    ax.set_ylabel("photons removed by GbmTte.merge (%)")
    ax.set_title("Photons removed by GbmTte.merge (force_unique=True)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"P7_merge_coincidences.{ext}"), dpi=150)
    print(f"\nSaved {OUT_DIR}/summary.json and {PLOT_DIR}/P7_merge_coincidences.png/.pdf  ({(time.time() - t0) / 60:.1f} min)")
