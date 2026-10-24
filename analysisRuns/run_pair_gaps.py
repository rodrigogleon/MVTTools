"""
Pair-gap test: how many of the photons removed by GbmTte.merge are genuine duplicates (one physical event recorded
by two detectors) and how many are random coincidences (two separate photons that happen to share a time tick)?

Idea. Event times are stored in ticks of length tau (about 1.9 microseconds). For every pair of detectors, count the
pairs of events (one in each detector) that are exactly g ticks apart, for g = -10 ... +10.
  - Random coincidences do not depend on the exact gap: over a few microseconds the count rate is constant, so the
    number of pairs is the same at every gap (a flat level).
  - Genuine duplicates are simultaneous: they add a bump at g = 0 only.
The flat level is measured from the gaps 3 <= |g| <= 10 (the "control gaps"); g = +-1 and +-2 are left out of the
control in case linked events are sometimes split across neighbouring ticks. The excess at g = 0 above the flat
level is the number of genuine duplicate pairs. GbmTte.merge deletes events only at g = 0.

For the pairs at g = 0 and at the control gaps, the energies of both events are compared (below 8 keV, 8-900 keV,
above 900 keV), and the excess is listed per detector pair, to help identify the cause (particles crossing several
detectors would favour high energies; a photon scattering from one detector into another would favour neighbours).

Sets: the burst intervals and detector sets of the coincidence test, plus a background-only interval with all 12
NaI detectors for each burst (no burst photons there, so any bump comes from the background).

Criteria fixed before running:
  1. method check: in every set, the counts at the control gaps are flat (chi-square test, p > 0.01)
  2. duplicates exist: in every burst set, the g = 0 count is above the flat level by more than 5 sigma
     (if it is within 3 sigma everywhere, all deletions are random coincidences)
  3. consistency: in every burst set, the excess pairs at g = 0 agree within 20% with the extra deletions found by
     the coincidence test (observed - predicted)
  4. background floor: in the background-only intervals the g = 0 excess is above 5 sigma, with a rate between
     200 and 800 per second (the low-rate floor read from P7 was about 400 per second for 12 detectors)
Also checked: the number of deletions computed here (events sharing a tick) matches the coincidence test within 0.5%.

Output: results/pair_gaps/summary.json and plot P7b in results/plots/

Usage
  python run_pair_gaps.py
"""
import os, json, glob, time
from itertools import combinations
import numpy as np
from scipy.stats import chi2

DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
OUT_DIR = "/mnt/data/rgl36/FYP/results/pair_gaps"
PLOT_DIR = "/mnt/data/rgl36/FYP/results/plots"
COINC_SUMMARY = "/mnt/data/rgl36/FYP/results/coincidence_test/summary.json"
ENERGY_KEV = (8.0, 900.0)
HIGH_KEV = (900.0, 1e5)
MAX_GAP = 10
CONTROL_MIN = 3
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]
CATS = ["low", "band", "high"]                      # below 8 keV, 8-900 keV, above 900 keV
COMBOS = [(a, b) for i, a in enumerate(CATS) for b in CATS[i:]]

# name in the coincidence-test summary (None for background-only sets), detectors, interval (s)
SETS = [
    ("GRB211211A", "3 det", ["n1", "n2", "na"], (-0.9, 15.0), True),
    ("GRB211211A", "12 det", ALL_NAI, (-0.9, 15.0), True),
    ("GRB211211A", "12 det, background only", ALL_NAI, (-60.0, -20.0), False),
    ("GRB230307A", "5 det", ["n0", "n1", "n6", "n7", "na"], (-1.6, 45.3), True),
    ("GRB230307A", "7 det", ["n0", "n1", "n6", "n7", "n9", "na", "nb"], (-1.64, 18.0), True),
    ("GRB230307A", "12 det", ALL_NAI, (-1.64, 18.0), True),
    ("GRB230307A", "12 det, background only", ALL_NAI, (-60.0, -20.0), False),
]


def tteFile(grb, det):
    files = glob.glob(f"{DATA_ROOT}/{grb}/current/glg_tte_{det}_bn*_v0*.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    return files[0]


def inWindow(times, interval):
    return times[(times >= interval[0]) & (times < interval[1])]


def tickLength(times):
    """Tick length: median of the smallest time steps between distinct events (robust to rounding of the times)."""
    d = np.diff(np.unique(times))
    return float(np.median(d[d < 1.5 * d.min()]))


def pairsAtGap(tA, tB, g, tau):
    """Pairs (one event in A, one in B) whose time difference tB - tA is g ticks, i.e. lies in [(g-0.5), (g+0.5)) tau.
    Events in one detector are more than a tick apart (dead time), so each window holds at most one event of B.
    Returns the indices of the paired events in A and in B. tA and tB must be sorted."""
    idx = np.searchsorted(tB, tA + (g - 0.5) * tau)
    ok = idx < tB.size
    ok[ok] = tB[idx[ok]] < tA[ok] + (g + 0.5) * tau
    return np.nonzero(ok)[0], idx[ok]


def gapCounts(tA, tB, maxGap, tau):
    """Number of pairs at each gap g = -maxGap ... maxGap (ticks)."""
    return np.array([pairsAtGap(tA, tB, g, tau)[0].size for g in range(-maxGap, maxGap + 1)])


def pairCategories(tA, cA, tB, cB, g, tau):
    """Energy categories of the pairs at gap g, counted per unordered combination (low-low, low-band, ...)."""
    iA, iB = pairsAtGap(tA, tB, g, tau)
    lo, hi = np.minimum(cA[iA], cB[iB]), np.maximum(cA[iA], cB[iB])
    return np.array([np.sum((lo == a) & (hi == b)) for a, b in
                     [(CATS.index(x), CATS.index(y)) for x, y in COMBOS]])


def flatLevel(counts, gaps):
    """Flat level from the control gaps, its chi-square p-value, and the g = 0 excess with its sigma."""
    ctrl = counts[np.abs(gaps) >= CONTROL_MIN]
    mean = ctrl.mean()
    pFlat = float(chi2.sf(np.sum((ctrl - mean) ** 2 / mean), ctrl.size - 1)) if mean > 0 else 1.0
    n0 = counts[gaps == 0][0]
    sigma = np.sqrt(n0 + mean / ctrl.size)
    return mean, pFlat, n0 - mean, (n0 - mean) / sigma if sigma > 0 else 0.0


if __name__ == "__main__":
    from gdt.missions.fermi.gbm.tte import GbmTte
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    coinc = json.load(open(COINC_SUMMARY))
    gaps = np.arange(-MAX_GAP, MAX_GAP + 1)
    ctrlMask = np.abs(gaps) >= CONTROL_MIN
    report, plotData, checks = {}, [], {1: [], 2: [], 3: [], 4: []}

    for grb, name, dets, interval, isBurst in SETS:
        # 1. event times, tick numbers and energy category for each detector
        raw = {}
        for d in dets:
            tte = GbmTte.open(tteFile(grb, d))
            sliced = tte.slice_time(interval)
            tte.close()
            raw[d] = dict(all=inWindow(np.array(sliced.data.times), interval),
                          band=inWindow(np.array(sliced.slice_energy(ENERGY_KEV).data.times), interval),
                          high=inWindow(np.array(sliced.slice_energy(HIGH_KEV).data.times), interval))
        allTimes = np.concatenate([r["all"] for r in raw.values()])
        tau = tickLength(allTimes)
        ev = {}
        for d, r in raw.items():
            t = np.sort(r["all"])
            if np.min(np.diff(t)) < 0.5 * tau:
                raise ValueError(f"{d}: two events less than half a tick apart inside one detector")
            cat = np.zeros(t.size, dtype=int)                              # low by default
            cat[np.isin(t, r["high"])] = CATS.index("high")
            cat[np.isin(t, r["band"])] = CATS.index("band")                # band wins at the 900 keV edge
            ev[d] = (t, cat)

        # 2. deletions by the merge = events with exactly the same time as an event of another detector (np.unique)
        _, mult = np.unique(allTimes, return_counts=True)
        deletions = int(np.sum(mult - 1))
        nEvents = int(allTimes.size)

        # 3. pair counts at each gap, summed over detector pairs; energies and per-pair excess
        total = np.zeros(gaps.size, dtype=np.int64)
        cat0 = np.zeros(len(COMBOS), dtype=np.int64)
        catCtrl = np.zeros(len(COMBOS), dtype=np.int64)
        perPair, identical = {}, 0
        for a, b in combinations(dets, 2):
            (tA, cA), (tB, cB) = ev[a], ev[b]
            c = gapCounts(tA, tB, MAX_GAP, tau)
            total += c
            identical += np.intersect1d(tA, tB).size                   # the merge's rule: exactly equal times
            perPair[f"{a}-{b}"] = float(c[gaps == 0][0] - c[ctrlMask].mean())
            cat0 += pairCategories(tA, cA, tB, cB, 0, tau)
            for g in gaps[ctrlMask]:
                catCtrl += pairCategories(tA, cA, tB, cB, int(g), tau)
        catCtrl = catCtrl / ctrlMask.sum()                     # average per control gap
        mean, pFlat, excess, nSigma = flatLevel(total, gaps)
        duration = interval[1] - interval[0]

        print(f"\n===== {grb}, {name}, {list(interval)} s, {len(dets)} detectors "
              f"({len(dets) * (len(dets) - 1) // 2} pairs), tick {tau * 1e6:.4f} microseconds "
              f"(2^-19 s = {2 ** -19 * 1e6:.4f}) =====")
        print(f"events {nEvents}, deleted by the merge {deletions} ({100 * deletions / nEvents:.3f}%)")
        print(f"check: gap-0 pairs {total[gaps == 0][0]} vs pairs with exactly equal times (merge rule) {identical}")
        print("pairs at each gap (ticks):")
        print("  " + " ".join(f"{g:>7d}" for g in gaps))
        print("  " + " ".join(f"{n:>7d}" for n in total))
        print(f"flat level (control gaps {CONTROL_MIN}-{MAX_GAP}): {mean:.1f}  flatness p = {pFlat:.3f}")
        print(f"gap +-1: {total[gaps == -1][0]}, {total[gaps == 1][0]}   gap +-2: {total[gaps == -2][0]}, {total[gaps == 2][0]}")
        print(f"gap 0: {total[gaps == 0][0]}  excess {excess:.0f} pairs ({nSigma:.1f} sigma), "
              f"{excess / duration:.0f} per second, {100 * excess / max(total[gaps == 0][0], 1):.1f}% of gap-0 pairs")

        res = dict(detectors=dets, interval_s=list(interval), tick_us=tau * 1e6, events=nEvents, deletions=deletions,
                   gaps=gaps.tolist(), pairCounts=total.tolist(), flatLevel=mean, flatnessP=pFlat,
                   excessPairs=excess, excessSigma=nSigma, excessPerSecond=excess / duration)
        checks[1].append(pFlat > 0.01)

        if isBurst:
            c = coinc[f"{grb} {name}"]
            obsDel = c["eventsAll"] - c["eventsMerged"]
            extra = (c["fracRemovedObs"] - c["fracRemovedPred"]) * c["eventsAll"]
            print(f"check: deletions here {deletions} vs coincidence test {obsDel} "
                  f"({100 * (deletions / obsDel - 1):+.2f}%)")
            print(f"excess pairs {excess:.0f} vs extra deletions in the coincidence test (obs - pred) {extra:.0f} "
                  f"(ratio {excess / extra:.2f})")
            print(f"share of the merge's deletions that are genuine duplicates: about {100 * excess / deletions:.1f}%")
            res.update(deletionsCoincTest=obsDel, extraDeletionsCoincTest=extra, ratioToCoincTest=excess / extra)
            checks[2].append(nSigma > 5)
            checks[3].append(abs(excess / extra - 1) <= 0.20)
        else:
            checks[4].append(nSigma > 5 and 200 <= excess / duration <= 800)

        # energies: share of each combination at gap 0, in the control gaps, and in the excess
        exc = cat0 - catCtrl
        print(f"{'energies (both events)':24s}{'gap 0 %':>9s}{'control %':>11s}{'excess %':>10s}")
        for i, (x, y) in enumerate(COMBOS):
            print(f"  {x + '-' + y:22s}{100 * cat0[i] / cat0.sum():>9.1f}{100 * catCtrl[i] / catCtrl.sum():>11.1f}"
                  f"{(100 * exc[i] / exc.sum() if exc.sum() > 0 else float('nan')):>10.1f}")
        top = sorted(perPair.items(), key=lambda kv: -kv[1])
        print("largest excess by detector pair: " + ", ".join(f"{p} {v:.0f}" for p, v in top[:6]))
        print("smallest: " + ", ".join(f"{p} {v:.0f}" for p, v in top[-3:]))
        res.update(energyCombos=[f"{x}-{y}" for x, y in COMBOS], energyGap0=cat0.tolist(),
                   energyControl=catCtrl.tolist(), excessPerPair=perPair)
        report[f"{grb} {name}"] = res
        plotData.append((f"{grb} {name}", gaps, total / mean if mean > 0 else total))

    names = {1: "control gaps flat in every set", 2: "duplicates exist (gap-0 excess > 5 sigma) in every burst set",
             3: "excess agrees with the coincidence test within 20% in every burst set",
             4: "background-only intervals: excess > 5 sigma at 200-800 per second"}
    print("\n===== CRITERIA (set before running) =====")
    for i in range(1, 5):
        print(f"{i}. {names[i]}: {'YES' if all(checks[i]) else 'NO'}  ({sum(checks[i])}/{len(checks[i])})")
    report["criteria"] = {names[i]: bool(all(checks[i])) for i in range(1, 5)}
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    # Plot P7b: pair counts against gap, divided by the flat level (1 = random coincidences only)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, g, y in plotData:
        ax.plot(g, y, "o-", ms=4, lw=1, label=label)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.axvspan(-CONTROL_MIN + 0.5, CONTROL_MIN - 0.5, color="0.9", zorder=0)
    ax.set_xlabel("gap between the two events (ticks of 1.9 microseconds)")
    ax.set_ylabel("pairs / flat level")
    ax.set_title("Pairs of events in two detectors against their time gap")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"P7b_pair_gaps.{ext}"), dpi=150)
    print(f"\nSaved {OUT_DIR}/summary.json and {PLOT_DIR}/P7b_pair_gaps.png/.pdf  ({(time.time() - t0) / 60:.1f} min)")
