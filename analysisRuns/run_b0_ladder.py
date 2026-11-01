"""
B0: how much do the injection-recovery MVTs change when the timescale ladder is made finer?

Main setting (unchanged everywhere else): outputBinningRatio = 4, i.e. 4 rungs per doubling (steps of about x1.19),
as in Bala et al. and their validation curve. Here the same simulations are analysed with outputBinningRatio = 8
(8 rungs per doubling, steps of about x1.09). This is a built-in option of haarPowerMod, passed as an argument; no
MVTTools code is changed. The option also sets the other steps tied to the ladder (fitting window of
outputBinningRatio + 1 rungs, departure threshold, uncertainty scaling), so they keep the same span in time.

Same photons. Every simulated observation is made with exactly the seeds of run_injection_recovery.py
(SeedSequence([2026, pulse index, amplitude index]).spawn(300)), so the photons are identical to the saved
runs and the only difference is the ladder. Before each setting, the first 2 realisations are re-analysed with
outputBinningRatio = 4 and must reproduce the saved values (relative 1e-9), or the script stops.

Settings: all 7 pulses x amplitudes 3, 30, 300 and the bright run 3000 x bin widths 0.1, 0.01, 0.003 ms,
300 realisations each (84 pulse-amplitude-bin-width settings). Output, one JSON per pulse and amplitude, so an
interrupted run resumes: results/injection_recovery/b0_ratio8/settings/.

Analysis (--post, also run at the end)
  Per setting: median, 16-84%, valid % with 4 and with 8; change of the median (8 / 4 - 1); for realisations
  valid with both, the ratio MVT(8) / MVT(4) of each realisation (median and 16-84%).
  True MVT with 8 (same rule as before: anchor at the smallest bin width, 10% gate) against the true MVT with 4.
  Ladder check: position of the 8-rung values between the 8-rung ladder (as A4) and KS test against uniform.
  Plots: results/plots/B0_medians.png and B0_hist_bright.png (.pdf too).

Decision rules (fixed before running)
  D1 (the main question). Over settings with at least 30 valid values with both settings and not bin-limited
     (bin width <= BW* of the ratio-4 truth), and amplitude >= 300 (bright: where the steps dominate):
     |change of median| = |median(8) / median(4) - 1|.
     SMALL if the median of |change| <= 5% and the largest <= 10%: report the method's step resolution only.
     Otherwise LARGE: quote the change as a systematic uncertainty of the MVT (in the report and the notes).
     The same numbers are also reported for amplitudes 3 and 30 (not part of the decision).
  D2 (mechanism). The ratio-8 values are tied to the ratio-8 ladder (KS p < 0.01 in more than half of the
     settings with at least 30 valid values) -> the stepping comes from the ladder (confirmed); otherwise not.
  D3 (reported, no pass/fail). True MVT per pulse with 8 against 4, and whether the change between 0.01 and
     0.003 ms bins in G1, G3, N3_10 and N10_30 is still there with 8.

Usage (from the scripts folder, next to run_injection_recovery.py and analyse_rung_positions.py)
  python run_b0_ladder.py                 simulations, then the analysis
  python run_b0_ladder.py --post          only the analysis and plots from saved results
"""
import os, sys, json, time, argparse, logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import run_injection_recovery as ir

RATIO_NEW = 8
RATIO_OLD = 4
B0_AMPS = [3.0, 30.0, 300.0, ir.BRIGHT_AMPLITUDE]
ALL_AMPS = list(ir.AMPLITUDES) + [ir.BRIGHT_AMPLITUDE]          # same order as runGrid, for the seeds
OUT = os.path.join(ir.OUT_DIR, "b0_ratio8")
MIN_VALID = 30
hm = None


def loadHaar():
    sys.path.insert(0, ir.REPO)
    here = os.getcwd()
    os.chdir(ir.REPO)
    from loggerSetup import initialiseLogging
    initialiseLogging()
    logging.disable(logging.CRITICAL)
    import analysisTools.haarMethods as m
    os.chdir(here)
    return m


def analyse(counts, bw, ratio):
    try:
        r = hm.haarPowerMod(counts, np.sqrt(counts), minimumBinSizeSeconds=bw, outputBinningRatio=ratio)
        return float(r[2]), float(r[3])
    except Exception:
        return float("nan"), float("nan")


def _task(args):
    """Worker: one realisation (same photons as the saved run), MVT and uncertainty (s) at every bin width."""
    global hm
    if hm is None:
        hm = loadHaar()
    key, amp, seed, ratio = args
    tstart, tstop = ir.window(ir.PULSES[key])
    times, _ = ir.simulate(key, amp, seed)
    return [analyse(ir.lightCurve(times, tstart, tstop, bw), bw, ratio) for bw in ir.BIN_WIDTHS]


def seedsFor(key, amp):
    pi, ai = list(ir.PULSES).index(key), ALL_AMPS.index(amp)
    return np.random.SeedSequence([ir.SEED, pi, ai]).spawn(ir.N_REAL)


def savedPath(key, amp):
    return os.path.join(ir.OUT_DIR, "settings", f"{key}_{ir.ampTag(amp)}.json")


def newPath(key, amp):
    return os.path.join(OUT, "settings", f"{key}_{ir.ampTag(amp)}.json")


def samePhotonsCheck(key, amp, seeds):
    """Re-analyse the first 2 realisations with the old ladder; they must reproduce the saved values."""
    saved = json.load(open(savedPath(key, amp)))
    for j in range(2):
        again = _task((key, amp, seeds[j], RATIO_OLD))
        for b, (mvt, _) in enumerate(again):
            old = saved["bins"][b]["mvt_ms"][j] / 1e3
            same = (np.isnan(mvt) and np.isnan(old)) or abs(mvt - old) <= 1e-9 * max(abs(old), 1e-30)
            if not same:
                sys.exit(f"STOP: {key} amp {amp:g} realisation {j} bin {ir.BIN_WIDTHS[b] * 1e3:g} ms: "
                         f"recomputed {mvt} s, saved {old} s (photons or code differ from the saved run)")


def run(workers):
    os.makedirs(os.path.join(OUT, "settings"), exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key in ir.PULSES:
            for amp in B0_AMPS:
                path = newPath(key, amp)
                if not os.path.exists(savedPath(key, amp)):
                    print(f"WARNING: {key} amplitude {amp:g}: saved ratio-4 run not found, skipped")
                    continue
                if os.path.exists(path):
                    print(f"{key:8s} amplitude {amp:>6g}: already done, skipped")
                    continue
                t0 = time.time()
                seeds = seedsFor(key, amp)
                samePhotonsCheck(key, amp, seeds)
                res = list(pool.map(_task, [(key, amp, s, RATIO_NEW) for s in seeds], chunksize=4))
                bins = []
                for b, bw in enumerate(ir.BIN_WIDTHS):
                    mvt = [r[b][0] for r in res]
                    unc = [r[b][1] for r in res]
                    bins.append(dict(binWidth_s=bw, mvt_ms=[v * 1e3 for v in mvt], unc_ms=[v * 1e3 for v in unc],
                                     **ir.summarise(mvt, unc)))
                with open(path, "w") as f:
                    json.dump(dict(pulse=key, amplitude=amp, outputBinningRatio=RATIO_NEW, nRealisations=ir.N_REAL,
                                   seed=[ir.SEED, list(ir.PULSES).index(key), ALL_AMPS.index(amp)], bins=bins),
                              f, default=float)
                line = "  ".join(f"{p['binWidth_s'] * 1e3:g}ms: {p['median']:.3g} ms ({p['successPct']:.0f}% valid)"
                                 for p in bins)
                print(f"{key:8s} amplitude {amp:>6g}: {line}  [same photons checked; {(time.time() - t0) / 60:.1f} min]",
                      flush=True)


def validMask(mvt, unc):
    mvt, unc = np.asarray(mvt, float), np.asarray(unc, float)
    return np.isfinite(mvt) & (mvt > 0) & (unc > 0)


def truth(recs):
    out = {}
    for key in ir.PULSES:
        b = recs.get((key, ir.BRIGHT_AMPLITUDE))
        if b is None:
            continue
        meds = {p["binWidth_s"]: p["median"] for p in b["bins"]}
        anchor = meds[min(ir.BIN_WIDTHS)]
        if not np.isfinite(anchor):
            continue
        ok = [bw for bw in ir.BIN_WIDTHS if np.isfinite(meds[bw]) and abs(meds[bw] - anchor) / anchor <= ir.BIAS_GATE]
        bwStar = max(ok) if ok else min(ir.BIN_WIDTHS)
        out[key] = dict(tTrue=float(np.mean([meds[bw] for bw in ir.BIN_WIDTHS if bw <= bwStar])), bwStar=bwStar,
                        meds=meds)
    return out


def post():
    from scipy.stats import kstest
    import analyse_rung_positions as A
    old, new = {}, {}
    for key in ir.PULSES:
        for amp in ALL_AMPS:
            if os.path.exists(savedPath(key, amp)):
                old[(key, amp)] = json.load(open(savedPath(key, amp)))
            if os.path.exists(newPath(key, amp)):
                new[(key, amp)] = json.load(open(newPath(key, amp)))
    missing = [(k, a) for k in ir.PULSES for a in B0_AMPS if (k, a) not in new]
    if missing:
        print(f"WARNING: {len(missing)} B0 settings missing: {missing}")
    t4, t8 = truth(old), truth(new)

    # ladder of the new setting, checked against haarNDWT itself
    def rungs8(n, bw):
        return A.rungsFast(n, bw, binFactor=RATIO_NEW, oversample=RATIO_NEW * 8.0)
    one = np.ones(15000)
    lo, up, _, _, _ = hm.haarNDWT(one, one, one, binSizeInSeconds=1e-5, timeSeriesOversampleFactor=RATIO_NEW * 8.0,
                                  totalSignalRepetitions=2, binFactor=RATIO_NEW)
    if not np.allclose(np.unique(0.5 * (lo + up) * 1e3), rungs8(15000, 1e-5), rtol=1e-12, atol=0):
        sys.exit("STOP: 8-rung ladder reconstruction differs from haarNDWT")
    print("self-check: 8-rung ladder reconstruction identical to haarNDWT")

    rows = []
    print("\n===== B0. MEDIANS WITH 4 AND 8 RUNGS PER DOUBLING (same photons) =====")
    print(f"{'pulse':8s}{'amp':>6s}{'BW ms':>7s}{'binlim':>7s}{'median 4':>10s}{'16-84% (4)':>15s}{'valid4':>7s}"
          f"{'median 8':>10s}{'16-84% (8)':>15s}{'valid8':>7s}{'change':>8s}{'pair ratio 8/4 (16-50-84%)':>29s}"
          f"{'KS p (8)':>10s}")
    for (key, amp), rn in new.items():
        ro = old.get((key, amp))
        if ro is None or key not in t4:
            continue
        tstart, tstop = ir.window(ir.PULSES[key])
        for pn, po in zip(rn["bins"], ro["bins"]):
            bw = pn["binWidth_s"]
            vo, vn = validMask(po["mvt_ms"], po["unc_ms"]), validMask(pn["mvt_ms"], pn["unc_ms"])
            both = vo & vn
            ratio = np.asarray(pn["mvt_ms"], float)[both] / np.asarray(po["mvt_ms"], float)[both]
            q = np.percentile(ratio, [16, 50, 84]) if ratio.size >= 2 else [np.nan] * 3
            R = rungs8(ir.lightCurve(np.array([]), tstart, tstop, bw).size, bw)
            u = A.positions(np.asarray(pn["mvt_ms"], float)[vn], R)
            ks = float(kstest(u, "uniform").pvalue) if u.size >= 2 else float("nan")
            change = pn["median"] / po["median"] - 1 if (np.isfinite(pn["median"]) and np.isfinite(po["median"])) else np.nan
            bl = bw > t4[key]["bwStar"]
            row = dict(pulse=key, amplitude=amp, binWidth_ms=bw * 1e3, binLimited=bl,
                       median4=po["median"], p16_4=po["p16"], p84_4=po["p84"], valid4=int(vo.sum()),
                       median8=pn["median"], p16_8=pn["p16"], p84_8=pn["p84"], valid8=int(vn.sum()),
                       nBoth=int(both.sum()), change=change, pairRatio16=q[0], pairRatio50=q[1], pairRatio84=q[2],
                       ksP8=ks, nLadder8=int(u.size))
            rows.append(row)
            r4 = f"{po['p16']:.3g}-{po['p84']:.3g}"
            r8 = f"{pn['p16']:.3g}-{pn['p84']:.3g}"
            pr = f"{q[0]:.3f} {q[1]:.3f} {q[2]:.3f}"
            print(f"{key:8s}{amp:>6g}{bw * 1e3:>7g}{('yes' if bl else ''):>7s}{po['median']:>10.4g}{r4:>15s}"
                  f"{int(vo.sum()):>7d}{pn['median']:>10.4g}{r8:>15s}{int(vn.sum()):>7d}{100 * change:>+7.1f}%"
                  f"{pr:>29s}{ks:>10.2g}")

    def judged(sub):
        return [x for x in sub if x["valid4"] >= MIN_VALID and x["valid8"] >= MIN_VALID and not x["binLimited"]
                and np.isfinite(x["change"])]

    bright = judged([x for x in rows if x["amplitude"] >= 300])
    print("\n===== D1. SIZE OF THE CHANGE (not bin-limited, >= 30 valid with both) =====")
    d1 = {}
    for name, sub in (("bright (amp >= 300)", bright), ("amp 30", judged([x for x in rows if x["amplitude"] == 30])),
                      ("amp 3", judged([x for x in rows if x["amplitude"] == 3]))):
        if not sub:
            print(f"  {name}: no settings")
            continue
        a = np.abs([x["change"] for x in sub])
        d1[name] = dict(n=len(sub), medianAbsChange=float(np.median(a)), maxAbsChange=float(a.max()),
                        meanSignedChange=float(np.mean([x["change"] for x in sub])))
        print(f"  {name}: {len(sub)} settings; |change| median {100 * np.median(a):.1f}%, largest {100 * a.max():.1f}%, "
              f"mean signed change {100 * np.mean([x['change'] for x in sub]):+.1f}%")
    if "bright (amp >= 300)" in d1:
        b = d1["bright (amp >= 300)"]
        verdict = "SMALL" if (b["medianAbsChange"] <= 0.05 and b["maxAbsChange"] <= 0.10) else "LARGE"
    else:
        verdict = "NOT JUDGED"
    print(f"D1 verdict (bright settings): {verdict}"
          + ("  -> report the method's step resolution only" if verdict == "SMALL" else
             "  -> quote the change as a systematic uncertainty" if verdict == "LARGE" else ""))

    kj = [x for x in rows if x["nLadder8"] >= MIN_VALID and np.isfinite(x["ksP8"])]
    tied = sum(x["ksP8"] < 0.01 for x in kj)
    d2 = "CONFIRMED" if kj and tied > len(kj) / 2 else ("NOT CONFIRMED" if kj else "NOT JUDGED")
    print(f"\n===== D2. 8-RUNG VALUES TIED TO THE 8-RUNG LADDER: {tied}/{len(kj)} -> mechanism {d2} =====")

    print("\n===== D3. TRUE MVT (bright run, same rule) WITH 4 AND 8 RUNGS PER DOUBLING =====")
    print(f"{'pulse':8s}{'medians 4: 0.1 / 0.01 / 0.003 ms':>36s}{'true 4':>9s}{'medians 8: 0.1 / 0.01 / 0.003 ms':>36s}"
          f"{'true 8':>9s}{'change':>8s}")
    d3 = {}
    for key in ir.PULSES:
        if key not in t4 or key not in t8:
            continue
        m4 = " / ".join(f"{t4[key]['meds'][bw]:.3f}" for bw in ir.BIN_WIDTHS)
        m8 = " / ".join(f"{t8[key]['meds'][bw]:.3f}" for bw in ir.BIN_WIDTHS)
        ch = t8[key]["tTrue"] / t4[key]["tTrue"] - 1
        d3[key] = dict(true4=t4[key]["tTrue"], true8=t8[key]["tTrue"], bwStar4_ms=t4[key]["bwStar"] * 1e3,
                       bwStar8_ms=t8[key]["bwStar"] * 1e3, change=ch,
                       step001to0003_4=t4[key]["meds"][3e-6] / t4[key]["meds"][1e-5] - 1,
                       step001to0003_8=t8[key]["meds"][3e-6] / t8[key]["meds"][1e-5] - 1)
        print(f"{key:8s}{m4:>36s}{t4[key]['tTrue']:>9.3f}{m8:>36s}{t8[key]['tTrue']:>9.3f}{100 * ch:>+7.1f}%")
    print("change of the bright median from 0.01 to 0.003 ms bins (4 -> 8): " + ", ".join(
        f"{k} {100 * v['step001to0003_4']:+.0f}% -> {100 * v['step001to0003_8']:+.0f}%" for k, v in d3.items()))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(dict(D1=d1, D1verdict=verdict, D2=dict(tied=tied, judged=len(kj), verdict=d2), D3=d3, rows=rows),
                  f, indent=1, default=float)
    print(f"\nSaved {OUT}/summary.json")
    plots(rows, old, new)


def plots(rows, old, new):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(ir.PLOT_DIR, exist_ok=True)
    keys = list(ir.PULSES)

    # medians against amplitude, 4 (open) and 8 (filled), one panel per pulse
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, key in zip(axes.ravel(), keys):
        for c, bw in enumerate(ir.BIN_WIDTHS):
            sub = sorted([x for x in rows if x["pulse"] == key and abs(x["binWidth_ms"] - bw * 1e3) < 1e-12],
                         key=lambda x: x["amplitude"])
            if not sub:
                continue
            a = np.array([x["amplitude"] for x in sub])
            for tag, sh, mfc in (("4", 0.94, "none"), ("8", 1.06, f"C{c}")):
                m = np.array([x[f"median{tag}"] for x in sub])
                lo = m - np.array([x[f"p16_{tag}"] for x in sub])
                hi = np.array([x[f"p84_{tag}"] for x in sub]) - m
                ax.errorbar(a * sh * (1 + 0.03 * (c - 1)), m, yerr=[lo, hi], fmt="o", color=f"C{c}", mfc=mfc, ms=5,
                            capsize=2, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("amplitude"); ax.set_ylabel("median MVT (ms)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
    ax = axes.ravel()[-1]
    for c, bw in enumerate(ir.BIN_WIDTHS):
        ax.plot([], [], "o", color=f"C{c}", label=f"{bw * 1e3:g} ms bins")
    ax.plot([], [], "o", color="0.3", mfc="none", label="4 rungs per doubling (main)")
    ax.plot([], [], "o", color="0.3", label="8 rungs per doubling (B0)")
    ax.legend(fontsize=9, loc="center"); ax.set_axis_off()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ir.PLOT_DIR, f"B0_medians.{ext}"), dpi=130)
    plt.close(fig)

    # bright-run histograms, 4 against 8
    fig, axes = plt.subplots(len(keys), len(ir.BIN_WIDTHS), figsize=(13, 2.3 * len(keys)))
    for i, key in enumerate(keys):
        for j, bw in enumerate(ir.BIN_WIDTHS):
            ax = axes[i, j]
            ro, rn = old.get((key, ir.BRIGHT_AMPLITUDE)), new.get((key, ir.BRIGHT_AMPLITUDE))
            if ro is None or rn is None:
                ax.set_axis_off(); continue
            po, pn = ro["bins"][j], rn["bins"][j]
            vo = np.asarray(po["mvt_ms"], float)[validMask(po["mvt_ms"], po["unc_ms"])]
            vn = np.asarray(pn["mvt_ms"], float)[validMask(pn["mvt_ms"], pn["unc_ms"])]
            allv = np.concatenate([vo, vn])
            if allv.size >= 2:
                edges = np.linspace(allv.min() * 0.98, allv.max() * 1.02, 100)
                ax.hist(vo, bins=edges, histtype="step", color="k", lw=1.2, label="4 per doubling")
                ax.hist(vn, bins=edges, color="C1", alpha=0.6, label="8 per doubling")
            ax.set_title(f"{key}, bright, {bw * 1e3:g} ms", fontsize=8)
    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
            break
    axes[-1, 1].set_xlabel("MVT of each valid realisation (ms)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ir.PLOT_DIR, f"B0_hist_bright.{ext}"), dpi=130)
    plt.close(fig)
    print(f"Saved plots: {ir.PLOT_DIR}/B0_medians.png, B0_hist_bright.png (.pdf too)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true", help="only the analysis and plots from saved results")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    hm = loadHaar()
    if not args.post:
        commit = os.popen(f"git -C {ir.REPO} log -1 --format='%h %s'").read().strip()
        print(f"MVTTools code in use: {commit}", flush=True)
        t0 = time.time()
        run(args.workers)
        print(f"Simulations finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    post()
