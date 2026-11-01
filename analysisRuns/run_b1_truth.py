"""
B1: settle the true MVT of the narrow pulses with smaller bins, then recompute criteria 1-4 for them.

Why. In the injection run the bright-run (amplitude 3000) median MVT of G1, G3, N3_10 and N10_30 was still falling
between 0.01 and 0.003 ms bins (-28%, -16%, -16%, -8%), also with 8 rungs per doubling (B0), so their true MVT and
everything judged against it (errors, bin-limited flags, criteria 1-3) is uncertain. Bala et al. went down to
0.0001 ms in their simulations. This also tells us what bin width a 1-2 ms MVT needs (real GBM data cannot go below
the 2 microsecond tick, i.e. 0.002 ms).

Stage 1 (bright runs, new bin widths)
  Narrow pulses G1, G3, N3_10, N10_30: amplitude 3000 at 0.001, 0.0003 and 0.0001 ms.
  Wide pulses G10, G30, N30_100: amplitude 3000 at 0.001 ms (check of a second flat step).
  Same seeds as run_injection_recovery.py, so the same photons as the saved runs; the first 2 realisations are
  re-analysed at 0.003 ms and must reproduce the saved values (relative 1e-9) or the script stops.
  Main ladder setting (outputBinningRatio 4). Norris photons keep the 0.3 microsecond time grid of the saved runs
  (so the photons stay identical): within 0.3 microseconds the Norris rate changes by less than 0.1%, while the
  Poisson noise in a 0.1 microsecond bin is of order 100%, so the grid has no measurable effect.

Truth (T1, T2), from the saved bin widths plus the new ones
  Eligible bin widths: at least 50% valid realisations (our rule: below that the median rests on too few values).
  T1 (Bala et al.'s rule, main): anchor = median at the smallest eligible bin width; BW* = largest eligible bin
     width whose median is within 10% of the anchor; true MVT = mean of the medians at eligible bin widths <= BW*.
  T2 (settled, our check): the last two steps between the three smallest eligible bin widths each change the
     median by <= 10%. Otherwise the pulse is UNSETTLED: its true MVT is quoted as an upper bound and it is left
     out of stage 2 and of the recomputed criteria.

Stage 2 (test grid at the new bin widths, settled narrow pulses only)
  All 7 test amplitudes at the new bin widths that are eligible in the bright run and <= BW*; same seeds as the
  saved test grid (same-photons check as above); MVT and SNR_MVT computed exactly as in run_injection_recovery.py.

Stage 3 (analysis)
  Bright runs: median, bootstrap 16-84% of the median (2000 resamples), valid / limit / failed, peak counts per bin,
     share of values in the most populated rung gap (4-rung ladder), for every bin width (saved and new) of every
     pulse.
  T1, T2 per pulse, against the original truth.
  Criteria 1-3 recomputed for the settled narrow pulses with the new truth and BW* (definitions unchanged), using
     the saved test settings (0.1, 0.01, 0.003 ms) and the new ones; criterion 1 also against the bright median at
     the same bin width (post hoc, as A3). Criterion 4 (true MVT / sigma) for the Gaussians.
  Smallest bin width each pulse needs to settle, against the 0.002 ms real-data limit.
  Plots: results/plots/B1_bright_medians.png, B1_hist_bright.png (.pdf too).

Output: results/injection_recovery/b1_truth/ (bright/, test/, summary.json). Resumable.

Usage (from the scripts folder, next to run_injection_recovery.py and analyse_rung_positions.py)
  python run_b1_truth.py                  stages 1, 2 and 3
  python run_b1_truth.py --stage 1        bright runs only (then the analysis of what exists)
  python run_b1_truth.py --stage 3        analysis and plots only
"""
import os, sys, json, time, argparse, logging, io, contextlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import run_injection_recovery as ir

NARROW = ["G1", "G3", "N3_10", "N10_30"]
WIDE = ["G10", "G30", "N30_100"]
NEW_BW = {**{k: [1e-6, 3e-7, 1e-7] for k in NARROW}, **{k: [1e-6] for k in WIDE}}
ALL_AMPS = list(ir.AMPLITUDES) + [ir.BRIGHT_AMPLITUDE]          # same order as runGrid, for the seeds
MIN_VALID_FRAC = 0.5
GATE = ir.BIAS_GATE                                             # 10%
N_BOOT = 2000
OUT = os.path.join(ir.OUT_DIR, "b1_truth")
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


def _task(args):
    """Worker: one realisation (same photons as the saved run), MVT and uncertainty (s) at the given bin widths."""
    global hm
    if hm is None:
        hm = loadHaar()
    key, amp, seed, bws = args
    tstart, tstop = ir.window(ir.PULSES[key])
    times, _ = ir.simulate(key, amp, seed)
    out = []
    for bw in bws:
        c = ir.lightCurve(times, tstart, tstop, bw)
        try:
            r = hm.haarPowerMod(c, np.sqrt(c), minimumBinSizeSeconds=bw)
            out.append((float(r[2]), float(r[3])))
        except Exception:
            out.append((float("nan"), float("nan")))
    return out


def seedsFor(key, amp):
    return np.random.SeedSequence([ir.SEED, list(ir.PULSES).index(key), ALL_AMPS.index(amp)]).spawn(ir.N_REAL)


def savedPath(key, amp):
    return os.path.join(ir.OUT_DIR, "settings", f"{key}_{ir.ampTag(amp)}.json")


def brightPath(key):
    return os.path.join(OUT, "bright", f"{key}.json")


def testPath(key, amp):
    return os.path.join(OUT, "test", f"{key}_{ir.ampTag(amp)}.json")


def samePhotonsCheck(key, amp, seeds):
    saved = json.load(open(savedPath(key, amp)))
    pb = next(p for p in saved["bins"] if abs(p["binWidth_s"] - 3e-6) < 1e-15)
    for j in range(2):
        mvt = _task((key, amp, seeds[j], [3e-6]))[0][0]
        old = pb["mvt_ms"][j] / 1e3
        if not ((np.isnan(mvt) and np.isnan(old)) or abs(mvt - old) <= 1e-9 * max(abs(old), 1e-30)):
            sys.exit(f"STOP: {key} amp {amp:g} realisation {j} at 0.003 ms: recomputed {mvt} s, saved {old} s")


def runSetting(pool, key, amp, bws, path):
    t0 = time.time()
    seeds = seedsFor(key, amp)
    samePhotonsCheck(key, amp, seeds)
    res = list(pool.map(_task, [(key, amp, s, bws) for s in seeds], chunksize=2))
    bins = []
    for b, bw in enumerate(bws):
        mvt = [r[b][0] for r in res]
        unc = [r[b][1] for r in res]
        bins.append(dict(binWidth_s=bw, mvt_ms=[v * 1e3 for v in mvt], unc_ms=[v * 1e3 for v in unc],
                         **ir.summarise(mvt, unc)))
    medians = [p["median"] / 1e3 if np.isfinite(p["median"]) else None for p in bins]
    snrs = list(pool.map(ir._snrTask, [(key, amp, s, medians) for s in seeds], chunksize=4))
    for b, p in enumerate(bins):
        vals = np.array([s[b] for s in snrs])
        good = np.isfinite(vals) & (vals > 0)
        p["snrMvt"] = float(vals[good].mean()) if good.any() else float("nan")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(pulse=key, amplitude=amp, nRealisations=ir.N_REAL,
                       seed=[ir.SEED, list(ir.PULSES).index(key), ALL_AMPS.index(amp)], bins=bins), f, default=float)
    line = "  ".join(f"{p['binWidth_s'] * 1e3:g}ms: {p['median']:.4g} ms ({p['successPct']:.0f}% valid)" for p in bins)
    print(f"{key:8s} amplitude {amp:>6g}: {line}  [same photons checked; {(time.time() - t0) / 60:.1f} min]", flush=True)


def stage1(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key in NARROW + WIDE:
            if os.path.exists(brightPath(key)):
                print(f"{key:8s} bright: already done, skipped")
                continue
            if not os.path.exists(savedPath(key, ir.BRIGHT_AMPLITUDE)):
                print(f"WARNING: {key}: saved bright run not found, skipped")
                continue
            runSetting(pool, key, ir.BRIGHT_AMPLITUDE, NEW_BW[key], brightPath(key))


# ---------------------------------------------------------------- truth
def brightBins(key):
    """All bright-run bin widths of a pulse: saved (0.1, 0.01, 0.003 ms) and new, sorted from large to small."""
    out = {}
    if os.path.exists(savedPath(key, ir.BRIGHT_AMPLITUDE)):
        for p in json.load(open(savedPath(key, ir.BRIGHT_AMPLITUDE)))["bins"]:
            out[p["binWidth_s"]] = p
    if os.path.exists(brightPath(key)):
        for p in json.load(open(brightPath(key)))["bins"]:
            out[p["binWidth_s"]] = p
    return dict(sorted(out.items(), reverse=True))


def validFrac(p):
    return p["nValid"] / max(len(p["mvt_ms"]), 1)


def truthOf(bins):
    eligible = [bw for bw, p in bins.items() if np.isfinite(p["median"]) and validFrac(p) >= MIN_VALID_FRAC]
    if not eligible:
        return None
    meds = {bw: bins[bw]["median"] for bw in eligible}
    anchor = meds[min(eligible)]
    ok = [bw for bw in eligible if abs(meds[bw] - anchor) / anchor <= GATE]
    bwStar = max(ok)
    tTrue = float(np.mean([meds[bw] for bw in eligible if bw <= bwStar]))
    small = sorted(eligible)[:3]                                   # three smallest eligible, small to large
    steps = [meds[small[i]] / meds[small[i + 1]] - 1 for i in range(len(small) - 1)]
    settled = len(small) == 3 and all(abs(s) <= GATE for s in steps)
    # smallest bin width from which the median stays within 10% of the anchor (where it "settles")
    order = sorted(eligible, reverse=True)
    settleBw = next((bw for i, bw in enumerate(order) if all(abs(meds[b] - anchor) / anchor <= GATE for b in order[i:])),
                    None)
    return dict(tTrue=tTrue, bwStar=bwStar, eligible=eligible, anchorBw=min(eligible), steps=steps, settled=settled,
                settleBw=settleBw, meds=meds)


def originalTruth(key):
    p = {q["binWidth_s"]: q["median"] for q in json.load(open(savedPath(key, ir.BRIGHT_AMPLITUDE)))["bins"]}
    anchor = p[min(ir.BIN_WIDTHS)]
    ok = [bw for bw in ir.BIN_WIDTHS if np.isfinite(p[bw]) and abs(p[bw] - anchor) / anchor <= GATE]
    bwStar = max(ok) if ok else min(ir.BIN_WIDTHS)
    return float(np.mean([p[bw] for bw in ir.BIN_WIDTHS if bw <= bwStar])), bwStar


def stage2(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key in NARROW:
            t = truthOf(brightBins(key))
            if t is None or not t["settled"]:
                print(f"{key:8s}: true MVT not settled (or no bright data) -> no stage-2 test settings")
                continue
            bws = [bw for bw in NEW_BW[key] if bw in t["eligible"] and bw <= t["bwStar"]]
            if not bws:
                print(f"{key:8s}: no new bin width is eligible and <= BW* -> no stage-2 test settings")
                continue
            for amp in ir.AMPLITUDES:
                if os.path.exists(testPath(key, amp)):
                    print(f"{key:8s} amplitude {amp:>6g}: already done, skipped")
                    continue
                if not os.path.exists(savedPath(key, amp)):
                    print(f"WARNING: {key} amplitude {amp:g}: saved test setting not found, skipped")
                    continue
                runSetting(pool, key, amp, bws, testPath(key, amp))


# ---------------------------------------------------------------- analysis
def stage3():
    import analyse_rung_positions as A
    classify, band, _ = ir.loadValidationCurve()
    rng = np.random.default_rng(ir.SEED)
    summary = dict(bright={}, truth={}, criteria={})

    print("\n===== B1. BRIGHT RUNS (amplitude 3000), ALL BIN WIDTHS =====")
    print(f"{'pulse':8s}{'BW ms':>8s}{'peak cts/bin':>13s}{'valid':>6s}{'limit':>6s}{'fail':>5s}{'median':>9s}"
          f"{'boot 16-84% of median':>23s}{'16-84% values':>16s}{'top gap %':>10s}{'gap (ms)':>16s}{'eligible':>9s}")
    for key in ir.PULSES:
        bins = brightBins(key)
        if not bins:
            continue
        tstart, tstop = ir.window(ir.PULSES[key])
        summary["bright"][key] = {}
        for bw, p in bins.items():
            mvt, unc = np.asarray(p["mvt_ms"], float), np.asarray(p["unc_ms"], float)
            v = mvt[np.isfinite(mvt) & (mvt > 0) & (unc > 0)]
            if v.size >= 2:
                boot = np.median(rng.choice(v, size=(N_BOOT, v.size), replace=True), axis=1)
                b16, b84 = np.percentile(boot, [16, 84])
                R = A.rungsFast(ir.lightCurve(np.array([]), tstart, tstop, bw).size, bw)
                inside = v[(v > R[0]) & (v < R[-1])]
                gi = np.searchsorted(R, inside) - 1
                if gi.size:
                    top = int(np.bincount(gi).argmax())
                    topShare, gap = float(np.mean(gi == top)), (float(R[top]), float(R[top + 1]))
                else:
                    topShare, gap = float("nan"), (float("nan"), float("nan"))
            else:
                b16 = b84 = topShare = float("nan")
                gap = (float("nan"), float("nan"))
            peak = ir.BRIGHT_AMPLITUDE * ir.BKG_RATE * bw + ir.BKG_RATE * bw
            elig = np.isfinite(p["median"]) and validFrac(p) >= MIN_VALID_FRAC
            summary["bright"][key][f"{bw * 1e3:g}"] = dict(median=p["median"], boot16=b16, boot84=b84, p16=p["p16"],
                                                          p84=p["p84"], nValid=p["nValid"], nLimit=p["nLimit"],
                                                          nFailed=p["nFailed"], peakCountsPerBin=peak,
                                                          topGapShare=topShare, topGap_ms=gap, eligible=bool(elig))
            bs = f"{b16:.4g}-{b84:.4g}"
            vs = f"{p['p16']:.3g}-{p['p84']:.3g}"
            gs = f"{gap[0]:.3g}-{gap[1]:.3g}"
            print(f"{key:8s}{bw * 1e3:>8g}{peak:>13.3g}{p['nValid']:>6d}{p['nLimit']:>6d}{p['nFailed']:>5d}"
                  f"{p['median']:>9.4g}{bs:>23s}{vs:>16s}{(100 * topShare if np.isfinite(topShare) else float('nan')):>10.0f}"
                  f"{gs:>16s}{('yes' if elig else 'NO'):>9s}")

    print("\n===== T1 / T2. TRUE MVT (eligible bin widths: >= 50% valid) =====")
    print(f"{'pulse':8s}{'original true (BW*)':>22s}{'new true (BW*)':>20s}{'change':>8s}{'last two steps':>18s}"
          f"{'T2':>11s}{'settles from':>14s}")
    truths = {}
    for key in ir.PULSES:
        if not os.path.exists(savedPath(key, ir.BRIGHT_AMPLITUDE)):
            continue
        t = truthOf(brightBins(key))
        o, obw = originalTruth(key)
        if t is None:
            print(f"{key:8s}  no eligible bin width")
            continue
        truths[key] = t
        steps = ", ".join(f"{100 * s:+.0f}%" for s in t["steps"])
        sb = f"{t['settleBw'] * 1e3:g} ms" if t["settleBw"] else "-"
        oS = f"{o:.3f} ({obw * 1e3:g})"
        nS = f"{t['tTrue']:.3f} ({t['bwStar'] * 1e3:g})"
        print(f"{key:8s}{oS:>22s}{nS:>20s}{100 * (t['tTrue'] / o - 1):>+7.1f}%{steps:>18s}"
              f"{('settled' if t['settled'] else 'UNSETTLED'):>11s}{sb:>14s}")
        summary["truth"][key] = dict(original=o, originalBwStar_ms=obw * 1e3, new=t["tTrue"], newBwStar_ms=t["bwStar"] * 1e3,
                                     steps=t["steps"], settled=t["settled"],
                                     settleBw_ms=t["settleBw"] * 1e3 if t["settleBw"] else None,
                                     eligible_ms=[b * 1e3 for b in t["eligible"]])
    need = [k for k, t in truths.items() if t["settleBw"] and t["settleBw"] < 2e-6]
    print("pulses whose median only settles below the 0.002 ms real-data limit: " + (", ".join(need) if need else "none"))
    for k in ("G1", "G3", "G10", "G30"):
        if k in truths:
            print(f"criterion 4: {k} true MVT / sigma = {truths[k]['tTrue'] / (ir.PULSES[k]['sigma'] * 1e3):.3f}")

    print("\n===== CRITERIA 1-3 RECOMPUTED (settled narrow pulses, new truth and BW*) =====")
    for key in NARROW:
        t = truths.get(key)
        if t is None or not t["settled"]:
            print(f"{key}: not settled -> not recomputed")
            continue
        rows = []
        for amp in ir.AMPLITUDES:
            src = []
            if os.path.exists(savedPath(key, amp)):
                src += json.load(open(savedPath(key, amp)))["bins"]
            if os.path.exists(testPath(key, amp)):
                src += json.load(open(testPath(key, amp)))["bins"]
            for p in src:
                if not np.isfinite(p["median"]):
                    continue
                if not np.isfinite(p.get("snrMvt", float("nan"))):
                    print(f"WARNING: {key} amp {amp:g} BW {p['binWidth_s'] * 1e3:g} ms: no SNR_MVT saved, setting skipped")
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    cls, _ = classify(p["median"], p["snrMvt"])
                bw = p["binWidth_s"]
                ref = t["meds"].get(bw, brightBins(key).get(bw, {}).get("median", np.nan))
                rows.append(dict(amp=amp, bw=bw, bl=bw > t["bwStar"], med=p["median"], p16=p["p16"], p84=p["p84"],
                                 snr=p["snrMvt"], cls=cls, valid=p["successPct"],
                                 inT=bool(p["p16"] <= t["tTrue"] <= p["p84"]),
                                 inSame=bool(np.isfinite(ref) and p["p16"] <= ref <= p["p84"]),
                                 err=p["median"] / t["tTrue"] - 1))
        rows.sort(key=lambda x: (x["amp"], -x["bw"]))
        print(f"\n{key}: true MVT {t['tTrue']:.3f} ms, BW* {t['bwStar'] * 1e3:g} ms")
        print(f"{'amp':>6s}{'BW ms':>8s}{'binlim':>7s}{'median':>9s}{'16-84%':>16s}{'valid%':>7s}{'SNR':>8s}{'class':>8s}"
              f"{'in range':>9s}{'error':>8s}")
        for x in rows:
            rr = f"{x['p16']:.3g}-{x['p84']:.3g}"
            print(f"{x['amp']:>6g}{x['bw'] * 1e3:>8g}{('yes' if x['bl'] else ''):>7s}{x['med']:>9.3g}{rr:>16s}"
                  f"{x['valid']:>7.0f}{x['snr']:>8.3g}{x['cls']:>8s}{('yes' if x['inT'] else 'no'):>9s}{100 * x['err']:>+7.0f}%")
        rob = [x for x in rows if x["cls"] == "robust" and not x["bl"]]
        c1 = sum(x["inT"] for x in rob)
        c1s = sum(x["inSame"] for x in rob)
        c2 = sum(abs(x["err"]) <= ir.ROBUST_TOLERANCE for x in rob)
        work = [x for x in rows if not x["bl"] and x["inT"] and np.isfinite(x["snr"])]
        minSnr = min((x["snr"] for x in work), default=float("nan"))
        lo, mid, hi = band(t["tTrue"])
        c3 = "inside" if lo <= minSnr <= hi else ("left of band" if minSnr < lo else ("right of band" if minSnr > hi else "n/a"))
        print(f"criterion 1: {c1}/{len(rob)} robust not-bin-limited settings contain the true MVT "
              f"(same-bin-width bright median, post hoc: {c1s}/{len(rob)})")
        print(f"criterion 2: {c2}/{len(rob)} within {100 * ir.ROBUST_TOLERANCE:.0f}%")
        print(f"criterion 3: lowest working SNR {minSnr:.3g}; band {lo:.3g}-{hi:.3g} (median {mid:.3g}) -> {c3} "
              f"({'pass' if c3 in ('inside', 'left of band') else 'FAIL'})")
        summary["criteria"][key] = dict(tTrue=t["tTrue"], bwStar_ms=t["bwStar"] * 1e3, c1=[c1, len(rob)],
                                        c1sameBw=[c1s, len(rob)], c2=[c2, len(rob)], minSnr=minSnr,
                                        band=[lo, mid, hi], c3=c3, rows=rows)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"\nSaved {OUT}/summary.json")
    plots(truths)


def plots(truths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(ir.PLOT_DIR, exist_ok=True)
    keys = [k for k in ir.PULSES if brightBins(k)]
    rng = np.random.default_rng(ir.SEED)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, key in zip(axes.ravel(), keys):
        for bw, p in brightBins(key).items():
            mvt, unc = np.asarray(p["mvt_ms"], float), np.asarray(p["unc_ms"], float)
            v = mvt[np.isfinite(mvt) & (mvt > 0) & (unc > 0)]
            if v.size < 2:
                continue
            boot = np.median(rng.choice(v, size=(N_BOOT, v.size), replace=True), axis=1)
            b16, b84 = np.percentile(boot, [16, 84])
            elig = validFrac(p) >= MIN_VALID_FRAC
            new = os.path.exists(brightPath(key)) and bw in [q["binWidth_s"] for q in json.load(open(brightPath(key)))["bins"]]
            ax.errorbar(bw * 1e3, p["median"], yerr=[[p["median"] - b16], [b84 - p["median"]]], fmt="s" if new else "o",
                        color="C1" if new else "C0", mfc=None if elig else "none", ms=7, capsize=3)
            ax.annotate(f"{100 * validFrac(p):.0f}%", (bw * 1e3, p["median"]), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7)
        t = truths.get(key)
        if t:
            ax.axhline(t["tTrue"], color="C3", lw=1, label=f"new true MVT {t['tTrue']:.3g} ms"
                       + ("" if t["settled"] else " (UNSETTLED)"))
            ax.axvline(t["bwStar"] * 1e3, color="0.5", ls="--", lw=1, label="BW*")
        ax.axvline(0.002, color="k", ls=":", lw=1, label="0.002 ms (real-data limit)")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("bin width (ms)"); ax.set_ylabel("bright-run median MVT (ms)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=7)
    ax = axes.ravel()[-1]
    ax.plot([], [], "o", color="C0", label="saved bin widths")
    ax.plot([], [], "s", color="C1", label="new bin widths (B1)")
    ax.plot([], [], "o", color="0.3", mfc="none", label="< 50% valid (not used)")
    ax.plot([], [], " ", label="bars: bootstrap 16-84% of the median")
    ax.plot([], [], " ", label="numbers: % valid realisations")
    ax.legend(fontsize=9, loc="center"); ax.set_axis_off()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ir.PLOT_DIR, f"B1_bright_medians.{ext}"), dpi=130)
    plt.close(fig)

    rowsK = [k for k in NARROW if brightBins(k)]
    if rowsK:
        nb = max(len(brightBins(k)) for k in rowsK)
        fig, axes = plt.subplots(len(rowsK), nb, figsize=(3.2 * nb, 2.4 * len(rowsK)), squeeze=False)
        for i, key in enumerate(rowsK):
            for j, (bw, p) in enumerate(brightBins(key).items()):
                ax = axes[i, j]
                mvt, unc = np.asarray(p["mvt_ms"], float), np.asarray(p["unc_ms"], float)
                v = mvt[np.isfinite(mvt) & (mvt > 0) & (unc > 0)]
                if v.size >= 2:
                    ax.hist(v, bins=np.linspace(v.min() * 0.98, v.max() * 1.02, 80), color="C0")
                    ax.axvline(np.median(v), color="k", lw=1)
                    t = truths.get(key)
                    if t:
                        ax.axvline(t["tTrue"], color="C3", lw=1)
                ax.set_title(f"{key}, bright, {bw * 1e3:g} ms ({p['nValid']} valid)", fontsize=7)
            for j in range(len(brightBins(key)), nb):
                axes[i, j].set_axis_off()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(ir.PLOT_DIR, f"B1_hist_bright.{ext}"), dpi=130)
        plt.close(fig)
    print(f"Saved plots: {ir.PLOT_DIR}/B1_bright_medians.png, B1_hist_bright.png (.pdf too)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    hm = loadHaar()
    commit = os.popen(f"git -C {ir.REPO} log -1 --format='%h %s'").read().strip()
    print(f"MVTTools code in use: {commit}", flush=True)
    t0 = time.time()
    if args.stage in ("1", "all"):
        stage1(args.workers)
        print(f"Stage 1 finished ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if args.stage in ("2", "all"):
        stage2(args.workers)
        print(f"Stage 2 finished ({(time.time() - t0) / 60:.1f} min)", flush=True)
    stage3()
