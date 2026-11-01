"""
Injection-recovery test of the MVT pipeline with simulated pulses of known shape (sprint step 6).

Design (agreed before writing; simulation details follow Bala et al. 2026 and their public code, GBM_MVT_paper)
  Pulses      Gaussian, sigma = 1, 3, 10, 30 ms;  Norris (tau_1, tau_2) = (3, 10), (10, 30), (30, 100) ms.
              Norris: A * lambda * exp(-tau_1/(t-t0) - (t-t0)/tau_2), lambda = exp(2 sqrt(tau_1/tau_2)), peak = A.
  Brightness  peak source rate = amplitude x background; amplitudes 0.3, 1, 3, 10, 30, 100, 300 (test grid);
              3000 only for the very bright run that defines each pulse's true MVT.
  Background  constant 1000 counts/s.
  Windows     as Bala et al.: Gaussian centre +- 25 sigma; Norris from t0 - 2 tau_1 to sqrt(tau_1 tau_2) + 6 tau_2.
  Photons     source and background simulated as independent Poisson processes (equivalent to Bala et al.'s
              single stream split at random into source and background). Background times uniform; Gaussian
              times drawn exactly from a normal distribution; Norris times from the cumulative rate on a 0.3
              microsecond grid (finer than the smallest bin, so the pulse has no steps at the analysis bins).
  Bin widths  0.1, 0.01 and 0.003 ms; complete bins only. The same 300 photon sets are analysed at every
              bin width, as in Bala et al.
  MVT         main scheme: 300 independent simulated observations of the true pulse, each analysed once with
              haarPowerMod (errors sqrt(counts)), as Bala et al. built their validation curve.
              Each realisation: failed (MVT <= 0 or not finite), upper limit (uncertainty not > 0) or valid;
              median and 16/84th percentiles of the valid values (at least 2 needed).
  SNR_MVT     as Bala et al.: at the median MVT, for each realisation, maximum counts in bins of width MVT
              starting at the window start, divided by sqrt(background rate x MVT), with the background rate
              from that realisation's simulated background photons; averaged over realisations.
              No detector merge and no energy channels in these simulations, so physical = Bala-compatible.
  True MVT    as Bala et al.'s 08_Min_SNR_MVT.py: anchor = median MVT of the bright run (amplitude 3000) at the
              smallest bin width; BW* = largest bin width whose bright-run median is within 10% of the anchor;
              true MVT = mean of the bright-run medians at bin widths <= BW*. Bin widths > BW* are bin-limited.
  Class       Bala et al.'s validation curve (mvt_snr_fit_model.npz, clasify_mvt_value.py): UL / LUL / robust.
  Check       real-data scheme: Gaussian sigma = 3 ms, all 7 amplitudes, 0.01 ms, 5 independent simulated
              observations each; every observation resampled 300 times with estimateMvtDistribution
              (resampleErrors=True), exactly as for real bursts.

Criteria (fixed before running)
  1. Every test setting at a bin width <= BW* that is classed robust has the true MVT inside its 16-84% range.
  2. No test setting at a bin width <= BW* (not bin-limited) classed robust has a median MVT more than 50% from
     the true MVT. Bin-limited settings classed robust are reported separately (not part of the criterion).
  3. For each pulse, the lowest SNR_MVT among test settings at bin widths <= BW* whose 16-84% range contains the
     true MVT (Bala et al.'s rule) lies inside or to the left of Bala et al.'s 95% band at the true MVT.
  4. Gaussian pulses: true MVT reported against sigma (no pass/fail).
  5. Real-data scheme against the main scheme (sigma = 3 ms, 0.01 ms), at each amplitude: the main-scheme median
     lies within the 95% confidence interval of the mean of the real-data-scheme medians
     (mean +- t(0.975, n-1) s / sqrt(n), n = 5), and the success rates differ by at most 10 percentage points
     (main scheme against the mean of the 5). Amplitudes where either scheme has fewer than 2 valid medians are
     reported as not comparable.

Output: results/injection_recovery/ (one JSON per setting, so an interrupted run resumes; summary.json) and plots
P8 (recovered against true MVT) and P9 (settings on the validation curve) in results/plots/.

Usage
  python run_injection_recovery.py                 simulations (grid, then the real-data check), then analysis
  python run_injection_recovery.py --part grid     only the main grid
  python run_injection_recovery.py --part check    only the real-data check
  python run_injection_recovery.py --post          only the analysis and plots from saved results
"""
import os, sys, io, json, time, argparse, contextlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REPO = "/mnt/data/rgl36/FYP/MVTTools"
BALA_CODE = "/mnt/data/rgl36/FYP/external/GBM_MVT_paper/MVTfermi"
RESULTS = "/mnt/data/rgl36/FYP/results"
OUT_DIR = os.path.join(RESULTS, "injection_recovery")
PLOT_DIR = os.path.join(RESULTS, "plots")

BKG_RATE = 1000.0                                      # counts/s
AMPLITUDES = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
BRIGHT_AMPLITUDE = 3000.0
BIN_WIDTHS = [1e-4, 1e-5, 3e-6]                        # s, largest first
N_REAL = 300
SEED = 2026
NORRIS_GRID_S = 3e-7                                   # photon-time grid for Norris pulses
BIAS_GATE = 0.10                                       # Bala et al.'s bin-width gate for the true MVT
ROBUST_TOLERANCE = 0.50                                # criterion 2
SUCCESS_TOLERANCE = 10.0                               # criterion 5, percentage points
CHECK_PULSE, CHECK_BIN, CHECK_OBS = "G3", 1e-5, 5

PULSES = {
    "G1": dict(shape="gaussian", sigma=1e-3),
    "G3": dict(shape="gaussian", sigma=3e-3),
    "G10": dict(shape="gaussian", sigma=10e-3),
    "G30": dict(shape="gaussian", sigma=30e-3),
    "N3_10": dict(shape="norris", rise=3e-3, decay=10e-3),
    "N10_30": dict(shape="norris", rise=10e-3, decay=30e-3),
    "N30_100": dict(shape="norris", rise=30e-3, decay=100e-3),
}


def window(p):
    """Simulation and analysis window (s), as calculate_adaptive_simulation_params in Bala et al.'s code."""
    if p["shape"] == "gaussian":
        gridRes = p["sigma"] / 10.0                    # their padding uses the unclipped grid resolution
        half = 5 * p["sigma"] + 10.0 * gridRes * 20
        return -half, half
    t0 = 0.0
    return t0 - 2 * p["rise"], np.sqrt(p["rise"] * p["decay"]) + 6 * p["decay"]


def norrisRate(t, p, amp):
    """Source rate (counts/s) of a Norris pulse starting at t0 = 0, peak = amp x background."""
    out = np.zeros_like(t)
    m = t > 0.0
    lam = amp * BKG_RATE * np.exp(2.0 * np.sqrt(p["rise"] / p["decay"]))
    out[m] = lam * np.exp(-p["rise"] / t[m] - t[m] / p["decay"])
    return out


_norrisCache = {}


def norrisCumulative(key):
    """Cumulative rate per unit amplitude on a fine grid from t0 = 0 to the window end (cached per process)."""
    if key not in _norrisCache:
        p = PULSES[key]
        _, tstop = window(p)
        grid = NORRIS_GRID_S * np.arange(int(np.ceil(tstop / NORRIS_GRID_S)) + 1)
        cum = np.cumsum(norrisRate(grid, p, 1.0)) * NORRIS_GRID_S
        # just after t0 the rate underflows to exactly 0; keep a single leading zero so the cumulative rate
        # is strictly increasing, as np.interp requires
        i0 = max(int(np.argmax(cum > 0.0)) - 1, 0)
        _norrisCache[key] = (grid[i0:], cum[i0:])
    return _norrisCache[key]


def simulate(key, amp, seed):
    """One simulated observation: event times (s) and the number of background events."""
    p = PULSES[key]
    tstart, tstop = window(p)
    rng = np.random.default_rng(seed)
    nb = rng.poisson(BKG_RATE * (tstop - tstart))
    tb = rng.uniform(tstart, tstop, nb)
    if p["shape"] == "gaussian":
        n = rng.poisson(amp * BKG_RATE * p["sigma"] * np.sqrt(2 * np.pi))
        ts = rng.normal(0.0, p["sigma"], n)
        ts = ts[(ts >= tstart) & (ts < tstop)]
    else:
        grid, cum = norrisCumulative(key)
        n = rng.poisson(amp * cum[-1])
        ts = np.interp(rng.uniform(0.0, cum[-1], n), cum, grid)
        ts = ts[ts < tstop]                            # the grid ends up to one step after the window end
    return np.sort(np.concatenate([tb, ts])), int(nb)


def lightCurve(times, tstart, tstop, bw):
    """Counts in complete bins of width bw from tstart (a partial last bin is dropped)."""
    n = int(np.floor((tstop - tstart) / bw + 1e-9))
    return np.histogram(times, bins=tstart + bw * np.arange(n + 1))[0].astype(float)


def snrAt(times, nb, tstart, tstop, mvtS):
    """SNR_MVT of one observation as in Bala et al.'s compute_snr_for_mvt."""
    counts = np.histogram(times, bins=np.arange(tstart, tstop + mvtS, mvtS))[0]
    bkgPerBin = nb / (tstop - tstart) * mvtS
    return float(counts.max() / np.sqrt(bkgPerBin)) if bkgPerBin > 0 else float("nan")


def _mvtTask(args):
    """Worker: one realisation, MVT and uncertainty (s) at every bin width."""
    key, amp, seed = args
    tstart, tstop = window(PULSES[key])
    times, nb = simulate(key, amp, seed)
    out = []
    for bw in BIN_WIDTHS:
        counts = lightCurve(times, tstart, tstop, bw)
        try:
            r = haarMethods.haarPowerMod(counts, np.sqrt(counts), minimumBinSizeSeconds=bw)
            out.append((float(r[2]), float(r[3])))
        except Exception:
            out.append((float("nan"), float("nan")))
    return out, nb


def _snrTask(args):
    """Worker: one realisation (same seed, so the same photons), SNR_MVT at each bin width's median MVT."""
    key, amp, seed, medians = args
    tstart, tstop = window(PULSES[key])
    times, nb = simulate(key, amp, seed)
    return [snrAt(times, nb, tstart, tstop, m) if m is not None else float("nan") for m in medians]


def summarise(mvt, unc):
    """Failed / upper limit / valid split and the valid median and percentiles (ms), as in our pipeline."""
    mvt, unc = np.asarray(mvt), np.asarray(unc)
    failed = ~np.isfinite(mvt) | (mvt <= 0.0)
    limit = ~failed & ~(unc > 0.0)
    valid = ~failed & ~limit
    out = dict(nValid=int(valid.sum()), nLimit=int(limit.sum()), nFailed=int(failed.sum()),
               successPct=100.0 * valid.sum() / mvt.size)
    if valid.sum() >= 2:
        p16, med, p84 = np.percentile(mvt[valid] * 1e3, [16, 50, 84])
        out.update(median=float(med), p16=float(p16), p84=float(p84))
    else:
        out.update(median=float("nan"), p16=float("nan"), p84=float("nan"))
    return out


def ampTag(amp):
    return f"A{amp:g}".replace(".", "p")


def runGrid(workers):
    os.makedirs(os.path.join(OUT_DIR, "settings"), exist_ok=True)
    allAmps = [(i, a) for i, a in enumerate(AMPLITUDES)] + [(len(AMPLITUDES), BRIGHT_AMPLITUDE)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for pi, key in enumerate(PULSES):
            tstart, tstop = window(PULSES[key])
            for ai, amp in allAmps:
                path = os.path.join(OUT_DIR, "settings", f"{key}_{ampTag(amp)}.json")
                if os.path.exists(path):
                    print(f"{key:8s} amplitude {amp:>6g}: already done, skipped")
                    continue
                t0 = time.time()
                seeds = np.random.SeedSequence([SEED, pi, ai]).spawn(N_REAL)
                res = list(pool.map(_mvtTask, [(key, amp, s) for s in seeds], chunksize=4))
                perBin = []
                for j, bw in enumerate(BIN_WIDTHS):
                    mvt = [r[0][j][0] for r in res]
                    unc = [r[0][j][1] for r in res]
                    perBin.append(dict(binWidth_s=bw, mvt_ms=[v * 1e3 for v in mvt], unc_ms=[v * 1e3 for v in unc],
                                       **summarise(mvt, unc)))
                medians = [pb["median"] / 1e3 if np.isfinite(pb["median"]) else None for pb in perBin]
                snrs = list(pool.map(_snrTask, [(key, amp, s, medians) for s in seeds], chunksize=4))
                for j, pb in enumerate(perBin):
                    vals = np.array([s[j] for s in snrs])
                    good = np.isfinite(vals) & (vals > 0)
                    pb["snrMvt"] = float(vals[good].mean()) if good.any() else float("nan")
                rec = dict(pulse=key, params=PULSES[key], amplitude=amp, bright=amp == BRIGHT_AMPLITUDE,
                           window_s=[tstart, tstop], nRealisations=N_REAL, seed=[SEED, pi, ai],
                           meanBackgroundEvents=float(np.mean([r[1] for r in res])), bins=perBin)
                with open(path, "w") as f:
                    json.dump(rec, f, default=float)
                line = "  ".join(f"{pb['binWidth_s'] * 1e3:g}ms: {pb['median']:.3g} ms ({pb['successPct']:.0f}% valid, "
                                 f"SNR {pb['snrMvt']:.3g})" for pb in perBin)
                print(f"{key:8s} amplitude {amp:>6g}: {line}  [{(time.time() - t0) / 60:.1f} min]", flush=True)


def runCheck(workers):
    os.makedirs(os.path.join(OUT_DIR, "check"), exist_ok=True)
    tstart, tstop = window(PULSES[CHECK_PULSE])
    for ai, amp in enumerate(AMPLITUDES):
        for j in range(CHECK_OBS):
            path = os.path.join(OUT_DIR, "check", f"{CHECK_PULSE}_{ampTag(amp)}_obs{j}.json")
            if os.path.exists(path):
                print(f"check amplitude {amp:>6g} observation {j}: already done, skipped")
                continue
            t0 = time.time()
            times, nb = simulate(CHECK_PULSE, amp, np.random.SeedSequence([SEED, 99, ai, j]))
            counts = lightCurve(times, tstart, tstop, CHECK_BIN)
            r = haarMethods.estimateMvtDistribution(counts, np.sqrt(counts), haarMethods.haarPowerMod,
                                                    binSizeSeconds=CHECK_BIN, nRealisations=N_REAL,
                                                    resampleErrors=True, nWorkers=workers, seed=[SEED, 199, ai, j])
            s = summarise(r["mvtValues"], r["mvtUncertainties"])
            s["snrMvt"] = snrAt(times, nb, tstart, tstop, s["median"] / 1e3) if np.isfinite(s["median"]) else float("nan")
            with open(path, "w") as f:
                json.dump(dict(pulse=CHECK_PULSE, amplitude=amp, observation=j, binWidth_s=CHECK_BIN,
                               seed=[SEED, 99, ai, j], **s), f, default=float)
            print(f"check amplitude {amp:>6g} observation {j}: median {s['median']:.3g} ms, "
                  f"{s['successPct']:.0f}% valid, SNR {s['snrMvt']:.3g}  [{(time.time() - t0) / 60:.1f} min]", flush=True)


def loadValidationCurve():
    sys.path.insert(0, BALA_CODE)
    import clasify_mvt_value as v
    v.MODEL_FILE = os.path.join(BALA_CODE, "mvt_snr_fit_model.npz")
    model = v.load_model()
    if model is None:
        sys.exit(f"Could not load the validation curve from {v.MODEL_FILE}")
    interpolators, rangeLog = model
    names = {"Below 95% CI (Upper Limit)": "UL", "Within 95% CI (Likely Upper Limit)": "LUL",
             "Above 95% CI (Robust Measurement)": "robust"}

    def classify(mvtMs, snr):
        if not (np.isfinite(mvtMs) and np.isfinite(snr) and mvtMs > 0 and snr > 0):
            return "none", False
        with contextlib.redirect_stdout(io.StringIO()):         # silence the out-of-range warning; flagged below
            label, _ = v.classify_mvt_value(mvtMs, snr, interpolators, rangeLog)
        outside = not (rangeLog[0] <= np.log10(mvtMs) <= rangeLog[1])
        return names.get(label, label), outside

    def band(mvtMs):
        x = np.log10(mvtMs)
        return tuple(float(10 ** interpolators[k](x)) for k in ("lower", "median", "upper"))

    return classify, band, rangeLog


def post():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classify, band, rangeLog = loadValidationCurve()
    recs = {}
    for key in PULSES:
        for amp in AMPLITUDES + [BRIGHT_AMPLITUDE]:
            path = os.path.join(OUT_DIR, "settings", f"{key}_{ampTag(amp)}.json")
            if os.path.exists(path):
                recs[(key, amp)] = json.load(open(path))
    missing = [(k, a) for k in PULSES for a in AMPLITUDES + [BRIGHT_AMPLITUDE] if (k, a) not in recs]
    if missing:
        print(f"WARNING: {len(missing)} settings missing: {missing}")

    # true MVT and BW* per pulse
    truth = {}
    print("\n===== TRUE MVT (bright run, amplitude 3000) =====")
    for key in PULSES:
        b = recs.get((key, BRIGHT_AMPLITUDE))
        if b is None:
            continue
        meds = {pb["binWidth_s"]: pb["median"] for pb in b["bins"]}
        anchor = meds[min(BIN_WIDTHS)]
        if not np.isfinite(anchor):
            print(f"{key}: no valid median at the smallest bin width in the bright run; pulse skipped")
            continue
        ok = [bw for bw in BIN_WIDTHS if np.isfinite(meds[bw]) and abs(meds[bw] - anchor) / anchor <= BIAS_GATE]
        bwStar = max(ok) if ok else min(BIN_WIDTHS)
        tTrue = float(np.mean([meds[bw] for bw in BIN_WIDTHS if bw <= bwStar]))
        truth[key] = dict(tTrue_ms=tTrue, bwStar_s=bwStar, brightMedians_ms={f"{bw * 1e3:g}": meds[bw] for bw in BIN_WIDTHS},
                          brightSnr={f"{pb['binWidth_s'] * 1e3:g}": pb["snrMvt"] for pb in b["bins"]})
        extra = f", true MVT / sigma = {tTrue / (PULSES[key]['sigma'] * 1e3):.3f}" if PULSES[key]["shape"] == "gaussian" else ""
        print(f"{key:8s} bright medians (ms) " + ", ".join(f"{bw * 1e3:g}: {meds[bw]:.3f}" for bw in BIN_WIDTHS)
              + f" -> BW* = {bwStar * 1e3:g} ms, true MVT = {tTrue:.3f} ms{extra}")

    # every test setting
    rows = []
    for key in truth:
        T, bwStar = truth[key]["tTrue_ms"], truth[key]["bwStar_s"]
        for amp in AMPLITUDES:
            r = recs.get((key, amp))
            if r is None:
                continue
            for pb in r["bins"]:
                cls, outside = classify(pb["median"], pb["snrMvt"])
                inRange = bool(np.isfinite(pb["median"]) and pb["p16"] <= T <= pb["p84"])
                rows.append(dict(pulse=key, shape=PULSES[key]["shape"], amplitude=amp, binWidth_ms=pb["binWidth_s"] * 1e3,
                                 binLimited=pb["binWidth_s"] > bwStar, tTrue_ms=T, median_ms=pb["median"],
                                 p16_ms=pb["p16"], p84_ms=pb["p84"], successPct=pb["successPct"], snrMvt=pb["snrMvt"],
                                 cls=cls, outsideCurveRange=outside, trueInRange=inRange,
                                 relErr=(pb["median"] - T) / T if np.isfinite(pb["median"]) else float("nan")))

    print("\n===== ALL TEST SETTINGS =====")
    print(f"{'pulse':8s}{'amp':>6s}{'BW ms':>7s}{'binlim':>7s}{'true':>8s}{'median':>9s}{'16-84%':>17s}{'valid%':>8s}"
          f"{'SNR':>8s}{'class':>8s}{'in range':>9s}{'error':>8s}")
    for x in rows:
        rng = f"{x['p16_ms']:.3g}-{x['p84_ms']:.3g}" if np.isfinite(x["p16_ms"]) else "-"
        print(f"{x['pulse']:8s}{x['amplitude']:>6g}{x['binWidth_ms']:>7g}{('yes' if x['binLimited'] else ''):>7s}"
              f"{x['tTrue_ms']:>8.3g}{x['median_ms']:>9.3g}{rng:>17s}{x['successPct']:>8.0f}{x['snrMvt']:>8.3g}"
              f"{x['cls'] + ('*' if x['outsideCurveRange'] else ''):>8s}{('yes' if x['trueInRange'] else 'no'):>9s}"
              f"{x['relErr']:>+8.0%}")
    print("* = MVT outside the validation curve's fitted range (classification extrapolated)")

    # criteria
    c1 = [x for x in rows if not x["binLimited"] and x["cls"] == "robust"]
    c1fail = [x for x in c1 if not x["trueInRange"]]
    c2 = [x for x in rows if not x["binLimited"] and x["cls"] == "robust"]
    c2fail = [x for x in c2 if abs(x["relErr"]) > ROBUST_TOLERANCE]
    binLimitedRobust = [x for x in rows if x["binLimited"] and x["cls"] == "robust"]
    c3 = {}
    for key in truth:
        T = truth[key]["tTrue_ms"]
        ok = [x for x in rows if x["pulse"] == key and not x["binLimited"] and x["trueInRange"] and np.isfinite(x["snrMvt"])]
        lo, mid, hi = band(T)
        if ok:
            minSnr = min(x["snrMvt"] for x in ok)
            pos = "left of band" if minSnr < lo else ("inside band" if minSnr <= hi else "right of band")
            c3[key] = dict(tTrue_ms=T, minSnr=minSnr, band=[lo, mid, hi], position=pos, passed=bool(minSnr <= hi))
        else:
            c3[key] = dict(tTrue_ms=T, minSnr=float("nan"), band=[lo, mid, hi], position="never recovered", passed=False)
    c4 = {k: truth[k]["tTrue_ms"] / (PULSES[k]["sigma"] * 1e3) for k in truth if PULSES[k]["shape"] == "gaussian"}

    c5 = []
    if CHECK_PULSE in truth:
        for amp in AMPLITUDES:
            obs = [json.load(open(p)) for p in (os.path.join(OUT_DIR, "check", f"{CHECK_PULSE}_{ampTag(amp)}_obs{j}.json")
                                               for j in range(CHECK_OBS)) if os.path.exists(p)]
            main = next((pb for pb in recs.get((CHECK_PULSE, amp), {}).get("bins", []) if pb["binWidth_s"] == CHECK_BIN), None)
            if main is None or len(obs) < CHECK_OBS:
                c5.append(dict(amplitude=amp, comparable=False, reason="missing results"))
                continue
            meds = [o["median"] for o in obs if np.isfinite(o["median"])]
            succ = [o["successPct"] for o in obs]
            item = dict(amplitude=amp, mainMedian=main["median"], mainSuccess=main["successPct"], checkMedians=meds,
                        checkSuccess=succ, successDiff=main["successPct"] - float(np.mean(succ)))
            item["successOk"] = abs(item["successDiff"]) <= SUCCESS_TOLERANCE
            if np.isfinite(main["median"]) and len(meds) >= 2:
                from scipy.stats import t as studentT
                half = float(studentT.ppf(0.975, len(meds) - 1) * np.std(meds, ddof=1) / np.sqrt(len(meds)))
                item.update(comparable=True, checkMean=float(np.mean(meds)), ci95Half=half,
                            medianOk=bool(abs(main["median"] - np.mean(meds)) <= half))
            else:
                item.update(comparable=False, reason="fewer than 2 valid medians in a scheme", medianOk=None)
            c5.append(item)

    print("\n===== CRITERIA (set before running) =====")
    print(f"1. robust settings (not bin-limited) contain the true MVT: {len(c1) - len(c1fail)}/{len(c1)} -> "
          f"{'YES' if not c1fail else 'NO'}")
    for x in c1fail:
        print(f"     fails: {x['pulse']} amp {x['amplitude']:g} BW {x['binWidth_ms']:g} ms: median {x['median_ms']:.3g} "
              f"({x['p16_ms']:.3g}-{x['p84_ms']:.3g}), true {x['tTrue_ms']:.3g}")
    print(f"2. no robust setting (not bin-limited) more than {ROBUST_TOLERANCE:.0%} off: {len(c2) - len(c2fail)}/{len(c2)} -> "
          f"{'YES' if not c2fail else 'NO'}")
    for x in c2fail:
        print(f"     fails: {x['pulse']} amp {x['amplitude']:g} BW {x['binWidth_ms']:g} ms: median {x['median_ms']:.3g}, "
              f"true {x['tTrue_ms']:.3g} ({x['relErr']:+.0%})")
    print(f"   separately (not part of the criterion): bin-limited settings classed robust: {len(binLimitedRobust)}, "
          f"of which more than {ROBUST_TOLERANCE:.0%} off: {sum(abs(x['relErr']) > ROBUST_TOLERANCE for x in binLimitedRobust)}")
    for x in binLimitedRobust:
        print(f"     {x['pulse']} amp {x['amplitude']:g} BW {x['binWidth_ms']:g} ms: median {x['median_ms']:.3g}, "
              f"true {x['tTrue_ms']:.3g} ({x['relErr']:+.0%})")
    print("3. lowest working SNR inside or left of Bala et al.'s band: "
          f"{sum(v['passed'] for v in c3.values())}/{len(c3)} -> {'YES' if all(v['passed'] for v in c3.values()) else 'NO'}")
    for k, v in c3.items():
        print(f"     {k:8s} true {v['tTrue_ms']:.3g} ms: lowest SNR {v['minSnr']:.3g}; band {v['band'][0]:.3g}-"
              f"{v['band'][2]:.3g} (median {v['band'][1]:.3g}) -> {v['position']}")
    print("4. Gaussian true MVT / sigma: " + ", ".join(f"{k} {r:.3f}" for k, r in c4.items()))
    comp = [c for c in c5 if c.get("comparable")]
    c5pass = all(c["medianOk"] and c["successOk"] for c in comp) and bool(comp)
    print(f"5. real-data scheme agrees with the main scheme ({CHECK_PULSE}, {CHECK_BIN * 1e3:g} ms): "
          f"{sum(c['medianOk'] and c['successOk'] for c in comp)}/{len(comp)} comparable amplitudes -> "
          f"{'YES' if c5pass else 'NO'}")
    for c in c5:
        if c.get("comparable"):
            print(f"     amp {c['amplitude']:>5g}: main median {c['mainMedian']:.3g} vs check mean {c['checkMean']:.3g} +- "
                  f"{c['ci95Half']:.3g} (95% CI, {len(c['checkMedians'])} observations) ({'ok' if c['medianOk'] else 'NO'}); "
                  f"success {c['mainSuccess']:.0f}% vs {np.mean(c['checkSuccess']):.0f}% ({'ok' if c['successOk'] else 'NO'})")
        else:
            print(f"     amp {c['amplitude']:>5g}: not comparable ({c.get('reason')})")

    summary = dict(design=dict(pulses=PULSES, amplitudes=AMPLITUDES, brightAmplitude=BRIGHT_AMPLITUDE,
                               binWidths_s=BIN_WIDTHS, background_cps=BKG_RATE, nRealisations=N_REAL, seed=SEED),
                   truth=truth, settings=rows,
                   criteria={"1": dict(n=len(c1), failures=c1fail, passed=not c1fail),
                             "2": dict(n=len(c2), failures=c2fail, passed=not c2fail, binLimitedRobust=binLimitedRobust),
                             "3": dict(perPulse=c3, passed=all(v["passed"] for v in c3.values())),
                             "4": c4, "5": dict(perAmplitude=c5, passed=c5pass)})
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    clsColor = {"robust": "C2", "LUL": "C1", "UL": "C3", "none": "0.6"}
    fig, axes = plt.subplots(1, len(BIN_WIDTHS), figsize=(15, 5), sharey=True)
    for ax, bw in zip(axes, BIN_WIDTHS):
        sel = [x for x in rows if abs(x["binWidth_ms"] - bw * 1e3) < 1e-9 and np.isfinite(x["median_ms"])]
        for x in sel:
            ai = AMPLITUDES.index(x["amplitude"])
            xx = x["tTrue_ms"] * 1.06 ** (ai - 3)                    # small offsets so amplitudes do not overlap
            m = "o" if x["shape"] == "gaussian" else "s"
            ax.errorbar(xx, x["median_ms"], yerr=[[x["median_ms"] - x["p16_ms"]], [x["p84_ms"] - x["median_ms"]]],
                        fmt=m, color=clsColor[x["cls"]], mfc="none" if x["binLimited"] else clsColor[x["cls"]],
                        ms=5, lw=0.8, capsize=2)
        lims = np.array([0.3, 300.0])
        ax.plot(lims, lims, "k-", lw=1)
        ax.plot(lims, lims * (1 + ROBUST_TOLERANCE), "k:", lw=1)
        ax.plot(lims, lims * (1 - ROBUST_TOLERANCE), "k:", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.5, 200); ax.set_ylim(0.3, 300)
        ax.set_xlabel("true MVT (ms)")
        ax.set_title(f"bin width {bw * 1e3:g} ms ({len(sel)} settings with a median)", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.5)
    axes[0].set_ylabel("recovered median MVT (ms)")
    for c, col in clsColor.items():
        axes[0].plot([], [], "o", color=col, label=c)
    axes[0].plot([], [], "o", color="0.3", mfc="none", label="bin-limited (open)")
    axes[0].plot([], [], "o", color="0.3", label="Gaussian"); axes[0].plot([], [], "s", color="0.3", label="Norris")
    axes[0].plot([], [], "k-", label="recovered = true"); axes[0].plot([], [], "k:", label=f"+-{ROBUST_TOLERANCE:.0%}")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"P8_injection_recovery.{ext}"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    grid = np.linspace(rangeLog[0], rangeLog[1], 200)
    bands = np.array([band(10 ** g) for g in grid])
    ax.fill_betweenx(10 ** grid, bands[:, 0], bands[:, 2], color="0.85", label="Bala et al. 95% band")
    ax.plot(bands[:, 1], 10 ** grid, "k-", lw=1, label="Bala et al. median")
    keys = list(truth)
    for i, key in enumerate(keys):
        sel = [x for x in rows if x["pulse"] == key and np.isfinite(x["median_ms"]) and np.isfinite(x["snrMvt"])]
        for bw, m in zip(BIN_WIDTHS, ("o", "s", "^")):
            s2 = [x for x in sel if abs(x["binWidth_ms"] - bw * 1e3) < 1e-9]
            ax.plot([x["snrMvt"] for x in s2], [x["median_ms"] for x in s2], m, color=f"C{i}", ms=4, alpha=0.8,
                    mfc="none" if any(x["binLimited"] for x in s2) else f"C{i}")
        ax.plot([], [], "o", color=f"C{i}", label=key)
    pts = [(v["minSnr"], v["tTrue_ms"]) for v in c3.values() if np.isfinite(v["minSnr"])]
    if pts:
        ax.plot(*zip(*pts), "D", color="red", mec="k", ms=8, label="lowest working SNR (criterion 3)")
    for bw, m in zip(BIN_WIDTHS, ("o", "s", "^")):
        ax.plot([], [], m, color="0.3", label=f"{bw * 1e3:g} ms bins")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SNR_MVT"); ax.set_ylabel("MVT (ms)")
    ax.set_title("Simulated pulses on Bala et al.'s validation curve", fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"P9_injection_validation_curve.{ext}"), dpi=150)
    plt.close(fig)
    print(f"\nSaved {OUT_DIR}/summary.json and plots P8, P9 in {PLOT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["grid", "check", "all"], default="all")
    parser.add_argument("--post", action="store_true", help="only the analysis and plots from saved results")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not args.post:
        sys.path.insert(0, REPO)
        os.chdir(REPO)
        from loggerSetup import initialiseLogging
        initialiseLogging()
        import analysisTools.haarMethods as haarMethods
        commit = os.popen(f"git -C {REPO} log -1 --format='%h %s'").read().strip()
        print(f"MVTTools code in use: {commit}", flush=True)
        t0 = time.time()
        if args.part in ("grid", "all"):
            runGrid(args.workers)
        if args.part in ("check", "all"):
            runCheck(args.workers)
        print(f"Simulations finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    post()
