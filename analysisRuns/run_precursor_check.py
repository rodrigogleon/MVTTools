"""
Made-to-measure check of the GRB 230307A precursor MVT (follows run_precursor_mvt.py, sprint step 8).

Question. The precursor gave MVT = 15.2 (+2.2/-6.8) ms (6 detectors, 0.003 ms bins, real-data Monte Carlo) at
physical SNR_MVT 122, background 5753 counts/s, window 0.384 s, with 58% upper limits. If a pulse with a KNOWN MVT
is simulated in exactly these conditions and measured in exactly the same way, does the method return the right
answer, are its error bars honest, and does it give as many upper limits as the real precursor did?

The real precursor's numbers (MVT, SNR, background, detector set) are read from
results/precursor_mvt/GRB230307A/summary.json and selection.json, so this script always matches the saved result.

Part 1. Histograms of the real precursor's Monte Carlo values (6 detectors at 0.003 and 0.01 ms, all 12 at 0.003 ms)
  with the rungs of the timescale ladder (4 per doubling, rebuilt exactly as in analyse_rung_positions.py), and the
  share of valid values in each gap between rungs (to see whether the values split into separate groups).

Part 2. Background, neighbouring-bin test (follow-up of R8d). In the off-burst intervals, for the 6 detectors,
  counts in bins of 0.01, 0.1, 1, 10 and 100 ms; A = mean((n[i+1] - n[i])^2) / (2 mean(n)), which is 1 for Poisson
  noise and, unlike the Fano factor against a straight line, is almost insensitive to slow changes of the rate
  (they cancel between neighbouring bins, as they largely do in the Haar method).

Part 3. Simulations in the precursor's conditions
  Window -0.064 to 0.32 s; flat Poisson background at the measured rate (5753 counts/s, read from the summary).
  Two pulse shapes with a bright MVT near 15 ms:
    G: Gaussian, sigma = MVT / 0.72 (B1: MVT = 0.72 sigma for Gaussians), centred at 0.128 s (middle of the window)
    N: Norris, N30_100 scaled in time by MVT / 17.54 ms (its B1 true MVT), starting at 0 s
  Brightness: three levels, the amplitude chosen so that the mean physical SNR_MVT (measured at the precursor's MVT,
    as Bala et al.) is 0.5, 1 and 2 times the precursor's (61, 122, 244). Calibrated with 100 simulated observations,
    two iterations.
  Reference R: a bright run (30 times the matched amplitude, 300 observations) at the same bin width.
  Bin widths 0.01 and 0.003 ms (the plateau pair of step 8). 4 rungs per doubling.
  Two Monte Carlo schemes:
    main: 300 independent simulated observations, one MVT each (how a single measurement really scatters)
    real-data: the first 20 of those observations, each analysed exactly like the real precursor (300 Poisson
      resamples, resampleErrors=True) -> 20 medians with their 16-84% ranges
  Seeds: SeedSequence([2026, 900, shape, level]) for the observations; [2026, 901, shape, level, obs, bin] for the
  resamples.

Rules (fixed before running; matched level, 0.003 ms bins, judged for both shapes)
  C1 accuracy: the median over the 20 observations of (real-data median / R) is between 0.8 and 1.2 (within one
     ladder step) -> the precursor's MVT is not biased by more than one step; otherwise the bias factor is quoted.
  C2 honest error bars: at least 50% of the 20 real-data 16-84% ranges contain R (68% expected if exact).
  C3 resemblance: the mean real-data valid % is within 15 percentage points of the real precursor's (42%) -> the
     simple pulses behave like the real precursor; otherwise the real precursor differs (shape or background), which
     is what the denoised template (step 7) would test.
  C4 background: |A - 1| <= 0.05 at every timescale up to 10 ms (Part 2) -> the Poisson background used here is
     adequate.
  Also reported: the main-scheme median / R (the scheme comparison of criterion 5, here in precursor conditions), the
  calibration (mean SNR at the matched level within 15% of 122), and the gap shares of the simulated values.

Output: results/precursor_mvt/GRB230307A/check/ (calibration.json, sims/, summary.json); plot
results/plots/P12_precursor_check.png (.pdf too). Resumable.

Usage (from the scripts folder, next to run_injection_recovery.py and analyse_rung_positions.py)
  python run_precursor_check.py            everything
  python run_precursor_check.py --post     analysis and plot from saved results only
"""
import os, sys, json, glob, time, argparse, logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import run_injection_recovery as ir
import analyse_rung_positions as a4

PRE_DIR = os.path.join(ir.RESULTS, "precursor_mvt", "GRB230307A")
OUT = os.path.join(PRE_DIR, "check")
DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
GRB = "GRB230307A"
WINDOW = (-0.064, 0.32)
BKG_INTERVALS = [(-68.8, -18.8), (69.4, 119.4)]
ENERGY_KEV = (8.0, 900.0)
BIN_WIDTHS = [1e-5, 3e-6]
LEVELS = [0.5, 1.0, 2.0]
MATCHED = 1
BRIGHT_FACTOR = 30.0
N_OBS = 300
K_REAL = 20
N_RESAMPLE = 300
N_CAL = 100
GAUSS_CENTRE = 0.128
MVT_PER_SIGMA = 0.72
N30_100_TRUTH_MS = 17.54
ALLAN_SCALES_S = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
ALLAN_TOL = 0.05
ALLAN_JUDGED_MAX_S = 1e-2

# ---------------------------------------------------------------- the real precursor (saved by run_precursor_mvt.py)
_pre = json.load(open(os.path.join(PRE_DIR, "summary.json")))
_sel = json.load(open(os.path.join(PRE_DIR, "selection.json")))
OPT_DETS = _sel["used"]
_real = {(r["detectorSet"], r["binSize_s"]): r for r in _pre["runs"]}
REAL = _real[("opt", 3e-6)]
REAL_MVT_MS = REAL["median_ms"]
REAL_SNR = REAL["snr"]["physical"]
BKG = REAL["snr"]["bkgCps_band"]
REAL_VALID = REAL["validPercent"]

SHAPES = {
    "G": dict(shape="gaussian", sigma=REAL_MVT_MS / MVT_PER_SIGMA / 1e3),
    "N": dict(shape="norris", rise=30e-3 * REAL_MVT_MS / N30_100_TRUTH_MS, decay=100e-3 * REAL_MVT_MS / N30_100_TRUTH_MS),
}
for _k, _p in SHAPES.items():                       # registered so that ir.norrisCumulative can build the Norris shape
    ir.PULSES["PRE_" + _k] = _p
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


def simulate(key, amp, seed):
    """One observation in the precursor's window: flat background at BKG plus the pulse; peak rate = amp x BKG."""
    p = SHAPES[key]
    rng = np.random.default_rng(seed)
    nb = rng.poisson(BKG * (WINDOW[1] - WINDOW[0]))
    tb = rng.uniform(WINDOW[0], WINDOW[1], nb)
    if p["shape"] == "gaussian":
        n = rng.poisson(amp * BKG * p["sigma"] * np.sqrt(2 * np.pi))
        ts = rng.normal(GAUSS_CENTRE, p["sigma"], n)
    else:
        grid, cum = ir.norrisCumulative("PRE_" + key)
        n = rng.poisson(amp * BKG / ir.BKG_RATE * cum[-1])
        ts = np.interp(rng.uniform(0.0, cum[-1], n), cum, grid)
    ts = ts[(ts >= WINDOW[0]) & (ts < WINDOW[1])]
    return np.sort(np.concatenate([tb, ts])), int(nb)


def _snrTask(args):
    key, amp, seed, mvtS = args
    times, nb = simulate(key, amp, seed)
    return ir.snrAt(times, nb, WINDOW[0], WINDOW[1], mvtS)


def _mvtTask(args):
    global hm
    if hm is None:
        hm = loadHaar()
    key, amp, seed = args
    times, nb = simulate(key, amp, seed)
    out = []
    for bw in BIN_WIDTHS:
        c = ir.lightCurve(times, WINDOW[0], WINDOW[1], bw)
        try:
            r = hm.haarPowerMod(c, np.sqrt(c), minimumBinSizeSeconds=bw)
            out.append((float(r[2]), float(r[3])))
        except Exception:
            out.append((float("nan"), float("nan")))
    return out


def seedsFor(si, li):
    return np.random.SeedSequence([ir.SEED, 900, si, li]).spawn(N_OBS)


# ---------------------------------------------------------------- simulations
def calibrate(pool):
    path = os.path.join(OUT, "calibration.json")
    if os.path.exists(path):
        return json.load(open(path))
    cal = {}
    for si, key in enumerate(SHAPES):
        seeds = np.random.SeedSequence([ir.SEED, 899, si]).spawn(N_CAL)
        amp, hist = 10.0, []
        for it in range(3):
            snr = np.array(list(pool.map(_snrTask, [(key, amp, s, REAL_MVT_MS / 1e3) for s in seeds], chunksize=4)))
            m = float(np.mean(snr))
            hist.append(dict(amp=amp, meanSnr=m))
            print(f"calibration {key}: amplitude {amp:.4g} -> mean SNR {m:.1f} (target {REAL_SNR:.1f})", flush=True)
            if it < 2:
                amp *= REAL_SNR / m
        cal[key] = dict(matchedAmp=hist[-1]["amp"], history=hist)
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cal, f, indent=1)
    return cal


def mainScheme(pool, key, si, li, amp):
    tag = "bright" if li == 9 else f"L{LEVELS[li]:g}"
    path = os.path.join(OUT, "sims", f"{key}_{tag}_main.json")
    if os.path.exists(path):
        print(f"{key} {tag} main: already done, reused")
        return json.load(open(path))
    t0 = time.time()
    seeds = seedsFor(si, li)
    res = list(pool.map(_mvtTask, [(key, amp, s) for s in seeds], chunksize=4))
    out = dict(shape=key, level=tag, amplitude=amp, bins={})
    for j, bw in enumerate(BIN_WIDTHS):
        mvt = [r[j][0] for r in res]
        unc = [r[j][1] for r in res]
        s = ir.summarise(mvt, unc)
        med = s["median"] / 1e3 if np.isfinite(s["median"]) else None
        snr = np.array(list(pool.map(_snrTask, [(key, amp, sd, med) for sd in seeds], chunksize=4))) if med else np.array([])
        s["snrMvt"] = float(np.nanmean(snr)) if snr.size else float("nan")
        out["bins"][f"{bw:g}"] = dict(binWidth_s=bw, mvt_ms=[v * 1e3 for v in mvt], unc_ms=[v * 1e3 for v in unc], **s)
        print(f"{key} {tag:6s} main  {bw * 1e3:g} ms: median {s['median']:.3g} ms ({s['p16']:.3g}-{s['p84']:.3g}), "
              f"valid {s['successPct']:.0f}%, SNR {s['snrMvt']:.1f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    return out


def realScheme(key, si, li, amp):
    """The first K_REAL observations, each analysed as the real precursor (Poisson resamples of one light curve)."""
    global hm
    if hm is None:
        hm = loadHaar()
    tag = f"L{LEVELS[li]:g}"
    seeds = seedsFor(si, li)[:K_REAL]
    rows = []
    for o, sd in enumerate(seeds):
        path = os.path.join(OUT, "sims", f"{key}_{tag}_real_obs{o:02d}.json")
        if os.path.exists(path):
            rows.append(json.load(open(path)))
            continue
        times, nb = simulate(key, amp, sd)
        row = dict(obs=o, bins={})
        for j, bw in enumerate(BIN_WIDTHS):
            c = ir.lightCurve(times, WINDOW[0], WINDOW[1], bw).astype(float)
            r = hm.estimateMvtDistribution(c, np.sqrt(c), hm.haarPowerMod, binSizeSeconds=bw, nRealisations=N_RESAMPLE,
                                           nWorkers=args.workers, seed=[ir.SEED, 901, si, li, o, j], resampleErrors=True)
            row["bins"][f"{bw:g}"] = dict(median=r["median"] * 1e3, p16=r["p16"] * 1e3, p84=r["p84"] * 1e3,
                                         validPct=100.0 * r["nValid"] / r["nRealisations"],
                                         values_ms=[v * 1e3 for v in r["mvtValues"]],
                                         valid=[bool(v) for v in r["validMask"]])
        with open(path, "w") as f:
            json.dump(row, f, default=float)
        b = row["bins"][f"{3e-6:g}"]
        print(f"{key} {tag:6s} real-data obs {o:2d}: 0.003 ms median {b['median']:.3g} ({b['p16']:.3g}-{b['p84']:.3g}) ms, "
              f"valid {b['validPct']:.0f}%", flush=True)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- real data: histograms and background
def realValues(setName, bw):
    d = os.path.join(PRE_DIR, f"{setName}_{bw * 1e3:g}ms")
    v = np.load(os.path.join(d, "mvt_values_seconds.npy")) * 1e3
    u = np.load(os.path.join(d, "mvt_uncertainties_seconds.npy"))
    nBins = int(np.load(os.path.join(d, "counts.npy")).size)
    valid = np.isfinite(v) & (v > 0) & (u > 0)
    return v[valid], nBins


def gapShares(values, nBins, bw, minShare=0.10):
    """Share of the valid values in each gap between rungs (only gaps with at least minShare)."""
    R = a4.rungsFast(nBins, bw)
    v = values[(values > R[0]) & (values < R[-1])]
    if v.size == 0:
        return [], R
    i = np.searchsorted(R, v) - 1
    ids, counts = np.unique(i, return_counts=True)
    out = [dict(lo=float(R[k]), hi=float(R[k + 1]), share=float(c / v.size)) for k, c in zip(ids, counts) if c / v.size >= minShare]
    return sorted(out, key=lambda x: -x["share"]), R


def allan(dets):
    from gdt.missions.fermi.gbm.tte import GbmTte
    times = []
    for det in dets:
        files = glob.glob(f"{DATA_ROOT}/{GRB}/current/glg_tte_{det}_bn*_v0*.fit")
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
        tte = GbmTte.open(files[0])
        t = np.array(tte.slice_energy(ENERGY_KEV).data.times)
        tte.close()
        keep = np.zeros(t.size, bool)
        for a, b in BKG_INTERVALS:
            keep |= (t >= a) & (t < b)
        times.append(t[keep])
    times = np.concatenate(times)
    out = {}
    for tau in ALLAN_SCALES_S:
        num = den = 0.0
        nd = 0
        for a, b in BKG_INTERVALS:
            nBins = int(np.floor((b - a) / tau + 1e-9))
            n = np.histogram(times, bins=a + tau * np.arange(nBins + 1))[0].astype(float)
            num += np.sum(np.diff(n) ** 2)
            den += np.sum(n)
            nd += n.size - 1
        meanN = den / (nd + len(BKG_INTERVALS))
        A = (num / nd) / (2.0 * meanN)
        out[tau] = dict(A=float(A), err=float(np.sqrt((1.0 / (2.0 * meanN) + 2.0) / nd)), meanCountsPerBin=float(meanN))
    return out


# ---------------------------------------------------------------- analysis
def post(cal, sims):
    rep = dict(real=dict(mvt_ms=REAL_MVT_MS, snr=REAL_SNR, bkg=BKG, validPct=REAL_VALID, dets=OPT_DETS),
               shapes={k: dict(SHAPES[k]) for k in SHAPES}, calibration=cal)

    print("\n===== PART 1. REAL PRECURSOR: MONTE CARLO VALUES AND THE LADDER =====")
    rep["realGaps"] = {}
    for setName, bw in (("opt", 3e-6), ("opt", 1e-5), ("all", 3e-6)):
        v, nBins = realValues(setName, bw)
        g, _ = gapShares(v, nBins, bw)
        rep["realGaps"][f"{setName}_{bw:g}"] = g
        print(f"{setName} {bw * 1e3:g} ms: {v.size} valid values; gaps holding >= 10%: "
              + ", ".join(f"{x['lo']:.2f}-{x['hi']:.2f} ms {100 * x['share']:.0f}%" for x in g))

    print("\n===== PART 2. BACKGROUND, NEIGHBOURING-BIN TEST (off-burst, " + " ".join(OPT_DETS) + ") =====")
    al = allan(OPT_DETS)
    c4 = True
    for tau, a in al.items():
        judged = tau <= ALLAN_JUDGED_MAX_S
        good = abs(a["A"] - 1) <= ALLAN_TOL
        c4 &= good or not judged
        print(f"  {tau * 1e3:>7g} ms: A = {a['A']:.4f} ± {a['err']:.4f}  ({a['meanCountsPerBin']:.3g} counts/bin)"
              f"{'' if judged else '  [reported only]'}{'' if good else '  <- outside 1 ± 0.05'}")
    print(f"  C4: {'Poisson background adequate' if c4 else 'NOT Poisson'} up to {ALLAN_JUDGED_MAX_S * 1e3:g} ms")
    rep["allan"] = {str(k): v for k, v in al.items()}
    rep["C4"] = c4

    print("\n===== PART 3. SIMULATIONS IN THE PRECURSOR'S CONDITIONS =====")
    print(f"real precursor: MVT {REAL_MVT_MS:.2f} ms, physical SNR {REAL_SNR:.1f}, background {BKG:.0f} c/s, valid {REAL_VALID:.0f}%,"
          f" detectors {OPT_DETS}")
    for key in SHAPES:
        p = SHAPES[key]
        desc = (f"Gaussian sigma {p['sigma'] * 1e3:.2f} ms" if p["shape"] == "gaussian"
                else f"Norris rise {p['rise'] * 1e3:.2f} ms, decay {p['decay'] * 1e3:.2f} ms")
        print(f"{key}: {desc}; calibration: " + ", ".join(f"amp {h['amp']:.3g} -> SNR {h['meanSnr']:.1f}" for h in cal[key]["history"]))
    rep["results"] = {}
    verdicts = {}
    for key in SHAPES:
        bright = sims[key]["bright"]
        rep["results"][key] = {}
        for bw in BIN_WIDTHS:
            bk = f"{bw:g}"
            R = bright["bins"][bk]["median"]
            print(f"\n{key}, {bw * 1e3:g} ms bins: reference R (bright, x{BRIGHT_FACTOR:g}) = {R:.3f} ms "
                  f"({bright['bins'][bk]['successPct']:.0f}% valid)")
            print(f"{'level':>6s}{'amp':>8s}{'SNR':>7s} | {'main: median/R':>15s}{'16-84%/R':>14s}{'valid%':>7s} | "
                  f"{'real-data: median of medians/R':>31s}{'obs with R in 16-84%':>22s}{'mean valid%':>12s}")
            rows = {}
            for li, lv in enumerate(LEVELS):
                m = sims[key]["main"][li]["bins"][bk]
                rd = sims[key]["real"][li]
                meds = np.array([o["bins"][bk]["median"] for o in rd])
                cover = np.mean([o["bins"][bk]["p16"] <= R <= o["bins"][bk]["p84"] for o in rd])
                vpct = float(np.mean([o["bins"][bk]["validPct"] for o in rd]))
                ratio = float(np.nanmedian(meds / R))
                spread = np.nanpercentile(meds / R, [16, 84])
                rows[lv] = dict(amp=sims[key]["main"][li]["amplitude"], snr=m["snrMvt"], mainRatio=m["median"] / R,
                                mainRange=[m["p16"] / R, m["p84"] / R], mainValid=m["successPct"],
                                realRatio=ratio, realRatioSpread=list(spread), realCover=float(cover), realValid=vpct)
                print(f"{lv:>6g}{rows[lv]['amp']:>8.3g}{m['snrMvt']:>7.1f} | {m['median'] / R:>15.3f}"
                      f"{m['p16'] / R:>7.2f}-{m['p84'] / R:<6.2f}{m['successPct']:>7.0f} | "
                      f"{ratio:>14.3f} (obs 16-84%: {spread[0]:.2f}-{spread[1]:.2f}){100 * cover:>13.0f}%{vpct:>12.0f}")
            rep["results"][key][bk] = dict(R=R, rows=rows)
            # gap shares of the simulated values at the matched level (main scheme)
            mv = np.array(sims[key]["main"][MATCHED]["bins"][bk]["mvt_ms"])
            mu = np.array(sims[key]["main"][MATCHED]["bins"][bk]["unc_ms"])
            nBins = int(np.floor((WINDOW[1] - WINDOW[0]) / bw + 1e-9))
            g, _ = gapShares(mv[np.isfinite(mv) & (mv > 0) & (mu > 0)], nBins, bw)
            rv = np.concatenate([np.array(o["bins"][bk]["values_ms"])[np.array(o["bins"][bk]["valid"])] for o in sims[key]["real"][MATCHED]])
            gr, _ = gapShares(rv, nBins, bw)
            print("   matched level, gaps holding >= 10%: main " + ", ".join(f"{x['lo']:.2f}-{x['hi']:.2f} {100 * x['share']:.0f}%" for x in g)
                  + " | real-data (pooled) " + ", ".join(f"{x['lo']:.2f}-{x['hi']:.2f} {100 * x['share']:.0f}%" for x in gr))
        mt = rep["results"][key][f"{3e-6:g}"]["rows"][LEVELS[MATCHED]]
        c1 = 0.8 <= mt["realRatio"] <= 1.2
        c2 = mt["realCover"] >= 0.5
        c3 = abs(mt["realValid"] - REAL_VALID) <= 15
        calOk = abs(mt["snr"] / REAL_SNR - 1) <= 0.15
        verdicts[key] = dict(C1=c1, C2=c2, C3=c3, calibrationOk=calOk, biasFactor=mt["realRatio"],
                             mainRatio=mt["mainRatio"], cover=mt["realCover"], validSim=mt["realValid"])

    print("\n===== RULES (matched level, 0.003 ms) =====")
    for key, v in verdicts.items():
        print(f"{key}: C1 accuracy: real-data median/R = {v['biasFactor']:.3f} -> {'PASS' if v['C1'] else 'FAIL'}   "
              f"(main scheme {v['mainRatio']:.3f})")
        print(f"   C2 error bars: R inside the 16-84% range in {100 * v['cover']:.0f}% of {K_REAL} observations -> "
              f"{'PASS' if v['C2'] else 'FAIL'}")
        print(f"   C3 resemblance: simulated valid {v['validSim']:.0f}% vs real {REAL_VALID:.0f}% -> {'PASS' if v['C3'] else 'FAIL'}")
        print(f"   calibration: SNR at the matched level within 15% of {REAL_SNR:.0f} -> {'yes' if v['calibrationOk'] else 'NO'}")
    print(f"C4 background: {'PASS' if rep['C4'] else 'FAIL'}")
    if all(v["C1"] for v in verdicts.values()):
        f = [v["biasFactor"] for v in verdicts.values()]
        print(f"-> The method returns the right MVT within one ladder step in these conditions (bias factor "
              f"{min(f):.2f}-{max(f):.2f}); the precursor's {REAL_MVT_MS:.1f} ms corresponds to a true MVT of about "
              f"{REAL_MVT_MS / max(f):.1f}-{REAL_MVT_MS / min(f):.1f} ms for these shapes.")
    else:
        print("-> At least one shape is biased by more than one ladder step: quote the bias factor with the precursor's MVT.")
    rep["verdicts"] = verdicts
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"\nSaved {OUT}/summary.json")
    plot(sims, rep)


def plot(sims, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(ir.PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    bw, bk = 3e-6, f"{3e-6:g}"
    nBins = int(np.floor((WINDOW[1] - WINDOW[0]) / bw + 1e-9))
    edges = np.logspace(np.log10(3), np.log10(40), 60)

    a = ax[0, 0]
    v, nb = realValues("opt", bw)
    a.hist(v, bins=edges, color="k", histtype="step", lw=1.2, label=f"real precursor, {len(OPT_DETS)} detectors ({v.size} valid)")
    for r in a4.rungsFast(nb, bw):
        if edges[0] <= r <= edges[-1]:
            a.axvline(r, color="0.6", lw=0.6, ls=":")
    a.axvline(REAL_MVT_MS, color="C3", lw=1, label=f"median {REAL_MVT_MS:.1f} ms")
    a.set_xscale("log"); a.set_xlabel("MVT (ms)"); a.set_ylabel("realisations")
    a.set_title("Real precursor: Monte Carlo values (0.003 ms), rungs dotted", fontsize=10)
    a.legend(fontsize=8)

    a = ax[0, 1]
    for key, c in (("G", "C0"), ("N", "C1")):
        mv = np.array(sims[key]["main"][MATCHED]["bins"][bk]["mvt_ms"])
        mu = np.array(sims[key]["main"][MATCHED]["bins"][bk]["unc_ms"])
        a.hist(mv[np.isfinite(mv) & (mv > 0) & (mu > 0)], bins=edges, histtype="step", color=c, lw=1.2,
               label=f"{key}, main scheme (R = {rep['results'][key][bk]['R']:.1f} ms)")
        a.axvline(rep["results"][key][bk]["R"], color=c, lw=1, ls="--")
    for r in a4.rungsFast(nBins, bw):
        if edges[0] <= r <= edges[-1]:
            a.axvline(r, color="0.6", lw=0.6, ls=":")
    a.set_xscale("log"); a.set_xlabel("MVT (ms)"); a.set_ylabel("observations")
    a.set_title("Simulated pulses at the precursor's SNR (0.003 ms); dashed = R", fontsize=10)
    a.legend(fontsize=8)

    a = ax[1, 0]
    for key, c, dx in (("G", "C0", -0.15), ("N", "C1", 0.15)):
        R = rep["results"][key][bk]["R"]
        for o in sims[key]["real"][MATCHED]:
            b = o["bins"][bk]
            a.errorbar(o["obs"] + dx, b["median"] / R, yerr=[[b["median"] / R - b["p16"] / R], [b["p84"] / R - b["median"] / R]],
                       fmt="o", color=c, ms=4, capsize=2, lw=0.8)
        a.plot([], [], "o", color=c, label=f"{key}: real-data median and 16-84% per observation")
    a.axhline(1, color="k", lw=0.8)
    a.axhspan(0.8, 1.2, color="0.9", zorder=0)
    a.set_xlabel("simulated observation"); a.set_ylabel("MVT / R")
    a.set_title("Analysed like the real precursor (300 resamples each), matched SNR", fontsize=10)
    a.legend(fontsize=8)

    a = ax[1, 1]
    for key, c in (("G", "C0"), ("N", "C1")):
        rows = rep["results"][key][bk]["rows"]
        x = [rows[lv]["snr"] for lv in LEVELS]
        a.plot(x, [rows[lv]["mainRatio"] for lv in LEVELS], "o-", color=c, mfc="none", label=f"{key}, main scheme")
        a.plot(x, [rows[lv]["realRatio"] for lv in LEVELS], "s-", color=c, label=f"{key}, real-data scheme")
    a.axvline(REAL_SNR, color="C3", lw=1, label=f"real precursor SNR {REAL_SNR:.0f}")
    a.axhline(1, color="k", lw=0.8)
    a.axhspan(0.8, 1.2, color="0.9", zorder=0)
    a.set_xscale("log"); a.set_xlabel("SNR_MVT"); a.set_ylabel("median MVT / R")
    a.set_title("Bias against brightness (0.003 ms)", fontsize=10)
    a.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ir.PLOT_DIR, f"P12_precursor_check.{ext}"), dpi=130)
    plt.close(fig)
    print(f"Saved plot {ir.PLOT_DIR}/P12_precursor_check.png (.pdf too)")


def loadSims():
    cal = json.load(open(os.path.join(OUT, "calibration.json")))
    sims = {}
    for key in SHAPES:
        sims[key] = dict(bright=json.load(open(os.path.join(OUT, "sims", f"{key}_bright_main.json"))), main=[], real=[])
        for li, lv in enumerate(LEVELS):
            sims[key]["main"].append(json.load(open(os.path.join(OUT, "sims", f"{key}_L{lv:g}_main.json"))))
            sims[key]["real"].append([json.load(open(os.path.join(OUT, "sims", f"{key}_L{lv:g}_real_obs{o:02d}.json")))
                                      for o in range(K_REAL)])
    return cal, sims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "sims"), exist_ok=True)
    if not args.post:
        commit = os.popen(f"git -C {ir.REPO} log -1 --format='%h %s'").read().strip()
        print(f"MVTTools code in use: {commit}", flush=True)
        print(f"real precursor read from {PRE_DIR}: MVT {REAL_MVT_MS:.2f} ms, SNR {REAL_SNR:.1f}, background {BKG:.0f} c/s,"
              f" valid {REAL_VALID:.0f}%, detectors {OPT_DETS}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cal = calibrate(pool)
            for si, key in enumerate(SHAPES):
                amp0 = cal[key]["matchedAmp"]
                mainScheme(pool, key, si, 9, amp0 * BRIGHT_FACTOR)
                for li, lv in enumerate(LEVELS):
                    mainScheme(pool, key, si, li, amp0 * lv)
        for si, key in enumerate(SHAPES):
            amp0 = cal[key]["matchedAmp"]
            for li in (MATCHED, 0, 2):
                realScheme(key, si, li, amp0 * LEVELS[li])
        print(f"Simulations finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    cal, sims = loadSims()
    post(cal, sims)
