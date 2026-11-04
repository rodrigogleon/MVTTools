"""
Step 7: made-to-measure check of the GRB 230307A precursor with a template of its REAL shape (follows
run_precursor_check.py and run_precursor_artefacts.py).

Question. Simple fake pulses (one Gaussian or one Norris pulse) in the precursor's conditions are measured correctly,
but give more valid values (61-72%) than the real precursor (42%), and data gaps, dead time and the window do not
explain the difference. The remaining suspect is the shape: the real precursor has two peaks and a long tail. Here the
real light curve is smoothed with the Haar denoiser of MVTTools (haarDenoise, Derek's suggestion) into a noise-free
template, fake observations are drawn from that template, and they are analysed exactly like the real precursor.
Before the template is trusted, the same procedure is applied to fake pulses whose answer is known (7a).

Trust definition (agreed 3 Nov 2026, used for every rule below)
  A measurement is trusted when, for fakes in the same conditions,
    (i)  the median MVT is within one ladder step of the truth: between truth / 1.2 and truth x 1.2, and
    (ii) the truth lies inside the 16-84% range in at least 50% of the fakes.

Template (the same for known pulses and for the real data)
  Counts of the 6 detectors (8-900 keV) in 1 ms bins over -0.192 to 0.448 s (the precursor window -0.064 to 0.32 s
  plus 128 ms on each side, so that the edges of the denoiser are far from the window); haarDenoise with errors
  sqrt(max(counts, 1)), hard threshold, and a threshold factor f; the result divided by 1 ms is the template rate.
  Threshold factor: f = 1 is the code's default, but a local test showed that it sometimes keeps noise bumps, which then
  pose as fast variability (template MVT 0.4-0.6 of the truth), while a stricter threshold removes them; too strict a
  threshold would smooth away real structure. So f is chosen by a fixed rule: f = 1, 1.5, 2, 3 are tried in this order
  and the first one that passes 7a (A1 and A2 below, all three known pulses) is used for the real precursor.
  Source part = max(template rate - background, 0), background = the precursor's fitted background (5753 counts/s,
  read from step 8). Fakes: background photons uniform at that rate plus source photons drawn from the
  source part interpolated linearly between the 1 ms bin centres (sampled on a 1 microsecond grid, so the fakes have
  no steps at the 1 ms bins). haarDenoise uses a non-decimated Haar transform, so averaging over shifted copies
  ("cycle spinning") gives the same template (checked before writing this script) and is not used.
  Truth of a template, R_tpl: the median MVT (0.003 ms bins) of fakes with the source part x30 (as R in step 2): 100
  fakes for each 7a template, 300 for the real precursor's.
  A template made from noisy data keeps some faint fine structure (leftover noise, and the corners where it is joined
  up every 1 ms), invisible at the real brightness but visible once brightened; then R_tpl depends on the boost. For
  the real precursor R_tpl is therefore also measured at x10, and counts as defined only if the two agree within one
  step; if not, the rules that need R_tpl (B1, B2, C1, C2) are reported as not judged.

Part T. Known pulses in the precursor's conditions (window, background 5753 counts/s, SNR_MVT 122 at 15.2 ms)
  G, N: the two pulses of step 2 (Gaussian sigma 21.1 ms centred at 0.128 s; Norris rise 26.0 ms, decay 86.8 ms from
        0 s); their amplitude, truth R, and 20 real-data-scheme fakes are reused from step 2 (check/).
  D:    a double pulse like the real precursor: two Norris pulses (rise 13.0 ms, decay 43.4 ms, half of N's) with
        equal peaks at 0.020 and 0.136 s. Amplitude calibrated to SNR_MVT 122 at 15.2 ms (3 iterations, 100 fakes);
        truth R from 300 fakes at x30; 20 fakes analysed like the real precursor (300 resamples each).
Part A (7a). Does the template keep the MVT? For each of G, N, D, 20 single observations at the matched brightness,
  each turned into a template, then:
  A1 (MVT kept): R_tpl (from 100 fakes at x30) / R. Rule: median over the 20 within 1/1.2-1.2 AND at least 14 of 20
     within 1/1.2-1.2.
  A2 (fakes behave the same): one fake from each template at the matched brightness, analysed like the real precursor.
     Rules: (a) median over the 20 of (median / R_tpl) within 1/1.2-1.2; (b) R_tpl inside the 16-84% range in at least
     50%; (c) mean valid share within max(10, 2 standard errors) points of the 20 truth fakes of the same pulse.
  7a passes for a threshold factor if A1 and A2 pass for all three pulses (A2 is run only if A1 passes). The factors
  are tried in the order 1, 1.5, 2, 3; the first that passes is used for Parts B and C. If none passes, the script
  stops: the denoised template cannot be trusted at the precursor's brightness.
  Reported: chi2 per bin of observation against template (1 = the template keeps only the signal; below 1 it also
  keeps some noise), and for D its own C1-C3 against the real precursor (does a known double pulse give 42%?).
Part B (7b). The real precursor
  Z0, Z1 regression: the summed counts in the window, without and with the gap filled exactly as run R1 seed 0 of
     run_precursor_artefacts.py, equal that run's saved totals (R0 and R1); otherwise the script stops.
  Template from the gap-filled data (official); a template from the unfilled data is also made (its R_tpl reported).
  20 fakes at the precursor's brightness (template as it is), each analysed like the real precursor, in 3 variants
  with the same photons and resampling seeds (paired):
     clean    no gap
     gap      the real gaps (T0 + 3.6-5.0 ms, n0 n6 n7 na) applied as in run_precursor_artefacts.py
     gapfill  the gaps applied and then filled as there
  B0 sanity: mean SNR_MVT (at 15.2 ms) of the clean fakes within 15% of 122.
  Bd R_tpl defined: R_tpl at x10 and x30 within one step of each other (B1, B2, C1, C2 judged only if so).
  B1 accuracy (clean): median of median / R_tpl within 1/1.2-1.2.
  B2 honest error bars (clean): R_tpl inside the 16-84% range in at least 50%.
  B3 does the shape explain C3: 'gap' mean valid share within 15 points of the real precursor (42%, run R0) AND
     'clean' within 15 points of the gap-filled real precursor (mean of R1 seeds 0-2, 0.003 ms).
  B4 the gap (paired gap - clean): it matters if the mean change of the median, as a fraction of the precursor's MVT
     (15.2 ms), is beyond max(0.05, 2 SE) or that of the valid share beyond max(5, 2 SE) points; filling undoes it if
     gapfill - clean is within the same limits. (Independent of R_tpl.)
  Adoption (fixed before running; decides how the pipeline treats gaps in every window from now on)
     G1 gap matters and filling undoes it -> the pipeline fills gaps; the precursor's official value becomes run R1
        (gap filled, seed 0, 6 detectors, 0.003 ms) with the fill-seed spread as a systematic, R0 quoted as a check.
     G2 gap matters and filling does not undo it -> windows with gaps are flagged and the gap shift quoted as a
        systematic; the official value stays R0.
     G3 gap does not matter -> gaps are flagged only; the official value stays R0.
Part C (7c). Fading the real shape: the source part of the (gap-filled) template scaled by 1.2^k, k = -13 ... 4
  (0.09 to 2.1 times the precursor), 300 fakes each (one MVT each, as B3/B5).
  C1 threshold (as B3, with the trust definition (i)): the faintest level from which the median (0.003 ms) is within
     1/1.2-1.2 of R_tpl at that level and every brighter one; its mean SNR_MVT (at that level's median MVT, as Bala et
     al.) is the threshold for the real shape.
  C2 the same fakes at the threshold level, one level fainter and the brightest level (k = 4), first 20 analysed like
     the real precursor: R_tpl inside the 16-84% range in at least 50% at the threshold -> threshold confirmed under
     the full trust definition (i)+(ii).
  Reported: Bala et al.'s rule (lowest SNR whose 16-84% contains R_tpl), their 95% band at R_tpl, and the B5
  precursor-condition thresholds of simple pulses with R within a factor 2.

Output: results/precursor_mvt/GRB230307A/template/ (truth/; a_f<factor>/, b_f<factor>/, c_f<factor>/; summary.json); plot
results/plots/P14_precursor_template.png (.pdf too). Resumable.

Usage (from the scripts folder, next to run_injection_recovery.py, run_precursor_check.py, run_precursor_artefacts.py)
  python run_precursor_template.py            everything
  python run_precursor_template.py --post     analysis and plot from saved results only
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import run_injection_recovery as ir
import run_precursor_check as pc
import run_precursor_artefacts as pa

OUT = os.path.join(pc.PRE_DIR, "template")
WINDOW = pc.WINDOW
DEN_REGION = (-0.192, 0.448)
DEN_BIN = 1e-3
GRID_S = 1e-6
BIN_WIDTHS = [1e-5, 3e-6]
BK = f"{3e-6:g}"
STEP = 1.2
SEED = ir.SEED
K = pc.K_REAL
N_OBS = pc.N_OBS
N_RESAMPLE = pc.N_RESAMPLE
N_CAL = pc.N_CAL
N_BRIGHT_TPL = 100
BRIGHT = pc.BRIGHT_FACTOR
BRIGHT_LOW = 10.0
MIN_WITHIN = 14
COVER_MIN = 0.5
LEVEL_K = list(range(-13, 5))
B5_SUMMARY = os.path.join(ir.OUT_DIR, "b3_threshold", "summary.json")
SHAPE_KEYS = ["G", "N", "D"]
FACTORS = [1.0, 1.5, 2.0, 3.0]

BKG = pc.BKG
REAL_MVT_MS, REAL_SNR, REAL_VALID = pc.REAL_MVT_MS, pc.REAL_SNR, pc.REAL_VALID
N_RISE, N_DECAY = pc.SHAPES["N"]["rise"], pc.SHAPES["N"]["decay"]
D_RISE, D_DECAY = N_RISE / 2, N_DECAY / 2
D_PEAKS = (0.020, 0.136)
D_STARTS = tuple(p - np.sqrt(D_RISE * D_DECAY) for p in D_PEAKS)
hm = None


def loadHaar():
    global hm
    if hm is None:
        hm = pc.loadHaar()
    return hm


def fdir(part, f):
    return os.path.join(OUT, f"{part}_f{f:g}")


def within(x):
    return np.isfinite(x) and 1 / STEP <= x <= STEP


# ================================================================ photons
def norris(t, t0, rise, decay):
    out = np.zeros_like(t)
    m = t > t0
    x = t[m] - t0
    out[m] = np.exp(2 * np.sqrt(rise / decay)) * np.exp(-rise / x - x / decay)
    return out


def shapeRate(key, t):
    """Source rate per unit amplitude: peak 1 (times amplitude x background gives counts/s)."""
    if key == "G":
        return np.exp(-0.5 * ((t - pc.GAUSS_CENTRE) / pc.SHAPES["G"]["sigma"]) ** 2)
    if key == "N":
        return norris(t, 0.0, N_RISE, N_DECAY)
    return norris(t, D_STARTS[0], D_RISE, D_DECAY) + norris(t, D_STARTS[1], D_RISE, D_DECAY)


def centres():
    n = int(np.floor((DEN_REGION[1] - DEN_REGION[0]) / DEN_BIN + 1e-9))
    return DEN_REGION[0] + DEN_BIN * (np.arange(n) + 0.5)


def rateFn(spec):
    """spec = ('shape', key, amplitude) or ('tpl', source rate at the 1 ms centres (counts/s), scale)."""
    if spec[0] == "shape":
        _, key, amp = spec
        return lambda t: amp * BKG * shapeRate(key, t)
    _, src, scale = spec
    c, s = centres(), np.asarray(src)
    return lambda t: scale * np.interp(t, c, s)


def samplePhotons(rng, rate, lo, hi):
    """Poisson photon times with rate(t) (counts/s) on [lo, hi): inverse cumulative on a GRID_S grid."""
    g = lo + GRID_S * np.arange(int(round((hi - lo) / GRID_S)) + 1)
    r = np.clip(rate(g), 0.0, None)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1])) * GRID_S])
    n = rng.poisson(cum[-1])
    if n == 0:
        return np.empty(0)
    u = rng.uniform(0.0, cum[-1], n)
    j = np.clip(np.searchsorted(cum, u, "right") - 1, 0, g.size - 2)
    width = cum[j + 1] - cum[j]
    frac = np.where(width > 0, (u - cum[j]) / np.where(width > 0, width, 1.0), 0.5)
    t = g[j] + frac * GRID_S
    return t[(t >= lo) & (t < hi)]


def observe(spec, seed, lo, hi):
    rng = np.random.default_rng(seed)
    nb = rng.poisson(BKG * (hi - lo))
    tb = rng.uniform(lo, hi, nb)
    ts = samplePhotons(rng, rateFn(spec), lo, hi)
    return np.sort(np.concatenate([tb, ts])), int(nb)


def buildTemplate(counts, f):
    """Denoised 1 ms light curve (threshold factor f) -> template rate, source part (counts/s), chi2 per bin."""
    h = loadHaar()
    counts = np.asarray(counts, float)
    d = h.haarDenoise(counts, np.sqrt(np.maximum(counts, 1.0)), thresholdFactor=f)
    rate = d / DEN_BIN
    src = np.clip(rate - BKG, 0.0, None)
    chi = float(np.sum((counts - d) ** 2 / np.maximum(d, 1.0)) / counts.size)
    return rate, src, chi


# ================================================================ workers and runs
def _mvtTask(a):
    spec, seed = a
    h = loadHaar()
    times, nb = observe(spec, seed, *WINDOW)
    out = []
    for bw in BIN_WIDTHS:
        c = ir.lightCurve(times, WINDOW[0], WINDOW[1], bw)
        try:
            r = h.haarPowerMod(c, np.sqrt(c), minimumBinSizeSeconds=bw)
            out.append((float(r[2]), float(r[3])))
        except Exception:
            out.append((float("nan"), float("nan")))
    return out


def _snrTask(a):
    spec, seed, mvtS = a
    times, nb = observe(spec, seed, *WINDOW)
    return ir.snrAt(times, nb, WINDOW[0], WINDOW[1], mvtS)


def mainRun(pool, path, spec, seeds, label, withSnr=False):
    """One MVT per fake (main scheme), both bin widths; optional mean SNR_MVT at the 0.003 ms median."""
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    res = list(pool.map(_mvtTask, [(spec, s) for s in seeds], chunksize=4))
    out = dict(label=label, n=len(seeds), bins={})
    for j, bw in enumerate(BIN_WIDTHS):
        out["bins"][f"{bw:g}"] = ir.summarise([r[j][0] for r in res], [r[j][1] for r in res])
    b = out["bins"][BK]
    if withSnr and np.isfinite(b["median"]):
        b["snr"] = float(np.nanmean(list(pool.map(_snrTask, [(spec, s, b["median"] / 1e3) for s in seeds], chunksize=4))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    print(f"  {label:28s} main: median {b['median']:.3g} ms ({b['p16']:.3g}-{b['p84']:.3g}), valid {b['successPct']:.0f}%"
          + (f", SNR {b['snr']:.1f}" if "snr" in b else "") + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    return out


def realObs(path, spec, obsSeed, rsSeed, label, artef=None):
    """One fake analysed exactly like the real precursor (300 Poisson resamples of its light curve)."""
    if os.path.exists(path):
        return json.load(open(path))
    h = loadHaar()
    times, nb = observe(spec, obsSeed, *WINDOW)
    if artef is not None:
        gaps, correct, aseed = artef
        times = pa.artefacts(times, PARAMS, gaps, False, correct, np.random.default_rng(aseed), WINDOW)
    row = dict(label=label, bins={}, snr=ir.snrAt(times, nb, WINDOW[0], WINDOW[1], REAL_MVT_MS / 1e3))
    for j, bw in enumerate(BIN_WIDTHS):
        c = ir.lightCurve(times, WINDOW[0], WINDOW[1], bw)
        r = h.estimateMvtDistribution(c, np.sqrt(c), h.haarPowerMod, binSizeSeconds=bw, nRealisations=N_RESAMPLE,
                                      nWorkers=args.workers, seed=list(rsSeed) + [j], resampleErrors=True)
        row["bins"][f"{bw:g}"] = dict(median=r["median"] * 1e3, p16=r["p16"] * 1e3, p84=r["p84"] * 1e3,
                                     validPct=100.0 * r["nValid"] / r["nRealisations"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(row, f, default=float)
    b = row["bins"][BK]
    print(f"  {label:28s} real-data: median {b['median']:.3g} ({b['p16']:.3g}-{b['p84']:.3g}) ms, valid {b['validPct']:.0f}%",
          flush=True)
    return row


def calibrateD(pool):
    path = os.path.join(OUT, "truth", "cal_D.json")
    if os.path.exists(path):
        return json.load(open(path))["amp"]
    seeds = np.random.SeedSequence([SEED, 970, 2]).spawn(N_CAL)
    amp, hist = 10.0, []
    for it in range(3):
        snr = float(np.mean(list(pool.map(_snrTask, [(("shape", "D", amp), s, REAL_MVT_MS / 1e3) for s in seeds], chunksize=4))))
        hist.append(dict(amp=amp, meanSnr=snr))
        print(f"  calibration D: amplitude {amp:.4g} -> SNR {snr:.1f} (target {REAL_SNR:.1f})", flush=True)
        if it < 2:
            amp *= REAL_SNR / snr
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(amp=hist[-1]["amp"], history=hist), f)
    return hist[-1]["amp"]


# ================================================================ part T and part A
def amplitudes(pool=None):
    cal2 = json.load(open(os.path.join(pc.OUT, "calibration.json")))
    ampD = calibrateD(pool) if pool is not None else json.load(open(os.path.join(OUT, "truth", "cal_D.json")))["amp"]
    return {"G": cal2["G"]["matchedAmp"], "N": cal2["N"]["matchedAmp"], "D": ampD}


def truthDBright(pool, amp):
    si = SHAPE_KEYS.index("D")
    return mainRun(pool, os.path.join(OUT, "truth", "D_bright_main.json"), ("shape", "D", amp * BRIGHT),
                   np.random.SeedSequence([SEED, 971, si]).spawn(N_OBS), "D bright (truth R)")


def truthDReal(amp):
    si = SHAPE_KEYS.index("D")
    seeds = np.random.SeedSequence([SEED, 972, si]).spawn(K)
    for o in range(K):
        realObs(os.path.join(OUT, "truth", f"D_real_obs{o:02d}.json"), ("shape", "D", amp), seeds[o],
                [SEED, 973, si, o], f"D truth obs {o}")


def truthR(key):
    if key == "D":
        return json.load(open(os.path.join(OUT, "truth", "D_bright_main.json")))["bins"][BK]["median"]
    return json.load(open(os.path.join(pc.OUT, "summary.json")))["results"][key][BK]["R"]


def truthReal(key):
    if key == "D":
        return [json.load(open(os.path.join(OUT, "truth", f"D_real_obs{o:02d}.json"))) for o in range(K)]
    return [json.load(open(os.path.join(pc.OUT, "sims", f"{key}_L1_real_obs{o:02d}.json"))) for o in range(K)]


def templateOf(key, o, amp, f):
    path = os.path.join(fdir("a", f), f"{key}_obs{o:02d}_template.json")
    if os.path.exists(path):
        return json.load(open(path))
    si = SHAPE_KEYS.index(key)
    times, nb = observe(("shape", key, amp), [SEED, 974, si, o], *DEN_REGION)
    counts = ir.lightCurve(times, DEN_REGION[0], DEN_REGION[1], DEN_BIN)
    rate, src, chi = buildTemplate(counts, f)
    out = dict(counts=counts.tolist(), rate=rate.tolist(), src=src.tolist(), chi2=chi)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def partA1(pool, amps, f):
    for key in SHAPE_KEYS:
        si = SHAPE_KEYS.index(key)
        for o in range(K):
            t = templateOf(key, o, amps[key], f)
            mainRun(pool, os.path.join(fdir("a", f), f"{key}_obs{o:02d}_bright.json"), ("tpl", t["src"], BRIGHT),
                    np.random.SeedSequence([SEED, 975, si, o]).spawn(N_BRIGHT_TPL), f"f{f:g} {key} template {o} bright")


def partA2(amps, f):
    for key in SHAPE_KEYS:
        si = SHAPE_KEYS.index(key)
        for o in range(K):
            t = templateOf(key, o, amps[key], f)
            realObs(os.path.join(fdir("a", f), f"{key}_obs{o:02d}_real.json"), ("tpl", t["src"], 1.0), [SEED, 976, si, o],
                    [SEED, 977, si, o], f"f{f:g} {key} template {o} fake")


def judgeA1(f):
    rep, ok = {}, True
    for key in SHAPE_KEYS:
        try:
            R = truthR(key)
            q = np.array([json.load(open(os.path.join(fdir("a", f), f"{key}_obs{o:02d}_bright.json")))["bins"][BK]["median"] / R
                          for o in range(K)])
            chi = [json.load(open(os.path.join(fdir("a", f), f"{key}_obs{o:02d}_template.json")))["chi2"] for o in range(K)]
        except FileNotFoundError:
            return None, False
        n = int(sum(within(x) for x in q))
        good = within(float(np.nanmedian(q))) and n >= MIN_WITHIN
        ok &= good
        rep[key] = dict(R=R, ratios=q.tolist(), medianRatio=float(np.nanmedian(q)), nWithin=n, chi2=chi, ok=bool(good))
    return rep, ok


def judgeA2(f):
    rep, ok = {}, True
    for key in SHAPE_KEYS:
        try:
            Rt = [json.load(open(os.path.join(fdir("a", f), f"{key}_obs{o:02d}_bright.json")))["bins"][BK]["median"] for o in range(K)]
            rows = [json.load(open(os.path.join(fdir("a", f), f"{key}_obs{o:02d}_real.json")))["bins"][BK] for o in range(K)]
            tr = [r["bins"][BK] for r in truthReal(key)]
        except FileNotFoundError:
            return None, False
        ratio = float(np.nanmedian([r["median"] / R for r, R in zip(rows, Rt)]))
        cover = float(np.mean([r["p16"] <= R <= r["p84"] for r, R in zip(rows, Rt)]))
        v, vt = np.array([r["validPct"] for r in rows]), np.array([r["validPct"] for r in tr])
        se = float(np.sqrt(v.var(ddof=1) / v.size + vt.var(ddof=1) / vt.size))
        diff = float(v.mean() - vt.mean())
        a, b, c = within(ratio), cover >= COVER_MIN, abs(diff) <= max(10.0, 2 * se)
        ok &= a and b and c
        rep[key] = dict(ratio=ratio, cover=cover, valid=float(v.mean()), validTruth=float(vt.mean()), diff=diff, se=se,
                        a=bool(a), b=bool(b), c=bool(c), ok=bool(a and b and c))
    return rep, ok


def reportTruthD(amps):
    try:
        R = truthR("D")
        rows = [r["bins"][BK] for r in truthReal("D")]
    except FileNotFoundError:
        return None
    ratio = float(np.nanmedian([r["median"] / R for r in rows]))
    cover = float(np.mean([r["p16"] <= R <= r["p84"] for r in rows]))
    valid = float(np.mean([r["validPct"] for r in rows]))
    return dict(amp=amps["D"], R=R, ratio=ratio, cover=cover, valid=valid)


# ================================================================ part B (real precursor)
PARAMS = None


def loadParams():
    global PARAMS
    PARAMS = json.load(open(os.path.join(pa.OUT, "params.json")))
    return PARAMS


def realTemplates(f):
    path = os.path.join(fdir("b", f), "templates.json")
    if os.path.exists(path):
        return json.load(open(path))
    print("loading the detectors for the real template ...", flush=True)
    data = {d: pa.loadDetector(d) for d in pc.OPT_DETS}
    gaps = PARAMS["gapsAll"]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 920, 0, 0]))       # exactly as run R1, seed 0
    filled = {d: pa.correctTimes(data[d], gaps[d], rng, False) for d in pc.OPT_DETS}
    raw = {d: data[d]["band"] for d in pc.OPT_DETS}
    out = dict(totals={})
    for tag, td in (("raw", raw), ("filled", filled)):
        inWin = np.sort(np.concatenate([td[d][(td[d] >= WINDOW[0]) & (td[d] < WINDOW[1])] for d in pc.OPT_DETS]))
        out["totals"][tag] = int(pa.lightCurve(inWin, 3e-6, *WINDOW).sum())
        allT = np.sort(np.concatenate(list(td.values())))
        counts = ir.lightCurve(allT, DEN_REGION[0], DEN_REGION[1], DEN_BIN)
        rate, src, chi = buildTemplate(counts, f)
        out[tag] = dict(counts=counts.tolist(), rate=rate.tolist(), src=src.tolist(), chi2=chi)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def regression(tpl):
    ok = True
    for tag, run in (("raw", "R0_opt_0.003ms.json"), ("filled", "R1_opt_seed0_0.003ms.json")):
        saved = json.load(open(os.path.join(pa.OUT, "real", run)))["totalCounts"]
        same = saved == tpl["totals"][tag]
        ok &= same
        print(f"  Z{0 if tag == 'raw' else 1} regression ({tag}): counts in the window {tpl['totals'][tag]} vs saved {saved} "
              f"({run}) -> {'identical' if same else 'DIFFERENT'}")
    return ok


VARIANTS = {"clean": None, "gap": (True, False), "gapfill": (True, True)}


def partBBright(pool, tpl, f):
    for ti, tag in enumerate(("filled", "raw")):
        mainRun(pool, os.path.join(fdir("b", f), f"{tag}_bright.json"), ("tpl", tpl[tag]["src"], BRIGHT),
                np.random.SeedSequence([SEED, 978, ti]).spawn(N_OBS), f"real template ({tag}) x{BRIGHT:g}")
    mainRun(pool, os.path.join(fdir("b", f), "filled_bright10.json"), ("tpl", tpl["filled"]["src"], BRIGHT_LOW),
            np.random.SeedSequence([SEED, 978, 2]).spawn(N_OBS), f"real template (filled) x{BRIGHT_LOW:g}")


def templateTruthDefined(f):
    Rt = json.load(open(os.path.join(fdir("b", f), "filled_bright.json")))["bins"][BK]["median"]
    Rl = json.load(open(os.path.join(fdir("b", f), "filled_bright10.json")))["bins"][BK]["median"]
    return Rt, Rl, within(Rt / Rl)


def partBReal(tpl, f):
    for v, fl in VARIANTS.items():
        for o in range(K):
            artef = None if fl is None else (fl[0], fl[1], [SEED, 980, o])
            realObs(os.path.join(fdir("b", f), f"{v}_obs{o:02d}.json"), ("tpl", tpl["filled"]["src"], 1.0), [SEED, 979, o],
                    [SEED, 981, o], f"real template {v} {o}", artef)


def judgeB(f):
    try:
        Rt = json.load(open(os.path.join(fdir("b", f), "filled_bright.json")))["bins"][BK]["median"]
        RtRaw = json.load(open(os.path.join(fdir("b", f), "raw_bright.json")))["bins"][BK]["median"]
        _, RtLow, defined = templateTruthDefined(f)
        rows = {v: [json.load(open(os.path.join(fdir("b", f), f"{v}_obs{o:02d}.json"))) for o in range(K)] for v in VARIANTS}
    except FileNotFoundError:
        return None
    rep = dict(Rtpl=Rt, RtplRaw=RtRaw, RtplLow=RtLow, defined=bool(defined), variants={})
    for v, rr in rows.items():
        b = [r["bins"][BK] for r in rr]
        rep["variants"][v] = dict(ratio=float(np.nanmedian([x["median"] / Rt for x in b])),
                                  cover=float(np.mean([x["p16"] <= Rt <= x["p84"] for x in b])),
                                  valid=float(np.mean([x["validPct"] for x in b])),
                                  medianMs=float(np.nanmedian([x["median"] for x in b])),
                                  snr=float(np.mean([r["snr"] for r in rr])),
                                  ratios=[x["median"] / Rt for x in b], medians=[x["median"] for x in b],
                                  p16=[x["p16"] / Rt for x in b],
                                  p84=[x["p84"] / Rt for x in b], valids=[x["validPct"] for x in b])
    c = rep["variants"]["clean"]
    rep["B0"] = abs(c["snr"] / REAL_SNR - 1) <= 0.15
    rep["B1"] = within(c["ratio"]) if defined else None
    rep["B2"] = (c["cover"] >= COVER_MIN) if defined else None
    art = json.load(open(os.path.join(pa.OUT, "summary.json")))
    r1 = art.get("real", {}).get(f"R1_opt_{3e-6:g}")
    rep["realR1valid"] = float(np.mean(r1["valid"])) if r1 else float("nan")
    rep["realR1median"] = float(r1["medians"][0]) if r1 else float("nan")
    rep["B3"] = (abs(rep["variants"]["gap"]["valid"] - REAL_VALID) <= 15
                 and np.isfinite(rep["realR1valid"]) and abs(c["valid"] - rep["realR1valid"]) <= 15)

    def paired(v):
        dr = (np.array(rep["variants"][v]["medians"]) - np.array(c["medians"])) / REAL_MVT_MS
        dv = np.array(rep["variants"][v]["valids"]) - np.array(c["valids"])
        dr, dv = dr[np.isfinite(dr)], dv[np.isfinite(dv)]
        sr, sv = dr.std(ddof=1) / np.sqrt(dr.size), dv.std(ddof=1) / np.sqrt(dv.size)
        inside = abs(dr.mean()) <= max(0.05, 2 * sr) and abs(dv.mean()) <= max(5.0, 2 * sv)
        return dict(dRatio=float(dr.mean()), seRatio=float(sr), dValid=float(dv.mean()), seValid=float(sv), inside=bool(inside))
    rep["B4gap"], rep["B4fill"] = paired("gap"), paired("gapfill")
    matters = not rep["B4gap"]["inside"]
    undone = rep["B4fill"]["inside"]
    rep["adoption"] = "G1" if matters and undone else ("G2" if matters else "G3")
    return rep


# ================================================================ part C (fading)
def levelSpec(src, k):
    return ("tpl", src, STEP ** k)


def thresholdLevel(mains, Rt):
    ks = sorted(mains)
    ok = {k: within(mains[k]["bins"][BK]["median"] / Rt) for k in ks}
    if not ok[ks[-1]]:
        return None
    kthr = ks[-1]
    for k in reversed(ks):
        if not ok[k]:
            break
        kthr = k
    return kthr


def partCMain(pool, tpl, f):
    src = tpl["filled"]["src"]
    return {k: mainRun(pool, os.path.join(fdir("c", f), f"L{k + 100}_main.json"), levelSpec(src, k),
                       np.random.SeedSequence([SEED, 982, k + 100]).spawn(N_OBS), f"fading x{STEP ** k:.3f}", withSnr=True)
            for k in LEVEL_K}


def partCReal(tpl, mains, Rt, f):
    if not templateTruthDefined(f)[2]:
        print("  R_tpl depends on the boost (not defined): the fading levels are not analysed like the real precursor", flush=True)
        return
    src = tpl["filled"]["src"]
    kthr = thresholdLevel(mains, Rt)
    print(f"  threshold level (median rule): {'none' if kthr is None else f'x{STEP ** kthr:.3f}'}; analysing levels "
          + ", ".join(f"x{STEP ** k:.3f}" for k in realLevels(kthr)) + " like the real precursor", flush=True)
    for k in realLevels(kthr):
        seeds = np.random.SeedSequence([SEED, 982, k + 100]).spawn(N_OBS)[:K]
        for o in range(K):
            realObs(os.path.join(fdir("c", f), f"L{k + 100}_real_obs{o:02d}.json"), levelSpec(src, k), seeds[o],
                    [SEED, 983, k + 100, o], f"fading x{STEP ** k:.3f} obs {o}")


def realLevels(kthr):
    ks = [LEVEL_K[-1]] if kthr is None else [kthr, kthr - 1, LEVEL_K[-1]]
    return sorted({k for k in ks if k in LEVEL_K})


def b5Near(R):
    if not os.path.exists(B5_SUMMARY):
        return []
    d = json.load(open(B5_SUMMARY))["pulses"]
    out = []
    for key, v in d.items():
        p = v.get("precursor", {})
        if p.get("threshold") and 0.5 <= p["reference"] / R <= 2.0:
            out.append(dict(pulse=key, R=p["reference"], snr=p["threshold"]["snr"]))
    return out


def judgeC(Rt, f, defined=True):
    mains = {}
    for k in LEVEL_K:
        p = os.path.join(fdir("c", f), f"L{k + 100}_main.json")
        if not os.path.exists(p):
            return None
        mains[k] = json.load(open(p))
    kthr = thresholdLevel(mains, Rt) if defined else None
    rep = dict(defined=bool(defined), levels=[dict(k=k, scale=STEP ** k, medianMs=mains[k]["bins"][BK]["median"],
                                                   ratio=mains[k]["bins"][BK]["median"] / Rt,
                                                   valid=mains[k]["bins"][BK]["successPct"],
                            snr=mains[k]["bins"][BK].get("snr", float("nan")), p16=mains[k]["bins"][BK]["p16"] / Rt,
                            p84=mains[k]["bins"][BK]["p84"] / Rt) for k in LEVEL_K], kthr=kthr)
    rep["thrSnr"] = mains[kthr]["bins"][BK].get("snr") if kthr is not None else None
    rep["thrAtGridEdge"] = kthr == LEVEL_K[0]
    cont = [mains[k]["bins"][BK].get("snr", np.nan) for k in LEVEL_K
            if mains[k]["bins"][BK]["p16"] <= Rt <= mains[k]["bins"][BK]["p84"]]
    rep["balaRuleSnr"] = float(np.nanmin(cont)) if cont else None
    rep["real"] = {}
    for k in realLevels(kthr):
        try:
            rows = [json.load(open(os.path.join(fdir("c", f), f"L{k + 100}_real_obs{o:02d}.json")))["bins"][BK] for o in range(K)]
        except FileNotFoundError:
            continue
        rep["real"][k] = dict(ratio=float(np.nanmedian([r["median"] / Rt for r in rows])),
                              cover=float(np.mean([r["p16"] <= Rt <= r["p84"] for r in rows])),
                              valid=float(np.mean([r["validPct"] for r in rows])))
    rep["C2"] = (kthr is not None and kthr in rep["real"] and rep["real"][kthr]["cover"] >= COVER_MIN) if defined else None
    try:
        _, band, _ = ir.loadValidationCurve()
        rep["balaBand"] = list(band(Rt))
    except SystemExit:
        rep["balaBand"] = None
    rep["b5"] = b5Near(Rt)
    return rep


# ================================================================ report
def post(amps):
    rep = dict(trust="median within 1/1.2-1.2 of the truth AND truth inside 16-84% in >= 50%")
    print("\n===== PART T. KNOWN PULSES (precursor conditions) =====")
    for key in SHAPE_KEYS:
        try:
            print(f"{key}: amplitude {amps[key]:.4g}, truth R {truthR(key):.2f} ms"
                  + ("  (amplitude, R and truth fakes reused from step 2)" if key != "D" else ""))
        except FileNotFoundError:
            print(f"{key}: not run yet")
    d = reportTruthD(amps)
    if d:
        rep["truthD"] = d
        print(f"D (known double pulse) analysed like the real precursor: median/R {d['ratio']:.3f}, R in 16-84% {100 * d['cover']:.0f}%,"
              f" valid {d['valid']:.0f}% (real precursor {REAL_VALID:.0f}%; within 15 points: "
              f"{'yes' if abs(d['valid'] - REAL_VALID) <= 15 else 'no'})")

    print("\n===== PART A (7a). DOES THE TEMPLATE KEEP THE MVT? (threshold factors tried in order) =====")
    rep["A"] = {}
    chosen = None
    for f in FACTORS:
        a1, ok1 = judgeA1(f)
        if a1 is None:
            print(f"factor {f:g}: A1 not run")
            break
        rep["A"][f"{f:g}"] = dict(A1=a1, A1ok=ok1)
        for key, v in a1.items():
            q = np.array(v["ratios"])
            print(f"f{f:g} A1 {key}: R_tpl / R over {K} templates: median {v['medianRatio']:.3f} (16-84% {np.nanpercentile(q, 16):.3f}-"
                  f"{np.nanpercentile(q, 84):.3f}), {v['nWithin']} of {K} within one step; chi2 per bin {np.mean(v['chi2']):.2f} "
                  f"-> {'PASS' if v['ok'] else 'FAIL'}")
        print(f"f{f:g} A1 (all pulses) -> {'PASS' if ok1 else 'FAIL'}")
        if not ok1:
            continue
        a2, ok2 = judgeA2(f)
        if a2 is None:
            print(f"factor {f:g}: A2 not run")
            break
        rep["A"][f"{f:g}"].update(A2=a2, A2ok=ok2)
        for key, v in a2.items():
            print(f"f{f:g} A2 {key}: template fakes: median/R_tpl {v['ratio']:.3f} ({'ok' if v['a'] else 'NO'}); R_tpl in 16-84% "
                  f"{100 * v['cover']:.0f}% ({'ok' if v['b'] else 'NO'}); valid {v['valid']:.0f}% vs truth fakes {v['validTruth']:.0f}% "
                  f"(difference {v['diff']:+.1f} ± {v['se']:.1f}; {'ok' if v['c'] else 'NO'}) -> {'PASS' if v['ok'] else 'FAIL'}")
        print(f"f{f:g} A2 (all pulses) -> {'PASS' if ok2 else 'FAIL'}")
        if ok2:
            chosen = f
            break
    rep["chosen"] = chosen
    if chosen is None:
        print("STOP: no threshold factor passes 7a (or a part is not complete); Parts B and C are not run.")
        return finish(rep)
    print(f"-> 7a PASSES with threshold factor {chosen:g}: used for the real precursor")

    print("\n===== PART B (7b). THE REAL PRECURSOR'S SHAPE =====")
    b = judgeB(chosen)
    rep["B"] = b
    if b is None:
        print("B not complete")
        return finish(rep)
    tpl = json.load(open(os.path.join(fdir("b", chosen), "templates.json")))
    print(f"template chi2 per bin: filled {tpl['filled']['chi2']:.2f}, unfilled {tpl['raw']['chi2']:.2f}")
    print(f"R_tpl (truth of the real template, x{BRIGHT:g}, 0.003 ms): gap filled {b['Rtpl']:.2f} ms, unfilled {b['RtplRaw']:.2f} ms;"
          f" real precursor measured {REAL_MVT_MS:.2f} ms (R0), {b['realR1median']:.2f} ms (gap filled, R1 seed 0)")
    print(f"Bd R_tpl at x{BRIGHT_LOW:g}: {b['RtplLow']:.2f} ms vs x{BRIGHT:g}: {b['Rtpl']:.2f} ms -> "
          + ("defined (within one step)" if b["defined"] else "NOT defined: the template's MVT depends on the boost; B1, B2, C1, C2 not judged"))
    print(f"{'variant':>8s} {'median (ms)':>12s} {'median/R_tpl':>13s} {'R_tpl in 16-84%':>16s} {'valid %':>8s} {'SNR':>6s}")
    for v, x in b["variants"].items():
        print(f"{v:>8s} {x['medianMs']:>12.2f} {x['ratio']:>13.3f} {100 * x['cover']:>15.0f}% {x['valid']:>8.0f} {x['snr']:>6.1f}")
    c = b["variants"]["clean"]
    print(f"B0 SNR sanity: clean fakes {c['snr']:.1f} vs {REAL_SNR:.1f} -> {'ok' if b['B0'] else 'NOT within 15%'}")
    nj = "not judged (R_tpl not defined)"
    print(f"B1 accuracy (clean): {c['ratio']:.3f} -> {nj if b['B1'] is None else ('PASS' if b['B1'] else 'FAIL')}")
    print(f"B2 honest error bars (clean): {100 * c['cover']:.0f}% -> {nj if b['B2'] is None else ('PASS' if b['B2'] else 'FAIL')}")
    print(f"B3 shape explains C3: gap fakes {b['variants']['gap']['valid']:.0f}% vs real {REAL_VALID:.0f}%; clean fakes "
          f"{c['valid']:.0f}% vs real gap-filled {b['realR1valid']:.0f}% (both within 15 points) -> {'YES' if b['B3'] else 'no'}")
    for tag, label in (("B4gap", "gap - clean"), ("B4fill", "gapfill - clean")):
        x = b[tag]
        print(f"B4 {label}: change of the median / {REAL_MVT_MS:.1f} ms {x['dRatio']:+.3f} (± {x['seRatio']:.3f}), of valid {x['dValid']:+.1f} "
              f"(± {x['seValid']:.1f}) points -> {'within limits' if x['inside'] else 'BEYOND limits'}")
    msg = {"G1": "G1: the gap matters and filling undoes it -> the pipeline fills gaps; the precursor's official value becomes "
                 f"R1 (gap filled) = {b['realR1median']:.2f} ms, with the fill-seed spread as a systematic; R0 quoted as a check.",
           "G2": "G2: the gap matters but filling does not undo it -> windows with gaps are flagged and the gap shift quoted as a "
                 "systematic; the official value stays R0.",
           "G3": "G3: the gap does not matter -> gaps are flagged only; the official value stays R0."}[b["adoption"]]
    print("ADOPTION " + msg)

    print("\n===== PART C (7c). FADING THE REAL SHAPE =====")
    cr = judgeC(b["Rtpl"], chosen, b["defined"])
    rep["C"] = cr
    if cr is None:
        print("C not complete")
        return finish(rep)
    print(f"{'scale':>7s} {'SNR':>7s} {'median (ms)':>12s} {'median/R_tpl':>13s} {'16-84%/R_tpl':>14s} {'valid %':>8s}")
    for L in cr["levels"]:
        mark = "  <- threshold" if L["k"] == cr["kthr"] else ("  <- real precursor" if L["k"] == 0 else "")
        print(f"{L['scale']:>7.3f} {L['snr']:>7.1f} {L['medianMs']:>12.2f} {L['ratio']:>13.3f} {L['p16']:>6.2f}-{L['p84']:<7.2f} "
              f"{L['valid']:>8.0f}{mark}")
    if not cr["defined"]:
        print("C1, C2: not judged (R_tpl not defined); the table shows how the median and the valid share change with brightness")
    elif cr["kthr"] is None:
        print("C1: even the brightest level is not within one step -> no threshold")
    else:
        print(f"C1 threshold for the real shape: SNR_MVT {cr['thrSnr']:.1f} (scale {STEP ** cr['kthr']:.3f})"
              + ("  [at the faintest level of the grid: the true threshold may be lower]" if cr["thrAtGridEdge"] else ""))
        print(f"   the real precursor (SNR {REAL_SNR:.0f}) is {1 / STEP ** cr['kthr']:.1f} times brighter than this threshold")
    if cr["defined"] and cr["balaRuleSnr"] is not None:
        print(f"   Bala et al.'s rule (lowest SNR whose 16-84% contains R_tpl): {cr['balaRuleSnr']:.1f}")
    if cr["defined"] and cr["balaBand"]:
        print(f"   Bala et al.'s 95% band at R_tpl {b['Rtpl']:.1f} ms: {cr['balaBand'][0]:.1f}-{cr['balaBand'][2]:.1f}")
    for x in (cr["b5"] if cr["defined"] else []):
        print(f"   B5 simple pulse {x['pulse']} (R {x['R']:.1f} ms), precursor conditions: threshold {x['snr']:.1f}")
    for k, x in cr["real"].items():
        print(f"C2 analysed like the real precursor at scale {STEP ** k:.3f}: median/R_tpl {x['ratio']:.3f}, R_tpl in 16-84% "
              f"{100 * x['cover']:.0f}%, valid {x['valid']:.0f}%")
    if cr["defined"]:
        print(f"C2 threshold confirmed under the full trust definition (coverage >= 50% there) -> {'YES' if cr['C2'] else 'NO'}")
    return finish(rep)


def finish(rep):
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rep, f, indent=1, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(f"\nSaved {OUT}/summary.json")
    try:
        plot(rep)
    except Exception as e:
        print(f"plot not made: {e}")
    return rep


def plot(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    c = centres()
    a = ax[0, 0]
    f = rep.get("chosen")
    tp = os.path.join(fdir("b", f), "templates.json") if f is not None else ""
    if tp and os.path.exists(tp):
        t = json.load(open(tp))
        a.step(c, t["filled"]["counts"], where="mid", color="0.5", lw=0.7, label="real precursor, 6 detectors (gap filled), 1 ms")
        a.plot(c, np.array(t["filled"]["rate"]) * DEN_BIN, color="C3", lw=1.3, label="denoised template")
        a.axhline(BKG * DEN_BIN, color="k", ls="--", lw=0.8, label="background")
        a.axvspan(*WINDOW, color="C0", alpha=0.08, label="precursor window")
        a.legend(fontsize=7)
    a.set_xlabel("time since trigger (s)"); a.set_ylabel("counts per 1 ms")
    a.set_title("Real precursor and its template", fontsize=10)

    a = ax[0, 1]
    tried = list(rep.get("A", {}).keys())
    for j, fk in enumerate(tried):
        A = rep["A"][fk]
        for i, key in enumerate(SHAPE_KEYS):
            q = A["A1"][key]["ratios"]
            x0 = i * (len(tried) + 1) + j
            a.plot(x0 + np.linspace(-0.3, 0.3, len(q)), q, "o", ms=3, color=f"C{j}", label=f"factor {fk}: R_tpl / R" if i == 0 else None)
            if "A2" in A:
                a.plot(x0, A["A2"][key]["ratio"], "s", ms=8, color=f"C{j}", mfc="none")
    if tried:
        a.plot([], [], "s", color="k", mfc="none", label="template fakes: median / R_tpl")
        a.set_xticks([i * (len(tried) + 1) + (len(tried) - 1) / 2 for i in range(len(SHAPE_KEYS))]); a.set_xticklabels(SHAPE_KEYS)
        a.axhspan(1 / STEP, STEP, color="0.9", zorder=0); a.axhline(1, color="k", lw=0.8)
        a.set_yscale("log"); a.legend(fontsize=7)
    a.set_ylabel("ratio")
    a.set_title("7a: do templates of known pulses keep the MVT? (band = one step)"
                + (f"; used factor {rep['chosen']:g}" if rep.get("chosen") else ""), fontsize=10)

    a = ax[1, 0]
    B = rep.get("B")
    defined = bool(B and B["defined"])
    unit = B["Rtpl"] if defined else 1.0
    if B:
        for i, v in enumerate(VARIANTS):
            x = B["variants"][v]
            o = np.arange(K) + (i - 1) * 0.25
            r = np.array(x["ratios"]) * B["Rtpl"] / unit
            lo, hi = np.array(x["p16"]) * B["Rtpl"] / unit, np.array(x["p84"]) * B["Rtpl"] / unit
            a.errorbar(o, r, yerr=[r - lo, hi - r], fmt="o", ms=3, lw=0.7, capsize=1.5, color=f"C{i}",
                       label=f"{v}: valid {x['valid']:.0f}%" + (f", R_tpl in bar {100 * x['cover']:.0f}%" if defined else ""))
        if defined:
            a.axhspan(1 / STEP, STEP, color="0.9", zorder=0); a.axhline(1, color="k", lw=0.8)
        else:
            a.axhline(REAL_MVT_MS, color="C3", lw=1, label=f"real precursor {REAL_MVT_MS:.1f} ms")
        a.set_yscale("log"); a.legend(fontsize=7)
        a.set_title(f"7b: fakes of the real shape; real precursor valid {REAL_VALID:.0f}%"
                    + (f"; R_tpl {B['Rtpl']:.1f} ms" if defined else "; R_tpl not defined"), fontsize=10)
    a.set_xlabel("fake observation"); a.set_ylabel("median / R_tpl (16-84%)" if defined else "median MVT (ms), 16-84%")

    a = ax[1, 1]
    if rep.get("C"):
        L = rep["C"]["levels"]
        s = [x["snr"] for x in L]
        g = 1.0 if defined else B["Rtpl"]
        a.plot(s, [x["ratio"] * g for x in L], "o-", color="C0", label="median / R_tpl" if defined else "median (ms)")
        a.fill_between(s, [x["p16"] * g for x in L], [x["p84"] * g for x in L], color="C0", alpha=0.15, label="16-84% of fakes")
        ax2 = a.twinx()
        ax2.plot(s, [x["valid"] for x in L], "s--", color="C1", ms=4, label="valid %")
        ax2.set_ylim(0, 100); ax2.set_ylabel("valid share (%)", color="C1")
        if defined:
            a.axhspan(1 / STEP, STEP, color="0.9", zorder=0); a.axhline(1, color="k", lw=0.8)
        else:
            a.axhline(REAL_MVT_MS, color="C3", lw=0.8, ls="--")
        if rep["C"]["thrSnr"]:
            a.axvline(rep["C"]["thrSnr"], color="C2", lw=1.2, label=f"threshold {rep['C']['thrSnr']:.0f}")
        a.axvline(REAL_SNR, color="C3", lw=1, label=f"real precursor {REAL_SNR:.0f}")
        for x in (rep["C"]["b5"] if defined else []):
            a.axvline(x["snr"], color="0.4", lw=0.8, ls=":")
        if defined and rep["C"]["balaBand"]:
            a.axvspan(rep["C"]["balaBand"][0], rep["C"]["balaBand"][2], color="C1", alpha=0.1, label="Bala et al. 95% band")
        if defined and rep["C"]["b5"]:
            a.plot([], [], ":", color="0.4", label="B5 simple-pulse thresholds")
        a.set_xscale("log"); a.set_yscale("log")
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=7)
    a.set_xlabel("SNR_MVT"); a.set_ylabel("median / R_tpl" if defined else "median MVT (ms)")
    a.set_title("7c: real shape faded (300 fakes per level, 0.003 ms)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ir.PLOT_DIR, f"P14_precursor_template.{ext}"), dpi=130)
    plt.close(fig)
    print(f"Saved plot {ir.PLOT_DIR}/P14_precursor_template.png (.pdf too)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    pc.args = pa.args = args
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    loadParams()
    if args.post:
        post(amplitudes())
        print(f"Total time {(time.time() - t0) / 60:.1f} min")
        sys.exit(0)
    commit = os.popen(f"git -C {ir.REPO} log -1 --format='%h %s'").read().strip()
    print(f"MVTTools code in use: {commit}", flush=True)
    print(f"real precursor (step 8): {REAL_MVT_MS:.2f} ms, SNR {REAL_SNR:.1f}, background {BKG:.0f} c/s, valid {REAL_VALID:.0f}%,"
          f" detectors {pc.OPT_DETS}", flush=True)
    print("\n----- Part T: known pulses -----", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        amps = amplitudes(pool)
        truthDBright(pool, amps["D"])
    print("\n----- Part T: the known double pulse D analysed like the real precursor -----", flush=True)
    truthDReal(amps["D"])
    chosen = None
    for f in FACTORS:
        print(f"\n----- Part A1, threshold factor {f:g}: templates of known pulses -----", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            partA1(pool, amps, f)
        if not judgeA1(f)[1]:
            print(f"  A1 fails with factor {f:g}", flush=True)
            continue
        print(f"\n----- Part A2, threshold factor {f:g}: fakes drawn from the templates -----", flush=True)
        partA2(amps, f)
        if judgeA2(f)[1]:
            chosen = f
            break
        print(f"  A2 fails with factor {f:g}", flush=True)
    if chosen is None:
        post(amps)
        sys.exit("STOP: no threshold factor passes 7a (see above)")
    print(f"\n----- Part B: the real precursor (threshold factor {chosen:g}) -----", flush=True)
    tpl = realTemplates(chosen)
    if not regression(tpl):
        sys.exit("STOP: the real-data counts do not reproduce runs R0/R1")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        partBBright(pool, tpl, chosen)
    partBReal(tpl, chosen)
    Rt = json.load(open(os.path.join(fdir("b", chosen), "filled_bright.json")))["bins"][BK]["median"]
    print("\n----- Part C: fading the real shape -----", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        mains = partCMain(pool, tpl, chosen)
    partCReal(tpl, mains, Rt, chosen)
    print(f"Runs finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    post(amps)
    print(f"Total time {(time.time() - t0) / 60:.1f} min")
