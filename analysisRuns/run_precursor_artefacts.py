"""
Data gaps, dead time and the window edge in the GRB 230307A precursor MVT (follows run_precursor_mvt.py and
run_precursor_check.py).

Question. The made-to-measure check (run_precursor_check.py) showed that the method measures simple pulses correctly
in the precursor's conditions, but the real precursor gives more upper limits (42% valid) than the simple pulses
(61-72%, rule C3 failed). Three instrumental or analysis effects that the simple pulses did not have are tested here,
each on the real data AND on the simulated pulses, together with the corrections for the first two:
  gaps       short stretches with no recorded photons in some detectors (data gaps; Ignacio found one of about 1.4-2.1 ms
             at T0 + 3-4 ms in n6, na and b1, inside the precursor)
  dead time  after each photon a NaI detector is blind for 2.6 microseconds (10 for the overflow channel), so at high
             rates a fraction of photons is lost, more at the peaks than in between
  edge       the precursor window ends at 0.32 s while the precursor is still well above background; the simple pulses
             had faded to almost nothing at the window edges
The results also fix, before the general pipeline is written, how every precursor will be measured (adoption rules).

Part R. Real data (6-detector set from selection.json, and all 12 NaI where stated; 8-900 keV; same background,
  Monte Carlo (300 resamples, seed 2026) and SNR as run_precursor_mvt.py)
  R-gaps   gaps in the 12 NaI detectors between -1 and 2 s (all channels). Candidates: intervals between consecutive
           photons longer than 0.2 ms whose chance probability exp(-local rate x length) is below 1e-3 (local rate:
           mean of the 20 ms before and after). Overlapping candidates in different detectors form segments, each with
           the chance probability that all the detectors empty there recorded nothing. A segment is accepted as a gap
           if its probability is below a false-alarm threshold measured on the data themselves: the smallest segment
           probability in 50 copies where each detector's photons are shifted in time by its own random amount (which
           destroys simultaneous gaps but keeps each detector's own statistics). False-alarm chance about 2% per scan.
  R-dead   live fraction of each detector in 1 ms slices: 1 - (photons below the overflow channel x 2.6 us + overflow
           photons x 10 us) / 1 ms, from all channels.
  R0       baseline, exactly as step 8: must reproduce the saved step-8 values exactly (regression check).
  R1       gaps filled: in each detector's gap, photons added at random at the detector's local 8-900 keV rate (the 20 ms
           on each side), 3 fill seeds.
  R2       gaps filled and dead time corrected: in addition, in each 1 ms slice of each detector, photons added at random
           to make up the expected loss, observed x dead / live, 3 seeds.
           R1 and R2 for the 6-detector set and for all 12.
  R3       window grid: start -0.2 or -0.064 s, end 0.25, 0.32, 0.45, 0.60 or 0.70 s (the main burst starts at about
           0.70 s), 6 detectors, without (R0 handling) and with corrections (R2 handling, seed 0).
  All at bin widths 0.01 and 0.003 ms. SNR: physical as step 8 for every run; Bala-compatible (GbmTte.merge) only for
  the uncorrected runs, because the merge works on the files and cannot include the added photons.

Part F. Simulated pulses (the G and N shapes, background, window, matched brightness, number of observations and
  resamples of run_precursor_check.py; the same observation seeds, so variants on the baseline placement are paired
  with step 2's fakes)
  The measured real quantities are applied to the fakes:
    share of each of the 6 detectors in the precursor's counts; the gaps found in R-gaps inside the window (a fake
    photon inside a detector's gap is removed with probability = that detector's share); dead time (a fake photon is
    removed with probability sum_d share_d x c_d x share_d x R(t), R(t) the fake's rate in 1 ms slices and c_d the
    real ratio dead fraction / 8-900 keV rate of detector d); edge placement (the pulse shifted so that, at the window
    end, its excess over background is the same fraction of its peak excess as the real precursor's, measured in 16 ms
    bins).
  Variants:  E (edge)   Gp (gaps)   Gf (gaps, then filled)   D (dead time)   Df (dead time, then corrected)
             EGD (all three)   EGDf (all three, gaps filled and dead time corrected)
  For each: main scheme (300 observations) and real-data scheme (20 observations x 300 resamples) at the matched
  brightness (step 2's for the baseline placement; a new calibration to the precursor's SNR for the edge placement).
  Accuracy is always judged against R = the bright, uncut pulse of step 2 (the pulse's true MVT); for the edge
  placement the bright CUT pulse is also run and its ratio reported (what the method gives for the cut pulse
  without noise).

Rules (fixed before running)
  Q0 regression: R0 (6 detectors, 0.003 and 0.01 ms) reproduces the step-8 median, 16%, 84% and number of valid
     values exactly. Otherwise the script stops.
  Q1 corrections validated on fakes: for Gf and Df, both shapes, compared observation by observation with the same 20
     fakes of step 2 (paired), the mean change of median / R is within max(0.05, 2 standard errors) and the mean
     change of the valid share within max(5 points, 2 standard errors): the correction undoes the effect.
  Q2 which effect explains C3: an effect explains it if, in its variant (E, Gp, D, EGD), the fakes' real-data valid
     share comes within 15 points of the real precursor's 42% for both shapes.
  Q3 effect on the real precursor: change of the median and of the valid share from R0 to R1 and R2 (reported), and
     the spread over the 3 fill seeds (a systematic of the correction).
  Q4 window sensitivity: the 0.003 ms medians of all 10 windows (each handling) within one ladder step (x/÷1.19) of
     the baseline window -> the MVT is robust to the window.
Adoption (fixed before running; it decides how every precursor is measured from now on)
  A1 if Q1 passes: the official precursor value is R2 (gaps filled, dead time corrected; seed 0), with the spread of
     the fill seeds added as a systematic, and R0 quoted as a check. Variant EGDf is then the made-to-measure check
     of the official value (no separate rerun). If Q1 fails: the official value stays R0 and the effects are quoted
     as systematics.
  A2 the official window stays -0.064 to 0.32 s (Dichiara et al.) if Q4 passes. If Q4 fails, the official window
     becomes -0.064 to 0.70 s (all the precursor emission before the main burst), both are quoted, and Part W runs.
  Part W (only if Q4 fails): the made-to-measure check (baseline placement, adopted handling) in the -0.064 to 0.70 s
     window, matched to the real precursor's SNR, background and valid share in that window.

Output: results/precursor_mvt/GRB230307A/artefacts/ (real/, sims/, params.json, summary.json); plot
results/plots/P13_precursor_artefacts.png (.pdf too). Resumable.

Usage (from the scripts folder, next to run_injection_recovery.py, analyse_rung_positions.py, run_precursor_check.py)
  python run_precursor_artefacts.py            everything
  python run_precursor_artefacts.py --post     analysis and plot from saved results only
"""
import os, sys, json, glob, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import run_injection_recovery as ir
import run_precursor_check as pc

OUT = os.path.join(pc.PRE_DIR, "artefacts")
GRB = pc.GRB
DATA_ROOT = pc.DATA_ROOT
ENERGY_KEV = pc.ENERGY_KEV
BKG_INTERVALS = pc.BKG_INTERVALS
BKG_BIN_S = 1.024
WINDOW = pc.WINDOW
SCAN = (-1.0, 2.0)
BIN_WIDTHS = [1e-5, 3e-6]
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]
OVERFLOW_CHANNEL = 127
DEAD_S, DEAD_OVERFLOW_S = 2.6e-6, 10e-6
GAP_MIN_S, GAP_SIDE_S, FILL_SIDE_S = 0.2e-3, 0.02, 0.02
SLICE_S = 1e-3
N_FILL = 3
N_REAL_MC = 300
STARTS, ENDS = [-0.2, -0.064], [0.25, 0.32, 0.45, 0.60, 0.70]
WIDE_WINDOW = (-0.064, 0.70)
EDGE_BIN_S = 0.016
LADDER = 2 ** 0.25
SEED = 2026
VARIANTS = {  # name: (placement, gaps, dead time, correct)
    "E": ("edge", False, False, False), "Gp": ("base", True, False, False), "Gf": ("base", True, False, True),
    "D": ("base", False, True, False), "Df": ("base", False, True, True),
    "EGD": ("edge", True, True, False), "EGDf": ("edge", True, True, True)}
TAG_ID = {name: i for i, name in enumerate(list(VARIANTS) + ["brightEdge", "W", "brightWide"])}
GAP_P_CAND, GAP_SHIFTS = 1e-3, 50
hm = None


def loadHaar():
    global hm
    if hm is None:
        hm = pc.loadHaar()
    return hm


# ================================================================ real data
def tteFile(det):
    files = glob.glob(f"{DATA_ROOT}/{GRB}/current/glg_tte_{det}_bn*_v0*.fit")
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one TTE file for {det}, found {files}")
    return files[0]


def loadDetector(det):
    """All-channel times and channels in SCAN, 8-900 keV times in SCAN, background model (as run_precursor_mvt.py)."""
    from gdt.missions.fermi.gbm.tte import GbmTte
    from gdt.core.binning.unbinned import bin_by_time
    from gdt.core.background.fitter import BackgroundFitter
    from gdt.core.background.binned import Polynomial
    tte = GbmTte.open(tteFile(det))
    t, ch = np.asarray(tte.data.times), np.asarray(tte.data.channels)
    m = (t >= SCAN[0] - 0.1) & (t < SCAN[1] + 0.1)
    allT, allC = t[m], ch[m]
    del t, ch
    o = np.argsort(allT, kind="stable")
    allT, allC = allT[o], allC[o]
    b = np.asarray(tte.slice_energy(ENERGY_KEV).data.times)
    band = np.sort(b[(b >= SCAN[0] - 0.1) & (b < SCAN[1] + 0.1)])
    phaii = tte.to_phaii(bin_by_time, BKG_BIN_S, time_ref=0.0)
    fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=BKG_INTERVALS)
    fitter.fit(order=1)
    model = fitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
    tte.close()
    return dict(allT=allT, allC=allC, band=band, tstart=np.asarray(model.tstart), tstop=np.asarray(model.tstop),
                ratesBand=np.asarray(model.integrate_energy(*ENERGY_KEV).rates).ravel(),
                ratesAll=np.asarray(model.integrate_energy().rates).ravel())


def bkgRate(d, window, key="ratesBand"):
    w = np.clip(np.minimum(d["tstop"], window[1]) - np.maximum(d["tstart"], window[0]), 0.0, None)
    return float(np.sum(w * d[key]) / w.sum())


def gapCandidates(t, det, skip=None):
    """Empty intervals between consecutive photons (all channels) longer than GAP_MIN_S whose chance probability,
    exp(-local rate x length), is below GAP_P_CAND; the local rate is the mean of the rates counted in the GAP_SIDE_S
    before and after the interval (i.e. the rate interpolated to the interval). skip: a time near which no candidate
    is kept (the join of a circular shift)."""
    dt = np.diff(t)
    out = []
    for i in np.where(dt > GAP_MIN_S)[0]:
        a, b = t[i], t[i + 1]
        if a < SCAN[0] + GAP_SIDE_S or b > SCAN[1] - GAP_SIDE_S:
            continue
        if skip is not None and abs(0.5 * (a + b) - skip) < 3 * GAP_SIDE_S:
            continue
        nL = np.searchsorted(t, a, "right") - np.searchsorted(t, a - GAP_SIDE_S)
        nR = np.searchsorted(t, b + GAP_SIDE_S, "right") - np.searchsorted(t, b)
        rate = (nL + nR) / (2 * GAP_SIDE_S)
        if rate > 0 and np.exp(-rate * (b - a)) < GAP_P_CAND:
            out.append((det, float(a), float(b), float(rate)))
    return out


def scanSegments(cands):
    """Overlapping candidates (any detectors) are grouped; inside each group every stretch between consecutive end
    points is a segment, empty in the detectors whose candidates cover it. Its chance probability is
    exp(-sum over those detectors of local rate x length): the probability that all of them recorded nothing there."""
    cands = sorted(cands, key=lambda c: c[1])
    groups, cur, curEnd = [], [], -np.inf
    for c in cands:
        if cur and c[1] >= curEnd:
            groups.append(cur); cur = []
        cur.append(c); curEnd = max(curEnd if len(cur) > 1 else -np.inf, c[2])
    if cur:
        groups.append(cur)
    segs = []
    for g in groups:
        pts = sorted({x for c in g for x in (c[1], c[2])})
        for a, b in zip(pts[:-1], pts[1:]):
            act = [c for c in g if c[1] <= a and c[2] >= b]
            if not act:
                continue
            exp = sum(c[3] * (b - a) for c in act)
            segs.append(dict(start=a, stop=b, length_ms=(b - a) * 1e3, dets=sorted(c[0] for c in act),
                             expected=exp, p=float(np.exp(-exp))))
    return segs


def findGaps(data):
    """Data gaps, with a false-alarm threshold calibrated on the data themselves. The candidates of all detectors give
    segments (scanSegments). The same search is repeated GAP_SHIFTS times after shifting each detector's photons in
    time by its own random amount (circularly within the scan), which destroys simultaneous gaps but keeps each
    detector's own statistics; the threshold is the smallest segment probability found in any shifted copy, so a
    segment of the real data below it has a false-alarm chance of about 1 / (GAP_SHIFTS + 1) for the whole scan.
    Returns the accepted gaps per detector, all segments, and the threshold."""
    times = {d: data[d]["allT"][(data[d]["allT"] >= SCAN[0]) & (data[d]["allT"] < SCAN[1])] for d in data}
    segs = scanSegments([c for d in data for c in gapCandidates(times[d], d)])
    L = SCAN[1] - SCAN[0]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 930]))
    nullMin = []
    for k in range(GAP_SHIFTS):
        cands = []
        for d in data:
            o = rng.uniform(0.05, L - 0.05)
            ts = np.sort(SCAN[0] + np.mod(times[d] - SCAN[0] + o, L))
            cands += gapCandidates(ts, d, skip=SCAN[0] + o)
        ss = scanSegments(cands)
        nullMin.append(min((x["p"] for x in ss), default=1.0))
    thr = float(min(nullMin))
    acc = {d: [] for d in data}
    for sg in segs:
        if sg["p"] < thr:
            for d in sg["dets"]:
                acc[d].append(dict(start=sg["start"], stop=sg["stop"], length_ms=sg["length_ms"], dets=sg["dets"],
                                   expected=sg["expected"], p=sg["p"]))
    return acc, segs, thr, nullMin


def deadFraction(d, lo, hi):
    """Dead fraction per 1 ms slice from all channels (2.6 us per photon, 10 us per overflow photon)."""
    edges = lo + SLICE_S * np.arange(int(round((hi - lo) / SLICE_S)) + 1)
    ov = d["allC"] >= OVERFLOW_CHANNEL
    nN = np.histogram(d["allT"][~ov], bins=edges)[0]
    nO = np.histogram(d["allT"][ov], bins=edges)[0]
    return edges, np.clip((nN * DEAD_S + nO * DEAD_OVERFLOW_S) / SLICE_S, 0.0, 0.9)


def correctTimes(d, gaps, rng, dead):
    """8-900 keV times of one detector with its gaps filled and (if dead) the dead-time loss made up."""
    t = d["band"]
    add = [t]
    for g in gaps:
        a, b = g["start"], g["stop"]
        n = (np.searchsorted(t, a) - np.searchsorted(t, a - FILL_SIDE_S)) + \
            (np.searchsorted(t, b + FILL_SIDE_S) - np.searchsorted(t, b, "right"))
        k = rng.poisson(n / (2 * FILL_SIDE_S) * (b - a))
        add.append(rng.uniform(a, b, k))
    t2 = np.sort(np.concatenate(add))
    if dead:
        edges, f = deadFraction(d, SCAN[0], SCAN[1])
        nb = np.histogram(t2, bins=edges)[0]
        k = rng.poisson(nb * f / (1.0 - f))
        extra = [rng.uniform(edges[i], edges[i + 1], k[i]) for i in np.nonzero(k)[0]]
        if extra:
            t2 = np.sort(np.concatenate([t2] + extra))
    return t2


def lightCurve(times, binS, tmin, tmax, phase=0.0):
    edges = np.arange(tmin + phase * binS, tmax + binS, binS)
    return np.histogram(times, bins=edges)[0]


def mergedTimes(dets, window):
    from gdt.missions.fermi.gbm.tte import GbmTte
    ttes = []
    for det in dets:
        tte = GbmTte.open(tteFile(det))
        ttes.append(tte.slice_time(window))
        tte.close()
    t = np.array(GbmTte.merge(ttes).slice_energy(ENERGY_KEV).data.times)
    return t[(t >= window[0]) & (t < window[1])]


def realRun(tag, timesByDet, dets, window, data, bw, merged=None):
    """One real-data Monte Carlo (as run_precursor_mvt.py) for a given set of per-detector times and a window."""
    path = os.path.join(OUT, "real", f"{tag}_{bw * 1e3:g}ms.json")
    if os.path.exists(path):
        return json.load(open(path))
    h = loadHaar()
    times = np.sort(np.concatenate([timesByDet[x][(timesByDet[x] >= window[0]) & (timesByDet[x] < window[1])] for x in dets]))
    counts = lightCurve(times, bw, *window).astype(float)
    t0 = time.time()
    r = h.estimateMvtDistribution(counts, np.sqrt(counts), h.haarPowerMod, binSizeSeconds=bw, nRealisations=N_REAL_MC,
                                  nWorkers=args.workers, seed=SEED, resampleErrors=True)
    v = r["mvtValues"][r["validMask"]] * 1e3
    s = dict(tag=tag, dets=dets, window=list(window), binWidth_s=bw, totalCounts=int(counts.sum()),
             nValid=r["nValid"], nLimit=r["nLimit"], nFailed=r["nFailed"], validPct=100.0 * r["nValid"] / r["nRealisations"],
             median=r["median"] * 1e3, p16=r["p16"] * 1e3, p84=r["p84"] * 1e3, values_ms=[float(x) for x in v],
             minutes=(time.time() - t0) / 60)
    if np.isfinite(s["median"]):
        mvtS = s["median"] / 1e3
        bkg = sum(bkgRate(data[x], window) for x in dets)
        bkgAll = sum(bkgRate(data[x], window, "ratesAll") for x in dets)
        s["bkg"] = bkg
        s["snrPhysical"] = float(lightCurve(times, mvtS, *window).max() / np.sqrt(bkg * mvtS))
        if merged is not None:
            s["snrBala"] = float(lightCurve(merged, mvtS, *window).max() / np.sqrt(bkgAll * mvtS))
    with open(path, "w") as f:
        json.dump(s, f, default=float)
    print(f"  {tag:34s} {bw * 1e3:g} ms: median {s['median']:.2f} ({s['p16']:.2f}-{s['p84']:.2f}) ms, valid {s['validPct']:.0f}%"
          f", SNR {s.get('snrPhysical', float('nan')):.1f}  [{s['minutes']:.1f} min]", flush=True)
    return s


def realPart():
    os.makedirs(os.path.join(OUT, "real"), exist_ok=True)
    print("loading the 12 NaI detectors ...", flush=True)
    data = {d: loadDetector(d) for d in ALL_NAI}
    opt = pc.OPT_DETS
    gaps, segs, thr, nullMin = findGaps(data)
    print("\n===== R-GAPS: gaps in the photon stream, -1 to 2 s (all channels, 12 NaI) =====")
    print(f"  false-alarm threshold from {GAP_SHIFTS} time-shifted copies: chance probability < {thr:.1e}"
          f" (smallest in the shifted copies; median of their minima {np.median(nullMin):.1e})")
    best = sorted(segs, key=lambda x: x["p"])[:12]
    print("  most significant segments in the real data (accepted ones marked):")
    for sg in best:
        inside = sg["stop"] > WINDOW[0] and sg["start"] < WINDOW[1]
        print(f"    {sg['start']:+.6f} to {sg['stop']:+.6f} s, {sg['length_ms']:.2f} ms, empty in {', '.join(sg['dets'])}; "
              f"{sg['expected']:.1f} photons expected, chance {sg['p']:.1e}"
              f"{'   ACCEPTED' if sg['p'] < thr else ''}{'   (inside the precursor window)' if inside else ''}")
    nAcc = sum(1 for sg in segs if sg["p"] < thr)
    print(f"  accepted gaps: {nAcc}")
    print("\n===== R-DEAD: live fraction in 1 ms slices inside the precursor window =====")
    deadSummary = {}
    for d in ALL_NAI:
        e, f = deadFraction(data[d], WINDOW[0], WINDOW[1])
        deadSummary[d] = dict(minLive=float(1 - f.max()), meanLive=float(1 - f.mean()))
        print(f"  {d}: live fraction min {1 - f.max():.3f}, mean {1 - f.mean():.3f}")

    # quantities passed to the simulations
    inWin = {d: data[d]["band"][(data[d]["band"] >= WINDOW[0]) & (data[d]["band"] < WINDOW[1])] for d in opt}
    tot = sum(v.size for v in inWin.values())
    share = {d: inWin[d].size / tot for d in opt}
    cRatio = {}
    for d in opt:
        e, f = deadFraction(data[d], WINDOW[0], WINDOW[1])
        nb = np.histogram(inWin[d], bins=e)[0] / SLICE_S
        cRatio[d] = float(np.sum(f) / np.sum(nb)) if np.sum(nb) > 0 else 0.0
    t6 = np.sort(np.concatenate(list(inWin.values())))
    eb = np.arange(WINDOW[0], WINDOW[1] + 1e-9, EDGE_BIN_S)
    c = np.histogram(t6, bins=eb)[0] / EDGE_BIN_S
    bk = sum(bkgRate(data[d], WINDOW) for d in opt)
    edgeFrac = float((c[-1] - bk) / (c.max() - bk))
    winGaps = [dict(det=d, **g) for d in opt for g in gaps[d] if g["stop"] > WINDOW[0] and g["start"] < WINDOW[1]]
    params = dict(share=share, cRatio=cRatio, edgeFraction=edgeFrac, gapsInWindow=winGaps, bkg6=bk)
    print("\nquantities passed to the simulations: detector shares " + ", ".join(f"{d} {share[d]:.3f}" for d in opt))
    print(f"  dead fraction per 8-900 keV count rate (c_d): " + ", ".join(f"{d} {cRatio[d] * 1e6:.2f} us" for d in opt))
    print(f"  edge fraction (excess at the window end / peak excess, 16 ms bins): {edgeFrac:.3f}")
    print(f"  gaps inside the window (6 detectors): {len(winGaps)}")
    with open(os.path.join(OUT, "params.json"), "w") as f:
        json.dump(dict(params, gapsAll=gaps, gapThreshold=thr, gapNullMinima=nullMin, dead=deadSummary), f, indent=1,
                  default=float)

    print("\n===== R0-R3: real-data Monte Carlo runs =====")
    merged6 = mergedTimes(opt, WINDOW)
    base = {d: data[d]["band"] for d in ALL_NAI}
    res = {}
    for bw in BIN_WIDTHS:
        res[("R0", "opt", 0, bw)] = realRun("R0_opt", base, opt, WINDOW, data, bw, merged6)
    # Q0 regression against step 8
    for bw in BIN_WIDTHS:
        saved = next(r for r in pc._pre["runs"] if r["detectorSet"] == "opt" and r["binSize_s"] == bw)
        s = res[("R0", "opt", 0, bw)]
        same = (abs(s["median"] - saved["median_ms"]) < 1e-9 * saved["median_ms"] and abs(s["p16"] - saved["p16_ms"]) < 1e-6
                and abs(s["p84"] - saved["p84_ms"]) < 1e-6 and s["nValid"] == saved["nValid"])
        print(f"  Q0 regression {bw * 1e3:g} ms: R0 {s['median']:.4f} ({s['nValid']} valid) vs step 8 {saved['median_ms']:.4f} "
              f"({saved['nValid']} valid) -> {'identical' if same else 'DIFFERENT'}")
        if not same:
            sys.exit("STOP: the baseline does not reproduce step 8; nothing else is run")
    for k in range(N_FILL):
        for handling, dead in (("R1", False), ("R2", True)):
            for setName, dets in (("opt", opt), ("all", ALL_NAI)):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 920, k, int(dead)]))
                corr = {d: correctTimes(data[d], gaps[d], rng, dead) for d in dets}
                for bw in BIN_WIDTHS:
                    res[(handling, setName, k, bw)] = realRun(f"{handling}_{setName}_seed{k}", corr, dets, WINDOW, data, bw)
    for s0 in STARTS:
        for e0 in ENDS:
            win = (s0, e0)
            merged = mergedTimes(opt, win)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 920, 0, 1]))
            corr = {d: correctTimes(data[d], gaps[d], rng, True) for d in opt}
            for bw in BIN_WIDTHS:
                res[("R3raw", f"{s0:g}_{e0:g}", 0, bw)] = realRun(f"R3_raw_{s0:g}_{e0:g}", base, opt, win, data, bw, merged)
                res[("R3cor", f"{s0:g}_{e0:g}", 0, bw)] = realRun(f"R3_cor_{s0:g}_{e0:g}", corr, opt, win, data, bw)
    return params, res


def loadReal():
    params = json.load(open(os.path.join(OUT, "params.json")))
    res = {}
    for f in glob.glob(os.path.join(OUT, "real", "*.json")):
        s = json.load(open(f))
        tag, bw = s["tag"], s["binWidth_s"]
        p = tag.split("_")
        if p[0] == "R0":
            res[("R0", "opt", 0, bw)] = s
        elif p[0] in ("R1", "R2"):
            res[(p[0], p[1], int(p[2].replace("seed", "")), bw)] = s
        elif p[0] == "R3":
            res[("R3raw" if p[1] == "raw" else "R3cor", f"{p[2]}_{p[3]}", 0, bw)] = s
    return params, res


# ================================================================ simulations
def placement(key, kind, edgeFrac, window):
    """Pulse position: 'base' as step 2; 'edge' shifted so its excess at the window end is edgeFrac of its peak."""
    p = pc.SHAPES[key]
    if kind == "base":
        return pc.GAUSS_CENTRE if p["shape"] == "gaussian" else 0.0
    f = min(max(edgeFrac, 1e-3), 0.9)
    if p["shape"] == "gaussian":
        return window[1] - p["sigma"] * np.sqrt(2 * np.log(1 / f))
    grid, cum = ir.norrisCumulative("PRE_" + key)
    rate = np.gradient(cum, grid)
    ipk = int(np.argmax(rate))
    after = np.nonzero(rate[ipk:] <= f * rate[ipk])[0]
    tEdge = grid[ipk + after[0]] if after.size else grid[-1]
    return window[1] - tEdge


def simulate(key, amp, seed, shift, window):
    """As run_precursor_check.simulate, with the pulse shifted and any window."""
    p = pc.SHAPES[key]
    rng = np.random.default_rng(seed)
    nb = rng.poisson(pc.BKG * (window[1] - window[0]))
    tb = rng.uniform(window[0], window[1], nb)
    if p["shape"] == "gaussian":
        n = rng.poisson(amp * pc.BKG * p["sigma"] * np.sqrt(2 * np.pi))
        ts = rng.normal(shift, p["sigma"], n)
    else:
        grid, cum = ir.norrisCumulative("PRE_" + key)
        n = rng.poisson(amp * pc.BKG / ir.BKG_RATE * cum[-1])
        ts = np.interp(rng.uniform(0.0, cum[-1], n), cum, grid) + shift
    ts = ts[(ts >= window[0]) & (ts < window[1])]
    return np.sort(np.concatenate([tb, ts])), int(nb)


def artefacts(times, params, gaps, dead, correct, rng, window):
    """Apply the measured gaps and dead time to a fake (and, if correct, undo them as for the real data)."""
    share, cR = params["share"], params["cRatio"]
    keep = np.ones(times.size, bool)
    if gaps:
        p = np.zeros(times.size)
        for g in params["gapsInWindow"]:
            m = (times >= g["start"]) & (times < g["stop"])
            p[m] += share[g["det"]]
        keep &= rng.uniform(size=times.size) >= np.clip(p, 0, 1)
    t = times[keep]
    edges = window[0] + SLICE_S * np.arange(int(round((window[1] - window[0]) / SLICE_S)) + 1)
    if dead:
        R = np.histogram(t, bins=edges)[0] / SLICE_S
        pd = sum(share[d] * min(cR[d] * share[d], 1.0) for d in share) * R
        idx = np.clip(np.searchsorted(edges, t, "right") - 1, 0, R.size - 1)
        t = t[rng.uniform(size=t.size) >= np.clip(pd[idx], 0, 0.9)]
    if correct:
        add = [t]
        if gaps:
            for g in params["gapsInWindow"]:
                a, b = g["start"], g["stop"]
                n = (np.searchsorted(t, a) - np.searchsorted(t, a - FILL_SIDE_S)) + \
                    (np.searchsorted(t, b + FILL_SIDE_S) - np.searchsorted(t, b, "right"))
                k = rng.poisson(n / (2 * FILL_SIDE_S) * (b - a) * share[g["det"]])
                add.append(rng.uniform(a, b, k))
        t = np.sort(np.concatenate(add))
        if dead:
            R = np.histogram(t, bins=edges)[0] / SLICE_S
            f = np.clip(sum(share[d] * cR[d] * share[d] for d in share) * R, 0, 0.9)
            k = rng.poisson(R * SLICE_S * f / (1 - f))
            extra = [rng.uniform(edges[i], edges[i + 1], k[i]) for i in np.nonzero(k)[0]]
            if extra:
                t = np.sort(np.concatenate([t] + extra))
    return t


def _fakeTask(a):
    key, amp, seed, shift, window, flags, params, aseed = a
    h = loadHaar()
    times, nb = simulate(key, amp, seed, shift, window)
    if any(flags[1:3]):
        times = artefacts(times, params, flags[1], flags[2], flags[3], np.random.default_rng(aseed), window)
    out = []
    for bw in BIN_WIDTHS:
        c = ir.lightCurve(times, window[0], window[1], bw)
        try:
            r = h.haarPowerMod(c, np.sqrt(c), minimumBinSizeSeconds=bw)
            out.append((float(r[2]), float(r[3])))
        except Exception:
            out.append((float("nan"), float("nan")))
    return out


def _snrTask(a):
    key, amp, seed, shift, window, mvtS = a
    times, nb = simulate(key, amp, seed, shift, window)
    return ir.snrAt(times, nb, window[0], window[1], mvtS)


def calibrate(pool, key, si, shift, window, target, tag):
    path = os.path.join(OUT, "sims", f"cal_{key}_{tag}.json")
    if os.path.exists(path):
        return json.load(open(path))["amp"]
    seeds = np.random.SeedSequence([SEED, 899, si]).spawn(pc.N_CAL)
    amp, hist = 10.0, []
    for it in range(3):
        snr = float(np.mean(list(pool.map(_snrTask, [(key, amp, s, shift, window, target[0] / 1e3) for s in seeds], chunksize=4))))
        hist.append(dict(amp=amp, meanSnr=snr))
        print(f"  calibration {key} {tag}: amplitude {amp:.4g} -> SNR {snr:.1f} (target {target[1]:.1f})", flush=True)
        if it < 2:
            amp *= target[1] / snr
    with open(path, "w") as f:
        json.dump(dict(amp=hist[-1]["amp"], history=hist), f)
    return hist[-1]["amp"]


def mainRun(pool, key, si, amp, shift, window, flags, params, tag, seedTag):
    path = os.path.join(OUT, "sims", f"{key}_{tag}_main.json")
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    seeds = np.random.SeedSequence([SEED, 900, si, seedTag]).spawn(pc.N_OBS)
    aseeds = [[SEED, 910, si, TAG_ID[tag], i] for i in range(pc.N_OBS)]
    res = list(pool.map(_fakeTask, [(key, amp, s, shift, window, flags, params, aseeds[i]) for i, s in enumerate(seeds)],
                        chunksize=4))
    out = dict(shape=key, tag=tag, amplitude=amp, shift=shift, window=list(window), bins={})
    for j, bw in enumerate(BIN_WIDTHS):
        s = ir.summarise([r[j][0] for r in res], [r[j][1] for r in res])
        out["bins"][f"{bw:g}"] = s
        print(f"  {key} {tag:10s} main {bw * 1e3:g} ms: median {s['median']:.3g} ms, valid {s['successPct']:.0f}%"
              f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    return out


def realScheme(key, si, amp, shift, window, flags, params, tag, seedTag):
    h = loadHaar()
    seeds = np.random.SeedSequence([SEED, 900, si, seedTag]).spawn(pc.N_OBS)[:pc.K_REAL]
    rows = []
    for o, sd in enumerate(seeds):
        path = os.path.join(OUT, "sims", f"{key}_{tag}_real_obs{o:02d}.json")
        if os.path.exists(path):
            rows.append(json.load(open(path)))
            continue
        times, nb = simulate(key, amp, sd, shift, window)
        if any(flags[1:3]):
            times = artefacts(times, params, flags[1], flags[2], flags[3],
                              np.random.default_rng([SEED, 910, si, TAG_ID[tag], o]), window)
        row = dict(obs=o, bins={})
        for j, bw in enumerate(BIN_WIDTHS):
            c = ir.lightCurve(times, window[0], window[1], bw).astype(float)
            r = h.estimateMvtDistribution(c, np.sqrt(c), h.haarPowerMod, binSizeSeconds=bw, nRealisations=pc.N_RESAMPLE,
                                          nWorkers=args.workers, seed=[SEED, 911, si, TAG_ID[tag], o, j],
                                          resampleErrors=True)
            row["bins"][f"{bw:g}"] = dict(median=r["median"] * 1e3, p16=r["p16"] * 1e3, p84=r["p84"] * 1e3,
                                         validPct=100.0 * r["nValid"] / r["nRealisations"])
        with open(path, "w") as f:
            json.dump(row, f, default=float)
        b = row["bins"][f"{3e-6:g}"]
        print(f"  {key} {tag:10s} real-data obs {o:2d}: median {b['median']:.3g} ({b['p16']:.3g}-{b['p84']:.3g}) ms, "
              f"valid {b['validPct']:.0f}%", flush=True)
        rows.append(row)
    return rows


def simPart(params):
    os.makedirs(os.path.join(OUT, "sims"), exist_ok=True)
    cal2 = json.load(open(os.path.join(pc.OUT, "calibration.json")))
    sims = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for si, key in enumerate(pc.SHAPES):
            sims[key] = {}
            shifts = {"base": placement(key, "base", 0, WINDOW), "edge": placement(key, "edge", params["edgeFraction"], WINDOW)}
            amps = {"base": cal2[key]["matchedAmp"],
                    "edge": calibrate(pool, key, si, shifts["edge"], WINDOW, (pc.REAL_MVT_MS, pc.REAL_SNR), "edge")}
            print(f"{key}: base position {shifts['base']:.4f} s, edge position {shifts['edge']:.4f} s; "
                  f"amplitudes base {amps['base']:.4g}, edge {amps['edge']:.4g}", flush=True)
            sims[key]["R_edge"] = mainRun(pool, key, si, amps["edge"] * pc.BRIGHT_FACTOR, shifts["edge"], WINDOW,
                                          ("edge", False, False, False), params, "brightEdge", 9)
            for v, flags in VARIANTS.items():
                sims[key][v] = dict(main=mainRun(pool, key, si, amps[flags[0]], shifts[flags[0]], WINDOW, flags, params, v, 1))
        for si, key in enumerate(pc.SHAPES):
            shifts = {"base": placement(key, "base", 0, WINDOW), "edge": placement(key, "edge", params["edgeFraction"], WINDOW)}
            amps = {"base": cal2[key]["matchedAmp"], "edge": json.load(open(os.path.join(OUT, "sims", f"cal_{key}_edge.json")))["amp"]}
            for v, flags in VARIANTS.items():
                sims[key][v]["real"] = realScheme(key, si, amps[flags[0]], shifts[flags[0]], WINDOW, flags, params, v, 1)
    return sims


def loadSims():
    sims = {}
    for key in pc.SHAPES:
        sims[key] = dict(R_edge=json.load(open(os.path.join(OUT, "sims", f"{key}_brightEdge_main.json"))))
        for v in VARIANTS:
            sims[key][v] = dict(main=json.load(open(os.path.join(OUT, "sims", f"{key}_{v}_main.json"))),
                                real=[json.load(open(os.path.join(OUT, "sims", f"{key}_{v}_real_obs{o:02d}.json")))
                                      for o in range(pc.K_REAL)])
    return sims


# ================================================================ analysis
def post(params, res, sims):
    rep = dict(params={k: v for k, v in params.items() if k != "gapsAll"})
    step2 = json.load(open(os.path.join(pc.OUT, "summary.json")))
    bk = f"{3e-6:g}"
    print("\n===== PART R. REAL PRECURSOR =====")
    r0 = res[("R0", "opt", 0, 3e-6)]
    print(f"R0 baseline (6 detectors, 0.003 ms): {r0['median']:.2f} ({r0['p16']:.2f}-{r0['p84']:.2f}) ms, valid {r0['validPct']:.0f}%")
    rep["real"] = {}
    for handling in ("R1", "R2"):
        for setName in ("opt", "all"):
            for bw in BIN_WIDTHS:
                rows = [res[(handling, setName, k, bw)] for k in range(N_FILL) if (handling, setName, k, bw) in res]
                if not rows:
                    continue
                meds = np.array([r["median"] for r in rows])
                vals = np.array([r["validPct"] for r in rows])
                ref = res.get(("R0", "opt", 0, bw)) if setName == "opt" else None
                print(f"{handling} {setName:3s} {bw * 1e3:g} ms: medians over {len(rows)} fill seeds "
                      + ", ".join(f"{m:.2f}" for m in meds) + f" ms; valid " + ", ".join(f"{v:.0f}" for v in vals) + "%"
                      + (f"; vs R0: median {100 * (meds[0] / ref['median'] - 1):+.1f}%, valid {vals[0] - ref['validPct']:+.0f} points"
                         if ref else ""))
                rep["real"][f"{handling}_{setName}_{bw:g}"] = dict(medians=meds.tolist(), valid=vals.tolist(),
                                                                   seedSpreadPct=float(100 * (meds.max() - meds.min()) / meds.mean()))
    print("\nR3 window grid (6 detectors, 0.003 ms): median (valid %)")
    print(f"{'window':>16s} {'uncorrected':>20s} {'corrected':>20s}")
    q4 = True
    rep["windows"] = {}
    for s0 in STARTS:
        for e0 in ENDS:
            k = f"{s0:g}_{e0:g}"
            a, b = res.get(("R3raw", k, 0, 3e-6)), res.get(("R3cor", k, 0, 3e-6))
            if not a or not b:
                continue
            for x in (a, b):
                if np.isfinite(x["median"]):
                    q4 &= (1 / LADDER) <= x["median"] / r0["median"] <= LADDER
            rep["windows"][k] = dict(raw=[a["median"], a["validPct"]], cor=[b["median"], b["validPct"]])
            print(f"{f'{s0:g} to {e0:g} s':>16s} {a['median']:>12.2f} ({a['validPct']:3.0f}%) {b['median']:>12.2f} ({b['validPct']:3.0f}%)")
    print(f"Q4 window sensitivity: all within one ladder step of the baseline -> {'PASS (robust)' if q4 else 'FAIL'}")
    a = res.get(("R3raw", f"{WINDOW[0]:g}_{WINDOW[1]:g}", 0, 3e-6))
    b = res.get(("R3cor", f"{WINDOW[0]:g}_{WINDOW[1]:g}", 0, 3e-6))
    c = res.get(("R2", "opt", 0, 3e-6))
    if a and b and c:
        same = abs(a["median"] - r0["median"]) < 1e-9 and abs(b["median"] - c["median"]) < 1e-9
        print(f"consistency: grid baseline window uncorrected = R0 ({a['median']:.4f} vs {r0['median']:.4f}), corrected = R2 seed 0 "
              f"({b['median']:.4f} vs {c['median']:.4f}) -> {'identical' if same else 'DIFFERENT'}")
    rep["Q4"] = q4

    print("\n===== PART F. SIMULATED PULSES (0.003 ms, matched SNR) =====")
    print(f"real precursor: valid {pc.REAL_VALID:.0f}%")
    print("median/R uses the TRUE pulse's R (bright, uncut, baseline placement); for the edge variants the ratio to the bright"
          " CUT pulse is also given (what the method returns for the cut pulse without noise)")
    print(f"{'variant':>8s} | {'G: median/R':>12s}{'(cut R)':>9s}{'valid%':>8s}{'R in bar':>9s} | {'N: median/R':>12s}{'(cut R)':>9s}"
          f"{'valid%':>8s}{'R in bar':>9s}")
    rep["sims"] = {}
    base = {key: step2["results"][key][bk]["rows"]["1.0"] for key in pc.SHAPES}
    line = f"{'step 2':>8s} | " + " | ".join(f"{base[k]['realRatio']:>12.3f}{'':>9s}{base[k]['realValid']:>8.0f}"
                                            f"{100 * base[k]['realCover']:>8.0f}%" for k in pc.SHAPES)
    print(line)
    for v in VARIANTS:
        out = {}
        for key in pc.SHAPES:
            R = step2["results"][key][bk]["R"]
            Rcut = sims[key]["R_edge"]["bins"][bk]["median"] if VARIANTS[v][0] == "edge" else None
            meds = np.array([o["bins"][bk]["median"] for o in sims[key][v]["real"]])
            out[key] = dict(R=R, Rcut=Rcut, realRatio=float(np.nanmedian(meds / R)),
                            realRatioCut=float(np.nanmedian(meds / Rcut)) if Rcut else None,
                            realValid=float(np.mean([o["bins"][bk]["validPct"] for o in sims[key][v]["real"]])),
                            cover=float(np.mean([o["bins"][bk]["p16"] <= R <= o["bins"][bk]["p84"] for o in sims[key][v]["real"]])),
                            mainRatio=sims[key][v]["main"]["bins"][bk]["median"] / R,
                            mainValid=sims[key][v]["main"]["bins"][bk]["successPct"])
        rep["sims"][v] = out
        cut = {k: (f"{out[k]['realRatioCut']:.3f}" if out[k]["Rcut"] else "") for k in pc.SHAPES}
        print(f"{v:>8s} | " + " | ".join(f"{out[k]['realRatio']:>12.3f}{cut[k]:>9s}{out[k]['realValid']:>8.0f}"
                                         f"{100 * out[k]['cover']:>8.0f}%" for k in pc.SHAPES))
    for key in pc.SHAPES:
        print(f"{key}: bright reference R (true pulse) {step2['results'][key][bk]['R']:.2f} ms; bright pulse cut at the window "
              f"edge {sims[key]['R_edge']['bins'][bk]['median']:.2f} ms")

    print("\n===== RULES =====")
    # Q1, paired: the same 20 simulated observations with and without the effect-and-correction
    q1 = True
    rep["Q1"] = {}
    for v in ("Gf", "Df"):
        for key in pc.SHAPES:
            R = step2["results"][key][bk]["R"]
            b0 = [json.load(open(os.path.join(pc.OUT, "sims", f"{key}_L1_real_obs{o:02d}.json"))) for o in range(pc.K_REAL)]
            dr = np.array([sims[key][v]["real"][o]["bins"][bk]["median"] / R - b0[o]["bins"][bk]["median"] / R for o in range(pc.K_REAL)])
            dv = np.array([sims[key][v]["real"][o]["bins"][bk]["validPct"] - b0[o]["bins"][bk]["validPct"] for o in range(pc.K_REAL)])
            dr, dv = dr[np.isfinite(dr)], dv[np.isfinite(dv)]
            se_r, se_v = dr.std(ddof=1) / np.sqrt(dr.size), dv.std(ddof=1) / np.sqrt(dv.size)
            ok = abs(dr.mean()) <= max(0.05, 2 * se_r) and abs(dv.mean()) <= max(5.0, 2 * se_v)
            q1 &= ok
            rep["Q1"][f"{v}_{key}"] = dict(meanDiffRatio=float(dr.mean()), seRatio=float(se_r), meanDiffValid=float(dv.mean()),
                                           seValid=float(se_v), ok=bool(ok))
            print(f"Q1 {v} {key}: paired change of median/R {dr.mean():+.3f} (± {se_r:.3f}), of valid share {dv.mean():+.1f} "
                  f"(± {se_v:.1f}) points -> {'ok' if ok else 'NOT undone'}")
    print(f"Q1 corrections undo their effect on the fakes (paired change within max(0.05, 2 SE) and max(5, 2 SE) points) -> "
          f"{'PASS' if q1 else 'FAIL'}")
    q2 = {v: all(abs(rep["sims"][v][k]["realValid"] - pc.REAL_VALID) <= 15 for k in pc.SHAPES) for v in ("E", "Gp", "D", "EGD")}
    for v, ok in q2.items():
        print(f"Q2 does '{v}' explain C3 (fakes' valid share within 15 points of {pc.REAL_VALID:.0f}%, both shapes) -> {'YES' if ok else 'no'}")
    r2 = rep["real"].get(f"R2_opt_{3e-6:g}")
    if r2:
        print(f"Q3 real precursor, corrected (R2, 6 detectors, 0.003 ms): {r2['medians'][0]:.2f} ms vs {r0['median']:.2f} ms "
              f"({100 * (r2['medians'][0] / r0['median'] - 1):+.1f}%); valid {r2['valid'][0]:.0f}% vs {r0['validPct']:.0f}%; "
              f"spread over fill seeds {r2['seedSpreadPct']:.1f}%")
    print(f"Q4 window: {'PASS' if q4 else 'FAIL'}")
    print("\n----- ADOPTION -----")
    if q1 and r2:
        print(f"A1: corrections validated -> official precursor value = R2 (gaps filled, dead time corrected): "
              f"{r2['medians'][0]:.2f} ms (seed 0), fill-seed systematic {r2['seedSpreadPct']:.1f}%; R0 {r0['median']:.2f} ms quoted "
              "as a check. Variant EGDf is its made-to-measure check.")
    else:
        print(f"A1: corrections NOT validated -> official value stays R0 = {r0['median']:.2f} ms; the effects are quoted as systematics.")
    print("A2: " + ("window robust -> official window stays -0.064 to 0.32 s." if q4 else
                    "window NOT robust -> official window becomes -0.064 to 0.70 s (both quoted); Part W runs."))
    rep.update(Q1=q1, Q2=q2)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"\nSaved {OUT}/summary.json")
    plot(params, res, rep)
    return q1, q4


def plot(params, res, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    gapsAll = json.load(open(os.path.join(OUT, "params.json")))["gapsAll"]
    for i, d in enumerate(pc.OPT_DETS):
        for g in gapsAll[d]:
            if g["stop"] > -0.1 and g["start"] < 0.75:
                a.plot([g["start"], g["stop"]], [i, i], lw=6, color="C3")
    a.set_yticks(range(len(pc.OPT_DETS))); a.set_yticklabels(pc.OPT_DETS)
    a.axvspan(*WINDOW, color="C0", alpha=0.1)
    a.set_xlim(-0.1, 0.75); a.set_xlabel("time since trigger (s)")
    a.set_title(f"Data gaps (red) in the 6 detectors; shaded = precursor window", fontsize=10)

    a = ax[0, 1]
    for s0, m in zip(STARTS, ("o", "s")):
        for kind, c in (("raw", "C0"), ("cor", "C1")):
            x, y = [], []
            for e0 in ENDS:
                w = rep["windows"].get(f"{s0:g}_{e0:g}")
                if w:
                    x.append(e0); y.append(w[kind][0])
            a.plot(x, y, m + "-", color=c, mfc="none" if s0 == STARTS[0] else c,
                   label=f"start {s0:g} s, {'uncorrected' if kind == 'raw' else 'corrected'}")
    r0 = res[("R0", "opt", 0, 3e-6)]["median"]
    a.axhspan(r0 / LADDER, r0 * LADDER, color="0.9", zorder=0, label="baseline ± one ladder step")
    a.set_xlabel("window end (s)"); a.set_ylabel("median MVT (ms), 0.003 ms bins")
    a.set_title("Real precursor: MVT against the window", fontsize=10)
    a.legend(fontsize=7)

    names = ["step 2"] + list(VARIANTS)
    step2 = json.load(open(os.path.join(pc.OUT, "summary.json")))
    x = np.arange(len(names))
    a, b = ax[1, 0], ax[1, 1]
    for j, (key, c) in enumerate((("G", "C0"), ("N", "C1"))):
        base = step2["results"][key][f"{3e-6:g}"]["rows"]["1.0"]
        vals = [base["realValid"]] + [rep["sims"][v][key]["realValid"] for v in VARIANTS]
        rat = [base["realRatio"]] + [rep["sims"][v][key]["realRatio"] for v in VARIANTS]
        a.bar(x + (j - 0.5) * 0.38, vals, width=0.38, color=c, alpha=0.7, label=f"{key} pulses")
        b.plot(x + (j - 0.5) * 0.15, rat, "o", color=c, label=f"{key} pulses")
    a.axhline(pc.REAL_VALID, color="C3", ls="--", label=f"real precursor {pc.REAL_VALID:.0f}%")
    a.axhspan(pc.REAL_VALID - 15, pc.REAL_VALID + 15, color="C3", alpha=0.08)
    a.set_xticks(x); a.set_xticklabels(names); a.set_ylim(0, 100)
    a.set_ylabel("valid share, real-data scheme (%)")
    a.set_title("Simulated pulses at the precursor's SNR: valid share by variant (band = within 15 points)", fontsize=10)
    a.legend(fontsize=8)
    b.axhspan(0.8, 1.2, color="0.9", zorder=0)
    b.axhline(1, color="k", lw=0.8)
    b.set_xticks(x); b.set_xticklabels(names); b.set_yscale("log")
    b.set_ylabel("real-data median / R (true pulse)")
    b.set_title("Simulated pulses: accuracy by variant (band = one ladder step)", fontsize=10)
    b.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ir.PLOT_DIR, f"P13_precursor_artefacts.{ext}"), dpi=130)
    plt.close(fig)
    print(f"Saved plot {ir.PLOT_DIR}/P13_precursor_artefacts.png (.pdf too)")


# ================================================================ part W (only if the window is not robust)
def partW(params, res, adoptCorrected):
    tag = "R3cor" if adoptCorrected else "R3raw"
    real = res[(tag, f"{WIDE_WINDOW[0]:g}_{WIDE_WINDOW[1]:g}", 0, 3e-6)]
    print(f"\n===== PART W. MADE-TO-MEASURE CHECK IN THE {WIDE_WINDOW} s WINDOW =====")
    print(f"real precursor in this window ({'corrected' if adoptCorrected else 'uncorrected'}): {real['median']:.2f} ms, "
          f"SNR {real['snrPhysical']:.1f}, valid {real['validPct']:.0f}%, background {real['bkg']:.0f} c/s")
    gapsAll = json.load(open(os.path.join(OUT, "params.json")))["gapsAll"]
    params = dict(params, gapsInWindow=[dict(det=d, **g) for d in pc.OPT_DETS for g in gapsAll[d]
                                        if g["stop"] > WIDE_WINDOW[0] and g["start"] < WIDE_WINDOW[1]])
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for si, key in enumerate(pc.SHAPES):
            shift = placement(key, "base", 0, WIDE_WINDOW)
            amp = calibrate(pool, key, si, shift, WIDE_WINDOW, (real["median"], real["snrPhysical"]), "wide")
            bright = mainRun(pool, key, si, amp * pc.BRIGHT_FACTOR, shift, WIDE_WINDOW, ("base", False, False, False), params,
                             "brightWide", 9)
            fl = ("base", True, True, True) if adoptCorrected else ("base", False, False, False)
            mainRun(pool, key, si, amp, shift, WIDE_WINDOW, fl, params, "W", 1)
            rows[key] = (bright, amp, shift, fl)
    out = {}
    for si, key in enumerate(pc.SHAPES):
        bright, amp, shift, fl = rows[key]
        rr = realScheme(key, si, amp, shift, WIDE_WINDOW, fl, params, "W", 1)
        R = bright["bins"][f"{3e-6:g}"]["median"]
        meds = np.array([o["bins"][f"{3e-6:g}"]["median"] for o in rr])
        out[key] = dict(R=R, realRatio=float(np.nanmedian(meds / R)),
                        cover=float(np.mean([o["bins"][f"{3e-6:g}"]["p16"] <= R <= o["bins"][f"{3e-6:g}"]["p84"] for o in rr])),
                        realValid=float(np.mean([o["bins"][f"{3e-6:g}"]["validPct"] for o in rr])))
        v = out[key]
        print(f"{key}: R {R:.2f} ms; real-data median/R {v['realRatio']:.3f} (C1 {'PASS' if 0.8 <= v['realRatio'] <= 1.2 else 'FAIL'}); "
              f"R in error bar {100 * v['cover']:.0f}% (C2 {'PASS' if v['cover'] >= 0.5 else 'FAIL'}); valid {v['realValid']:.0f}% vs real "
              f"{real['validPct']:.0f}% (C3 {'PASS' if abs(v['realValid'] - real['validPct']) <= 15 else 'FAIL'})")
    with open(os.path.join(OUT, "partW.json"), "w") as f:
        json.dump(dict(real=real, sims=out), f, indent=1, default=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    pc.args = args
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if not args.post:
        commit = os.popen(f"git -C {ir.REPO} log -1 --format='%h %s'").read().strip()
        print(f"MVTTools code in use: {commit}", flush=True)
        print(f"real precursor (step 8): {pc.REAL_MVT_MS:.2f} ms, SNR {pc.REAL_SNR:.1f}, background {pc.BKG:.0f} c/s, "
              f"valid {pc.REAL_VALID:.0f}%, detectors {pc.OPT_DETS}", flush=True)
        params, res = realPart()
        sims = simPart(params)
        print(f"Runs finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        params, res = loadReal()
        sims = loadSims()
    q1, q4 = post(params, res, sims)
    if not q4:
        partW(params, res, q1)
    print(f"Total time {(time.time() - t0) / 60:.1f} min")
