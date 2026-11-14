"""
Bayesian-block episodes of Fermi GBM bursts: precursor (PE), quiet gap (QE1), main emission (ME) and whole emission
(WE), with the rules of Zhu et al. (2026, ApJ 1003, 142, Sect. 2.1) written as code, and the comparison with their
Table 1. Every choice is fixed in the method plan "Bayesian-block episodes: method plan" (12 Nov 2026).

Credit: blocks_of() and the automatic background intervals start from Iñaki's sample_tools.py
(github.com/idbl2004/mphys-project, branch precursor-spectrum, commit befd65a). Block finding is GDT's binned
bayesian_blocks (gdt.misc.bayesian_blocks), as in his notebooks 12, 14 and 18.

What happens for one burst (section numbers are those of the method plan)
  1  photons: 8-1000 keV event times of the 12 NaI (GDT slice_energy); live time from the whole TTE (all channels)
  2  background: cubic per detector on automatic intervals (pre: TTE start + 1 s to min(-5, T90 start - 5) s;
     post: T90 end + 30 s to TTE end - 1 s); after each look at the blocks the pre-burst end moves to 5 s before the
     first block above 3 sigma and the post-burst start to 20 s after the last one; refit, at most 3 times
  3  detectors, pass 1: Bala et al.'s iterative SNR rule on the catalogue's T90 window
  4  Tres: our rule (smallest of 4-128 ms whose brightest T90 bin has >= 25 net counts) and Zhu's value
  5  blocks (p0 = 0.05) and Zhu's episode rules, for both Tres
  6  detectors, pass 2 (final): the same rule on the precursor that pass 1 found; blocks and rules again
  7  checks: p0 0.01 and 0.1, pass-1 detectors, longest gap, Iñaki's main-emission end, SNR with sqrt(N), Iñaki's
     first-block reference level, and (for the two MVP bursts) Iñaki's hand-picked background intervals
  8  comparison with Zhu: each duration within max(2 Tres, 10%) of theirs

Usage (from the folder that holds this script and settings/)
  python run_bayesian_blocks.py --part sample                 the sample, and which bursts have data
  python run_bayesian_blocks.py --part download               download missing TTE and catalogue files (HEASARC)
  python run_bayesian_blocks.py --part run --burst GRB230307A one burst
  python run_bayesian_blocks.py --part run                    every usable burst of the sample
  python run_bayesian_blocks.py --part summary                summary table and windows file
Results: RESULTS/bayesian_blocks/<burst>/ (info.json, episodes.json, blocks CSVs, plot), summary.csv, windows_bb.csv.
"""
import os, sys, json, glob, time, argparse, hashlib, subprocess
import numpy as np

sys.stdout.reconfigure(line_buffering=True)   # each line reaches the log at once (tail -f works)
DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
RESULTS = "/mnt/data/rgl36/FYP/results"
HERE = os.path.dirname(os.path.abspath(__file__))
ZHU_TABLE = os.path.join(HERE, "settings", "zhu2026_table1_mrt.txt")

# ---------------------------------------------------------------- fixed before running (method plan, 12 Nov 2026)
BAND = (8.0, 1000.0)                    # keV, Zhu's band; NaI only
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]
BKG_ORDER = 3
BKG_BIN_S = 1.024
PRE_MARGIN_S, POST_MARGIN_S = 5.0, 30.0       # Iñaki's background_windows()
PRE_GAP_S, POST_GAP_S = 5.0, 20.0             # background kept this far from the first / last block above 3 sigma
MAX_BKG_ITER = 10
MIN_POST_S, STOP_POST_S = 20.0, 5.0           # flag below 20 s of post-burst background; stop below 5 s
SEARCH_START_S = -30.0
P0_MAIN, P0_CHECKS = 0.05, (0.01, 0.1)
TRES_GRID = (0.004, 0.008, 0.016, 0.032, 0.064, 0.128)
K_NET = 25.0                                  # net counts in the brightest T90 bin
SIG = 3.0                                     # block above background (precursor start, whole emission)
QUIET, CONSISTENT = 3.0, 2.0                  # gap: SNR < 3; consistent with background below 2
PEAK_MARGIN = 3.0                             # sigma by which the main peak must beat the precursor's (2 <= SNR < 3)
MIN_GAP_TRES = 8                              # gap at least 8 Tres long (Zhu)
LONGER = 2.0                                  # a later gap replaces the first if more than twice as long (Iñaki)
INAKI_ME_FRACTION, INAKI_ME_MIN_S = 0.1, 0.5  # Iñaki's main_stop(), as a check
SELECT_BIN_SHORT_S, SELECT_BIN_LONG_S, SHORT_WINDOW_S = 0.016, 0.064, 1.0   # Bala et al.'s selection bins
EDGE_MOVE_TRES = 1.0                          # pass 1 -> pass 2: flag if an edge moves more than 1 Tres
TOL_TRES, TOL_FRAC = 2.0, 0.10                # pass rule against Zhu
USABLE_LEAD_S = 6.0                           # Iñaki: TTE must start >= 6 s before T90 start

# ---------------------------------------------------------------- the sample (method plan, section 7)
FIXED = [("230307A", "ours"), ("211211A", "ours"),
         ("170228A", "inaki"), ("200311A", "inaki"), ("200914A", "inaki"), ("180728A", "inaki"),
         ("190829A", "inaki"), ("190114C", "inaki"), ("090618A", "inaki"), ("211023A", "inaki"),
         ("131215B", "I-L"), ("170412B", "I-L"), ("211019A", "I-L")]
N_RANDOM = 10
SEED_SAMPLE = 2026
ORDER_IS_DOC = ["180605A", "141102A", "190905A", "090907B", "181126B", "190525A", "170802A", "150702A", "170709A",
                "111117A", "240513A", "150228A", "130310A", "150604B", "240411A", "160804B"]
ORDER_II_DOC = ["160326A", "180722C", "090101A", "100122A", "231104A", "140628C", "210927B", "110904A", "221126A",
                "190727B", "140824B", "100323A", "110926A", "130104A", "130106B", "210102C"]
# Iñaki's hand-picked intervals for the two MVP bursts (settings/GRB*.json, inakiBkgIntervals): a check only
INAKI_BKG = {"230307A": [(-89.0, -7.0), (162.0, 273.0)], "211211A": [(-130.0, -10.0), (150.0, 450.0)]}

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["sample", "download", "run", "summary"], required=True)
parser.add_argument("--burst", help="e.g. GRB230307A (default: the whole sample)")
args = parser.parse_args()
OUT = os.path.join(RESULTS, "bayesian_blocks")


def jdefault(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"cannot save {type(o)}")


def saveJson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=jdefault)
    os.replace(tmp, path)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gitCommit():
    try:
        return subprocess.run(["git", "-C", HERE, "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


# ================================================================ Zhu's table and the sample
def readZhu():
    """Zhu et al. (2026) Table 1, machine-readable file (byte columns as in its header)."""
    rows = {}
    for line in open(ZHU_TABLE):
        if len(line) < 100 or not line[:2].isdigit() or line[7] != " ":
            continue
        name = line[0:7].strip()
        num = lambda a, b: float(line[a:b]) if line[a:b].strip() else np.nan
        rows[name] = dict(fermi=line[8:17].strip(), tres=num(18, 23), pe=num(24, 31), me=num(32, 39), we=num(40, 47),
                          qe1=num(48, 55), mvt_pe=num(56, 62), mvt_me=num(63, 69), ep_me=num(70, 77), t90=num(94, 100),
                          type=line[101:104].strip())
    if len(rows) != 367:
        sys.exit(f"STOP: expected 367 bursts in {ZHU_TABLE}, read {len(rows)}")
    return rows


def randomOrders(Z):
    """numpy default_rng(2026).permutation of each type's bursts not in FIXED (and with a GBM trigger), sorted by
    name; I-S first, then Type II from the same generator. Must reproduce the orders written in the method plan."""
    taken = {b for b, _ in FIXED} | {b for b, v in Z.items() if v["type"] == "I-L"}
    rng = np.random.default_rng(SEED_SAMPLE)
    orders = {}
    for typ in ("I-S", "II"):
        pool = sorted(b for b, v in Z.items() if v["type"] == typ and b not in taken and v["fermi"])
        orders[typ] = [str(b) for b in rng.permutation(pool)]
    if orders["I-S"][:16] != ORDER_IS_DOC or orders["II"][:16] != ORDER_II_DOC:
        sys.exit("STOP: the random orders do not match the method plan")
    return orders


def folder(name):
    return os.path.join(DATA_ROOT, f"GRB{name}", "current")


def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def files(name, Z):
    trig = "bn" + Z[name]["fermi"]
    f = folder(name)
    tte = {d: latest(os.path.join(f, f"glg_tte_{d}_{trig}_v*.fit")) for d in ALL_NAI}
    return dict(trigger=trig, tte=tte, bcat=latest(os.path.join(f, f"glg_bcat_all_{trig}_v*.fit")))


def haveData(name, Z):
    F = files(name, Z)
    return F["bcat"] is not None and all(F["tte"].values())


def usable(name, Z):
    """Iñaki's test: TTE starts at least 6 s before T90 start, so there is background before the burst."""
    from astropy.io import fits
    F = files(name, Z)
    h = fits.getheader(F["bcat"], 0)
    t90start = float(h["T90START"])
    starts = [float(fits.getheader(F["tte"][d], 0)["TSTART"]) - float(fits.getheader(F["tte"][d], 0)["TRIGTIME"])
              for d in ALL_NAI]
    return t90start - max(starts) >= USABLE_LEAD_S, t90start, max(starts)


def download(name, Z):
    from gdt.missions.fermi.gbm.finders import TriggerFinder
    os.makedirs(folder(name), exist_ok=True)
    F = files(name, Z)
    missing = [d for d, p in F["tte"].items() if p is None]
    if not missing and F["bcat"] is not None:
        return
    finder = TriggerFinder(Z[name]["fermi"])
    if missing:
        finder.get_tte(folder(name), dets=missing, verbose=False)
    if F["bcat"] is None:
        finder.get_cat_files(folder(name), verbose=False)
    if not haveData(name, Z):
        print(f"   {name}: download incomplete")


def resolveSample(Z, fetch):
    """The fixed bursts plus the first 10 usable of each random order. A burst that is not usable is replaced by the
    next one in its order, and the replacement is written down."""
    orders = randomOrders(Z)
    sample, log = [], []
    for name, group in FIXED:
        if fetch:
            download(name, Z)
        sample.append(dict(burst=name, group=group))
    for typ in ("I-S", "II"):
        n = 0
        for name in orders[typ]:
            if n == N_RANDOM:
                break
            if fetch:
                download(name, Z)
            if not haveData(name, Z):
                log.append(f"{typ} {name}: no data yet")
                if fetch:
                    continue
                n += 1                          # without downloading, list it in its place
                sample.append(dict(burst=name, group=f"{typ} random"))
                continue
            ok, t90s, ts = usable(name, Z)
            if not ok:
                log.append(f"{typ} {name}: not usable (TTE starts {ts:.1f} s, T90 starts {t90s:.1f} s), replaced by the next")
                continue
            sample.append(dict(burst=name, group=f"{typ} random"))
            n += 1
    return sample, log


# ================================================================ one burst: data and background
def loadBurst(name, Z):
    from astropy.io import fits
    from gdt.missions.fermi.gbm.tte import GbmTte
    from gdt.core.binning.unbinned import bin_by_time
    F = files(name, Z)
    h = fits.getheader(F["bcat"], 0)
    B = dict(name=name, trigger=F["trigger"], t90=float(h["T90"]), t90start=float(h["T90START"]), tte={}, times={},
             phaii={}, files={d: os.path.basename(p) for d, p in F["tte"].items()}, md5={d: md5(p) for d, p in F["tte"].items()})
    t0 = time.time()
    for d in ALL_NAI:
        tte = GbmTte.open(F["tte"][d])
        B["times"][d] = np.sort(np.asarray(tte.slice_energy(BAND).data.times))
        B["phaii"][d] = tte.to_phaii(bin_by_time, BKG_BIN_S, time_ref=0.0)
        B["tte"][d] = tte
    rng = [tuple(float(x) for x in B["tte"][d].time_range) for d in ALL_NAI]
    B["tteRange"] = (max(r[0] for r in rng), min(r[1] for r in rng))
    print(f"{name} ({F['trigger']}): 12 NaI loaded ({(time.time() - t0) / 60:.1f} min); TTE {B['tteRange'][0]:.1f} to "
          f"{B['tteRange'][1]:.1f} s; T90 {B['t90']:.3f} s from {B['t90start']:.3f} s")
    return B


def fitBackground(B, intervals):
    from gdt.core.background.fitter import BackgroundFitter
    from gdt.core.background.binned import Polynomial
    fits_, chi2 = {}, {}
    for d in ALL_NAI:
        f = BackgroundFitter.from_phaii(B["phaii"][d], Polynomial, time_ranges=[tuple(x) for x in intervals])
        f.fit(order=BKG_ORDER)
        fits_[d] = f
        chi2[d] = float(np.nansum(f.statistic) / np.nansum(f.dof))
    return fits_, chi2


def bkgRate(fit, lo, hi):
    """Fitted background rate (counts/s, dead-time corrected) in BAND for bins lo-hi."""
    lo, hi = np.atleast_1d(np.asarray(lo, float)), np.atleast_1d(np.asarray(hi, float))
    if lo.size == 1:                       # GDT's integrate_energy fails on a single bin: use its two halves
        mid = 0.5 * (lo[0] + hi[0])
        return np.array([bkgRate(fit, [lo[0], mid], [mid, hi[0]]).mean()])
    m = fit.interpolate_bins(lo, hi)
    return np.asarray(m.integrate_energy(*BAND).rates).ravel()


def alignedEdges(t0, t1, bw):
    """Bin edges covering t0-t1 with one edge exactly on T0 (Iñaki's aligned_bin_edges)."""
    return np.round(np.arange(np.floor(t0 / bw), np.ceil(t1 / bw) + 1) * bw, 9)


_liveCache = {}


def liveTime(B, d, edges):
    """GDT live time per bin from the whole TTE (all channels: 2.6 us per event, 10 us in the overflow channel)."""
    from gdt.core.binning.unbinned import bin_by_edges
    key = (B["name"], d, float(edges[0]), float(edges[-1]), len(edges))
    if key not in _liveCache:
        ph = B["tte"][d].to_phaii(bin_by_edges, edges, time_range=(float(edges[0]), float(edges[-1])))
        e = np.asarray(ph.data.exposure, float)
        if e.size != edges.size - 1:
            raise RuntimeError(f"live time has {e.size} bins, expected {edges.size - 1}")
        _liveCache[key] = e
    return _liveCache[key]


def lightCurve(B, fits_, dets, edges):
    """Summed counts, mean live time and summed expected background counts (rate x live time) per bin."""
    counts = np.zeros(edges.size - 1)
    live = np.zeros(edges.size - 1)
    bkg = np.zeros(edges.size - 1)
    for d in dets:
        counts += np.histogram(B["times"][d], bins=edges)[0]
        lt = liveTime(B, d, edges)
        live += lt / len(dets)
        bkg += bkgRate(fits_[d], edges[:-1], edges[1:]) * lt
    return counts, live, bkg


# ================================================================ detectors and Tres
def selectDetectors(B, fits_, window):
    """Bala et al.'s iterative SNR maximisation (as selectDetectors() in run_window_mvt.py), cubic background."""
    binS = SELECT_BIN_SHORT_S if window[1] - window[0] < SHORT_WINDOW_S else SELECT_BIN_LONG_S
    e = np.arange(window[0], window[1] + binS, binS)
    e = e[e <= window[1] + 1e-6 * binS]
    if e.size < 2:
        e = np.array(window, float)
        binS = window[1] - window[0]
    curves = {d: np.histogram(B["times"][d], bins=e)[0] for d in ALL_NAI}
    bkg = {d: float(bkgRate(fits_[d], window[0], window[1])[0]) for d in ALL_NAI}
    single = {d: float(curves[d].max() / np.sqrt(bkg[d] * binS)) for d in ALL_NAI}
    ranked = sorted(single, key=single.get, reverse=True)
    rows, comb, bsum = [], np.zeros(e.size - 1), 0.0
    for k, d in enumerate(ranked, 1):
        comb = comb + curves[d]
        bsum += bkg[d]
        rows.append(dict(k=k, added=d, snr=float(comb.max() / np.sqrt(bsum * binS))))
    best = max(rows, key=lambda r: r["snr"])
    chosen = sorted(ranked[:best["k"]], key=ALL_NAI.index)
    return dict(window=list(window), binWidth_s=binS, single=single, cumulative=rows, detectors=chosen, snr=best["snr"])


def ourTres(B, fits_, dets):
    """Smallest of 4-128 ms for which the brightest bin of the T90 window has >= K_NET net counts."""
    w = (B["t90start"], B["t90start"] + B["t90"])
    trial = {}
    for bw in TRES_GRID:
        e = alignedEdges(w[0], w[1], bw)
        c, lt, b = lightCurve(B, fits_, dets, e)
        trial[f"{bw * 1e3:g}ms"] = float(np.max(c - b))
        if np.max(c - b) >= K_NET:
            return bw, trial, False
    return TRES_GRID[-1], trial, True


# ================================================================ blocks and episodes
def blocksOf(counts, live, bkg, edges, p0):
    """Iñaki's blocks_of(), with the SNR of the method plan: (N - B)/sqrt(B), B = cubic background counts."""
    from gdt.core.data_primitives import TimeBins
    from gdt.misc.bayesian_blocks import bayesian_blocks
    cp = bayesian_blocks(TimeBins(counts, edges[:-1], edges[1:], live), p0=p0)
    rows = []
    for a, b in zip(cp[:-1], cp[1:]):
        rows.append((edges[a], edges[b], counts[a:b].sum(), live[a:b].sum(), bkg[a:b].sum()))
    bl = {k: np.array(v, float) for k, v in zip(("start", "stop", "n", "live", "bkg"), zip(*rows))}
    return bl


def withSnr(bl, how="cubic"):
    """Net rate, its error and the SNR of every block. 'cubic': (N - B)/sqrt(B) (main). 'sqrtN': (N - B)/sqrt(N)
    (Iñaki's denominator). 'offset': background moved to the level of the first block (Iñaki's reference level)."""
    n, lt, b = bl["n"], bl["live"], bl["bkg"].copy()
    if how == "offset":
        b = b + (n[0] / lt[0] - b[0] / lt[0]) * lt
    out = dict(bl)
    out["bkg_used"] = b
    out["net_rate"] = (n - b) / lt
    out["net_rate_err"] = np.sqrt(np.maximum(n, 1.0)) / lt
    out["snr"] = (n - b) / np.sqrt(np.maximum(n if how == "sqrtN" else b, 1.0))
    return out


def episodes(bl, tres, gapChoice="first2x", meEnd="zhu"):
    """Zhu et al.'s rules (method plan, section 5). Returns the episodes, or found=False with the reason."""
    start, stop, snr, rate, err = bl["start"], bl["stop"], bl["snr"], bl["net_rate"], bl["net_rate_err"]
    dur = stop - start
    nb = start.size
    sig = snr > SIG
    res = dict(found=False, nBlocks=int(nb), flags=[])
    if not sig.any():
        res["reason"] = "no block above 3 sigma"
        return res
    first = int(np.argmax(sig))
    if first == 0:
        res["flags"].append("starts-at-search-start")
    bright = first + int(np.argmax(rate[first:]))
    gapType = lambda i: dur[i] >= MIN_GAP_TRES * tres * (1 - 1e-9) and snr[i] < QUIET
    cands, dropped = [], []
    for i in range(first + 1, bright):
        if not gapType(i):
            continue
        if snr[i] >= CONSISTENT:                       # only close to background: the main peak must be clearly higher
            p = first + int(np.argmax(rate[first:i]))
            margin = (rate[bright] - rate[p]) / np.hypot(err[bright], err[p])
            if margin <= PEAK_MARGIN:
                dropped.append(dict(start=start[i], stop=stop[i], snr=snr[i], peakMargin=margin))
                continue
        cands.append(i)
    res["gapCandidates"] = [dict(start=start[i], stop=stop[i], dur=dur[i], snr=snr[i]) for i in cands]
    res["gapsDroppedByPeakRule"] = dropped
    if not cands:
        res["reason"] = "no quiet block between the first significant block and the brightest block"
        return res
    if gapChoice == "longest":
        g = max(cands, key=lambda i: dur[i])
    else:
        g = cands[0]
        for i in cands[1:]:
            if dur[i] > LONGER * dur[g]:
                g = i
    lastSig = int(np.nonzero(sig)[0][-1])
    weEnd = stop[lastSig]
    meStop = weEnd
    if meEnd == "zhu":
        later = [i for i in range(bright + 1, nb) if gapType(i)]
        if later:
            meStop = min(start[later[0]], weEnd)
    else:                                              # Iñaki's main_stop()
        for i in range(bright + 1, nb):
            if rate[i] < INAKI_ME_FRACTION * rate[bright] and dur[i] >= INAKI_ME_MIN_S:
                meStop = start[i]
                break
        else:
            meStop = stop[-1]
    pePeak = first + int(np.argmax(rate[first:g]))
    res.update(found=True, pe=[start[first], start[g]], qe1=[start[g], stop[g]], me=[stop[g], meStop],
               we=[start[first], weEnd], gapSnr=snr[g], gapConsistent=bool(snr[g] < CONSISTENT),
               pePeakSnr=snr[pePeak], pePeakRate=rate[pePeak], mePeakRate=rate[bright], brightest=[start[bright], stop[bright]])
    res["durations"] = {k: res[k][1] - res[k][0] for k in ("pe", "qe1", "me", "we")}
    return res


def compareZhu(ep, zrow, tres):
    tol = lambda z: max(TOL_TRES * max(tres, zrow["tres"]), TOL_FRAC * z)
    out = {}
    for k, zk in (("pe", "pe"), ("qe1", "qe1"), ("me", "me"), ("we", "we")):
        z = zrow[zk]
        if not ep.get("found"):
            out[k] = dict(ours=None, zhu=z, pass_=False)
            continue
        o = ep["durations"][k]
        out[k] = dict(ours=o, zhu=z, diff=o - z, ratio=o / z if z else np.nan, tol=tol(z), pass_=bool(abs(o - z) <= tol(z)))
    return out


def saveBlocks(path, bl):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["start", "stop", "n", "live", "bkg_used", "net_rate", "net_rate_err", "snr"]
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for i in range(bl["start"].size):
            fh.write(",".join(f"{bl[c][i]:.6g}" for c in cols) + "\n")


# ================================================================ one burst, from start to end
def runBurst(name, Z):
    zrow = Z[name]
    outDir = os.path.join(OUT, f"GRB{name}")
    B = loadBurst(name, Z)
    info = dict(burst=name, trigger=B["trigger"], script_md5=md5(os.path.abspath(__file__)), commit=gitCommit(),
                run_date=time.strftime("%Y-%m-%d %H:%M"), t90=B["t90"], t90start=B["t90start"], tteRange=B["tteRange"],
                files=B["files"], md5=B["md5"], zhu=zrow, flags=[])
    t90end = B["t90start"] + B["t90"]
    pre = [B["tteRange"][0] + 1.0, min(-PRE_MARGIN_S, B["t90start"] - PRE_MARGIN_S)]
    post = [t90end + POST_MARGIN_S, B["tteRange"][1] - 1.0]
    history = []

    # ---- background intervals, moved after each look at the blocks (pass-1 detectors, our Tres)
    for it in range(MAX_BKG_ITER + 1):
        if post[1] - post[0] < STOP_POST_S or pre[1] - pre[0] < STOP_POST_S:
            info["stopped"] = f"less than {STOP_POST_S:g} s of background (pre {pre}, post {post})"
            saveJson(os.path.join(outDir, "info.json"), info)
            print("   STOP: " + info["stopped"])
            return
        fits_, chi2 = fitBackground(B, [tuple(pre), tuple(post)])
        sel1 = selectDetectors(B, fits_, (B["t90start"], t90end))
        tres, trial, tresFlag = ourTres(B, fits_, sel1["detectors"])
        s0 = max(B["tteRange"][0] + 1.0, SEARCH_START_S)
        edges = alignedEdges(s0, post[0], tres)
        edges = edges[(edges >= B["tteRange"][0] + 1.0 - 1e-9)]
        bl = withSnr(blocksOf(*lightCurve(B, fits_, sel1["detectors"], edges), edges, P0_MAIN))
        sig = np.nonzero(bl["snr"] > SIG)[0]
        history.append(dict(iteration=it, pre=list(pre), post=list(post), detectors=sel1["detectors"], tres=tres,
                            firstSig=bl["start"][sig[0]] if sig.size else None, lastSig=bl["stop"][sig[-1]] if sig.size else None))
        if not sig.size or it == MAX_BKG_ITER:
            break
        newPre = min(pre[1], max(bl["start"][sig[0]] - PRE_GAP_S, B["tteRange"][0] + 6.0))   # Iñaki: at least 5 s of pre-burst background
        newPost = max(post[0], bl["stop"][sig[-1]] + POST_GAP_S)
        if abs(newPre - pre[1]) < 1e-6 and abs(newPost - post[0]) < 1e-6:
            break
        pre[1], post[0] = newPre, newPost
    if history[-1]["iteration"] == MAX_BKG_ITER and history[-1]["lastSig"] is not None \
            and history[-1]["lastSig"] + POST_GAP_S > post[0] + 1e-6:
        info["flags"].append("bkg-not-converged")
    if post[1] - post[0] < MIN_POST_S:
        info["flags"].append("short-post-background")
    if tresFlag:
        info["flags"].append("tres-128ms-short")
    info.update(bkgIntervals=[pre, post], bkgHistory=history, redChi2=chi2, pass1=sel1, tresOurs=tres, tresTrial=trial,
                tresZhu=zrow["tres"])
    print(f"   background (cubic): pre {pre[0]:.1f} to {pre[1]:.1f} s, post {post[0]:.1f} to {post[1]:.1f} s "
          f"({len(history)} look(s)); reduced chi2 " + " ".join(f"{d} {chi2[d]:.2f}" for d in ALL_NAI))
    print(f"   pass 1 detectors (T90 window, {sel1['binWidth_s'] * 1e3:g} ms): {' '.join(sel1['detectors'])} (SNR {sel1['snr']:.1f})")
    print(f"   Tres: ours {tres * 1e3:g} ms (brightest T90 bin, net counts: "
          + ", ".join(f"{k} {v:.0f}" for k, v in trial.items()) + f"); Zhu {zrow['tres'] * 1e3:g} ms")
    if info["flags"]:
        print(f"   flags: {' '.join(info['flags'])}")

    s0 = max(B["tteRange"][0] + 1.0, SEARCH_START_S)
    results = {}
    for tag, tr in (("ours", tres), ("zhu", zrow["tres"])):
        if tag == "zhu" and abs(tr - tres) < 1e-9:
            results["zhu"] = dict(results["ours"], note="Zhu's Tres equals ours, so this is the same segmentation")
            print(f"   --- Tres zhu = {tr * 1e3:g} ms: the same as ours, nothing to rerun")
            continue
        edges = alignedEdges(s0, post[0], tr)
        edges = edges[(edges >= B["tteRange"][0] + 1.0 - 1e-9)]
        R = dict(tres=tr, variants={})
        t1 = time.time()
        raw1 = blocksOf(*lightCurve(B, fits_, sel1["detectors"], edges), edges, P0_MAIN)
        ep1 = episodes(withSnr(raw1), tr)
        R["variants"]["pass1"] = dict(detectors=sel1["detectors"], episodes=ep1)
        if ep1["found"]:
            sel2 = selectDetectors(B, fits_, tuple(ep1["pe"]))
        else:
            sel2 = dict(sel1, note="pass 1 found no precursor, so pass 2 keeps the pass-1 detectors")
        dets = sel2["detectors"]
        R["pass2"] = sel2
        raws = {P0_MAIN: blocksOf(*lightCurve(B, fits_, dets, edges), edges, P0_MAIN)}
        for p0 in P0_CHECKS:
            raws[p0] = blocksOf(*lightCurve(B, fits_, dets, edges), edges, p0)
        main = withSnr(raws[P0_MAIN])
        saveBlocks(os.path.join(outDir, f"blocks_tres-{tag}_main.csv"), main)
        R["variants"]["main"] = dict(detectors=dets, episodes=episodes(main, tr))
        for p0 in P0_CHECKS:
            R["variants"][f"p0_{p0:g}"] = dict(detectors=dets, episodes=episodes(withSnr(raws[p0]), tr))
        R["variants"]["gap_longest"] = dict(detectors=dets, episodes=episodes(main, tr, gapChoice="longest"))
        R["variants"]["me_end_inaki"] = dict(detectors=dets, episodes=episodes(main, tr, meEnd="inaki"))
        R["variants"]["snr_sqrtN"] = dict(detectors=dets, episodes=episodes(withSnr(raws[P0_MAIN], "sqrtN"), tr))
        R["variants"]["ref_offset"] = dict(detectors=dets, episodes=episodes(withSnr(raws[P0_MAIN], "offset"), tr))
        if name in INAKI_BKG:
            fitsI, _ = fitBackground(B, INAKI_BKG[name])
            c, lt, b = lightCurve(B, fitsI, dets, edges)
            rawI = dict(raws[P0_MAIN], bkg=np.array([b[np.searchsorted(edges, s0_):np.searchsorted(edges, s1_)].sum()
                                                    for s0_, s1_ in zip(raws[P0_MAIN]["start"], raws[P0_MAIN]["stop"])]))
            R["variants"]["bkg_inaki"] = dict(detectors=dets, intervals=INAKI_BKG[name], episodes=episodes(withSnr(rawI), tr))
        epm = R["variants"]["main"]["episodes"]
        if ep1["found"] and epm["found"]:
            moves = {k: max(abs(ep1[k][0] - epm[k][0]), abs(ep1[k][1] - epm[k][1])) for k in ("pe", "qe1", "me", "we")}
            R["edgeMovesPass1to2"] = moves
            R["detectorSensitive"] = bool(max(moves.values()) > EDGE_MOVE_TRES * tr + 1e-9)
        R["zhu"] = compareZhu(epm, zrow, tr)
        results[tag] = R
        printResult(tag, R, time.time() - t1)
    info["results"] = results
    saveJson(os.path.join(outDir, "episodes.json"), info)
    plotBurst(B, fits_, info, outDir)


def printResult(tag, R, secs):
    epm = R["variants"]["main"]["episodes"]
    print(f"   --- Tres {tag} = {R['tres'] * 1e3:g} ms ({secs / 60:.1f} min); pass 2 detectors: {' '.join(R['pass2']['detectors'])}")
    if not epm["found"]:
        print(f"      main: NO PRECURSOR ({epm['reason']})")
    else:
        f = lambda w: f"{w[0]:.3f} to {w[1]:.3f} s"
        print(f"      main: PE {f(epm['pe'])}, QE1 {f(epm['qe1'])} (SNR {epm['gapSnr']:.2f}), ME {f(epm['me'])}, WE {f(epm['we'])}"
              + (f"  flags {epm['flags']}" if epm["flags"] else ""))
        if "detectorSensitive" in R:
            print(f"      pass 1 -> 2 largest edge move {max(R['edgeMovesPass1to2'].values()):.3f} s"
                  + ("  -> detector-sensitive" if R["detectorSensitive"] else ""))
    for k, v in R["zhu"].items():
        if v["ours"] is None:
            print(f"      {k:>3}: ours -, Zhu {v['zhu']:.3f} s  FAIL")
        else:
            print(f"      {k:>3}: ours {v['ours']:.3f} s, Zhu {v['zhu']:.3f} s, diff {v['diff']:+.3f} s (tolerance {v['tol']:.3f})  "
                  + ("pass" if v["pass_"] else "FAIL"))
    for vname, v in R["variants"].items():
        if vname == "main":
            continue
        e = v["episodes"]
        d = e.get("durations")
        print(f"      check {vname:>12}: " + ("no precursor" if not e["found"] else
              f"PE {d['pe']:.3f}  QE1 {d['qe1']:.3f}  ME {d['me']:.3f}  WE {d['we']:.3f} s"))


def plotBurst(B, fits_, info, outDir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [t for t in info["results"] if "note" not in info["results"][t]]    # a reused segmentation is drawn once
    fig, axes = plt.subplots(len(tags), 2, figsize=(13, 3.6 * len(tags)), squeeze=False)
    s0 = max(B["tteRange"][0] + 1.0, SEARCH_START_S)
    post0 = info["bkgIntervals"][1][0]
    for row, tag in zip(axes, tags):
        R = info["results"][tag]
        tr, dets = R["tres"], R["variants"]["main"]["detectors"]
        ep = R["variants"]["main"]["episodes"]
        bl = np.genfromtxt(os.path.join(outDir, f"blocks_tres-{tag}_main.csv"), delimiter=",", names=True)
        for ax, (a, b), disp in ((row[0], (s0, post0), max(tr, 0.256)),
                                 (row[1], ((ep["pe"][0] - 1.0, ep["me"][0] + 3.0) if ep["found"] else (-2.0, 5.0)), max(tr, 0.016))):
            e = alignedEdges(a, b, disp)
            c, lt, bk = lightCurve(B, fits_, dets, e)
            ax.step(0.5 * (e[1:] + e[:-1]), c / lt / 1e3, where="mid", color="k", lw=0.5, label=f"{'+'.join(dets)}, {disp * 1e3:g} ms")
            ax.plot(0.5 * (e[1:] + e[:-1]), bk / lt / 1e3, color="tab:orange", lw=1.0, label="cubic background")
            ax.stairs(bl["n"] / bl["live"] / 1e3, np.append(bl["start"], bl["stop"][-1]), color="tab:blue", lw=1.0, label=f"blocks (Tres {tr * 1e3:g} ms)")
            if ep["found"]:
                for key, col in (("pe", "tab:red"), ("qe1", "0.6"), ("me", "tab:green")):
                    ax.axvspan(*ep[key], color=col, alpha=0.18, lw=0)
            ax.set_xlim(a, b)
            ax.set_ylabel("Rate [kcounts/s]")
            ax.set_xlabel("Time since GBM trigger [s]")
        row[0].set_title(f"GRB {info['burst']}: Tres {tag} = {tr * 1e3:g} ms (red PE, grey QE1, green ME)", fontsize=9)
        row[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, f"GRB{info['burst']}_episodes.png"), dpi=150)
    plt.close(fig)


# ================================================================ summary across the sample
def summary(Z):
    rows, wins = [], []
    for path in sorted(glob.glob(os.path.join(OUT, "GRB*", "episodes.json"))):
        info = json.load(open(path))
        for tag, R in info.get("results", {}).items():
            ep = R["variants"]["main"]["episodes"]
            r = dict(burst=info["burst"], type=info["zhu"]["type"], tres_version=tag, tres_ms=R["tres"] * 1e3,
                     tres_ours_ms=info["tresOurs"] * 1e3, tres_zhu_ms=info["tresZhu"] * 1e3, found=ep["found"],
                     detectors_pass2=" ".join(R["pass2"]["detectors"]), detector_sensitive=R.get("detectorSensitive", ""),
                     flags=" ".join(info["flags"] + ep.get("flags", [])))
            for k, v in R["zhu"].items():
                r[f"{k}_ours_s"] = "" if v["ours"] is None else f"{v['ours']:.4f}"
                r[f"{k}_zhu_s"] = f"{v['zhu']:.4f}"
                r[f"{k}_pass"] = v["pass_"]
            rows.append(r)
            if ep["found"]:
                for k in ("pe", "qe1", "me", "we"):
                    wins.append(dict(burst=f"GRB{info['burst']}", label=f"{k}_bb", t_start=ep[k][0], t_stop=ep[k][1], tres_version=tag))
    if not rows:
        sys.exit("no results yet")
    cols = list(rows[0])
    with open(os.path.join(OUT, "summary.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    with open(os.path.join(OUT, "windows_bb.csv"), "w") as fh:
        fh.write("# Episode edges from run_bayesian_blocks.py, in s from each burst's GBM trigger\nburst,label,t_start,t_stop,tres_version\n")
        for w in wins:
            fh.write(f"{w['burst']},{w['label']},{w['t_start']:.6g},{w['t_stop']:.6g},{w['tres_version']}\n")
    for tag in ("ours", "zhu"):
        sub = [r for r in rows if r["tres_version"] == tag]
        if not sub:
            continue
        print(f"Tres {tag}: {len(sub)} bursts, precursor found in {sum(r['found'] for r in sub)}")
        for k in ("pe", "qe1", "me", "we"):
            print(f"   {k:>3} pass: {sum(bool(r[f'{k}_pass']) for r in sub)}/{len(sub)}")
    same = [r for r in rows if r["tres_version"] == "ours"]
    if same:
        step = [abs(np.log2(r["tres_ours_ms"] / r["tres_zhu_ms"])) for r in same]
        print(f"our Tres = Zhu's in {sum(s < 0.01 for s in step)}/{len(step)}, within one step in {sum(s < 1.01 for s in step)}/{len(step)}")
    print(f"Saved {os.path.join(OUT, 'summary.csv')} and windows_bb.csv")


# ================================================================ main
def main():
    Z = readZhu()
    if args.part == "summary":
        summary(Z)
        return
    if args.burst:
        name = args.burst.replace("GRB", "")
        if name not in Z:
            sys.exit(f"STOP: {args.burst} is not in Zhu et al.'s table")
        names = [name]
    else:
        names = None
    if args.part == "sample":
        sample, log = resolveSample(Z, fetch=False)
        for s in sample:
            n = s["burst"]
            st = "data" if haveData(n, Z) else "no data yet"
            if st == "data":
                ok, t90s, ts = usable(n, Z)
                st += ", usable" if ok else f", NOT usable (TTE starts {ts:.1f} s, T90 starts {t90s:.1f} s)"
            print(f"GRB{n:8s} {s['group']:10s} {Z[n]['type']:4s} Tres(Zhu) {Z[n]['tres'] * 1e3:4.0f} ms  {st}")
        for l in log:
            print("   " + l)
        return
    if args.part == "download":
        if names:
            download(names[0], Z)
            print(f"GRB{names[0]}: " + ("all files present" if haveData(names[0], Z) else "files missing"))
            return
        sample, log = resolveSample(Z, fetch=True)
        saveJson(os.path.join(OUT, "sample.json"), dict(sample=sample, log=log, date=time.strftime("%Y-%m-%d %H:%M")))
        print(f"{len(sample)} bursts in the sample; saved {os.path.join(OUT, 'sample.json')}")
        for l in log:
            print("   " + l)
        return
    if args.part == "run":
        if names is None:
            p = os.path.join(OUT, "sample.json")
            if not os.path.exists(p):
                sys.exit("STOP: run --part download first (it fixes the sample)")
            names = [s["burst"] for s in json.load(open(p))["sample"]]
        t0 = time.time()
        for n in names:
            if not haveData(n, Z):
                print(f"GRB{n}: no data, skipped")
                continue
            runBurst(n, Z)
        print(f"Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
