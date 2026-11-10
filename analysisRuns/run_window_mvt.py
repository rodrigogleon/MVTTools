"""
MVT of any time window of a Fermi GBM burst, for the joint MVT - E_peak table (Rodrigo: MVT, Iñaki: E_peak).

One program for every window of every burst. All numbers specific to a burst are in settings/<burst>.json; the
windows are Iñaki's files (settings/windows_<burst>.csv, copied unchanged from his repo, commit befd65a). Every
measurement inside is code that was already run and tested in run_grb_mvt.py, run_precursor_mvt.py and
run_precursor_artefacts.py; the regression part (run first) proves that this program reproduces their numbers.

Steps for each window (the numbers are those of MVT_pipeline_for_Inaki_detailed.docx)
  0  file checks: md5 of the 12 NaI TTE files and TRIGTIME = T0 (stop if not)
  1  photons: 8-900 keV event times of the 12 NaI detectors (GDT slice_energy); detectors are added, never merged
  2  background: cubic (order-3) polynomial on Iñaki's / Dichiara's off-burst intervals (Dr Tsang, 6 Nov), 1.024 s
     bins, averaged over the window with each bin weighted by its overlap; the linear fit on Bala et al.'s intervals
     is kept as the alternative for the SNR check, and for the regression test, which reproduces runs made with it
  3  excluded and pile-up intervals (from the settings)
  4  detector selection (Bala et al. Section 4.1): 16 ms bins (window < 1 s) or 64 ms; then again with bins = MVT,
     up to 3 rounds; comparison sets (Iñaki's) at 0.003 ms
  5  data gaps: multi-detector search with a false-alarm level from 50 time-shifted copies; refill to measure the
     gap systematic (3 fill seeds); cross-check with Iñaki's per-detector list
  6  dead time: GDT live time (PHAII exposure) in 16 ms bins; checked against Iñaki's table where he has one
  7  MVT: 300 Monte Carlo realisations (seed 2026, resampled errors) at 0.1, 0.01 and 0.003 ms
  8  plateau: medians at 0.01 and 0.003 ms within one step (factor 1.2)
  9  SNR_MVT: physical (all photons) and Bala-compatible (GbmTte.merge, all-channel background)
  10 class from Bala et al.'s validation curve (8-900 keV only)
  11 Dichiara et al. comparison (GRB 230307A, 8-1000 keV, --band 8-1000)
  12 made-to-measure check (--part check, after the window has run): fakes with a known MVT in the window's own
     conditions (MVT, physical SNR, background, window; dead time if flagged), analysed exactly like the real window;
     rows have trusted = untested until it has run. Details and rules in madeToMeasure()
  13 systematics (detectors, bin width, gap, method, background)
  14 decision and the CSV row; the CSV writer refuses a file that breaks Iñaki's rules

Rules fixed before running: see the constants below and the decision rule in decide().

Usage (from the folder that holds this script and settings/)
  python run_window_mvt.py --burst GRB230307A --part regression            first: must PASS before any window runs
  python run_window_mvt.py --burst GRB230307A --part regression --rows pre C   only some regression items
  python run_window_mvt.py --burst GRB230307A --tier 1                     steps 0-11, 13, 14 for the Tier 1 windows
  python run_window_mvt.py --burst GRB230307A --windows pre m01            chosen labels
  python run_window_mvt.py --burst GRB230307A --band 8-1000                Dichiara et al. comparison (their 4 windows)
  python run_window_mvt.py --burst GRB230307A --tier 1 --post              report again from saved results (no Monte Carlo)
  python run_window_mvt.py --burst GRB230307A --part check --windows pre   step 12 for windows already measured
  python run_window_mvt.py --burst GRB230307A --part write                 write and check the CSV (no data needed)
Results: RESULTS/windows/<burst>/<band>/<label>/ (record.json, selection, Monte Carlo files, check.json and check/),
regression/, csv/.
"""
import os, sys, json, glob, time, argparse, subprocess, hashlib, zlib, csv, io, re, logging
import numpy as np

REPO = "/mnt/data/rgl36/FYP/MVTTools"
BALA_CODE = "/mnt/data/rgl36/FYP/external/GBM_MVT_paper/MVTfermi"
DATA_ROOT = "/mnt/data/rgl36/FYP/data/fermiGBM"
RESULTS = "/mnt/data/rgl36/FYP/results"
HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_DIR = os.path.join(HERE, "settings")

# ---------------------------------------------------------------- fixed before running
ENERGY_MAIN = (8.0, 900.0)              # Bala et al.'s validation curve band and Iñaki's NaI band
BIN_WIDTHS = [1e-4, 1e-5, 3e-6]         # s; the quoted value is the smallest
N_REALISATIONS = 300
SEED = 2026
BKG_BIN_S = 1.024
WINDOW_BKG, ALT_BKG = "inaki", "bala"   # windows: cubic on Iñaki's intervals (Dr Tsang, 6 Nov); check: linear on Bala's
BKG_NAMES = {"inaki": "cubic on Iñaki's intervals", "bala": "linear on Bala et al.'s intervals"}
N_PHASES = 10                           # bin phases for the SNR range
N_POISSON = 300                         # Poisson resamples for the mean SNR
SELECT_BIN_SHORT_S, SELECT_BIN_LONG_S, SHORT_WINDOW_S = 0.016, 0.064, 1.0
MAX_ROUNDS = 3                          # detector selection <-> MVT
ONE_STEP = 1.2                          # plateau and detector-sensitive: within x/÷ 1.2
METHOD_SYS = 0.19                       # one timescale step, 2^(1/4) - 1
LIVE_BIN_S = 0.016
LIVE_MIN = 0.95                         # deadtime flag below this
LIVE_TABLE_TOL = 1e-4                   # agreement with Iñaki's 4-decimal live fractions
OVERFLOW_CHANNEL = 127
DEAD_S, DEAD_OVERFLOW_S = 2.6e-6, 10e-6
GAP_MIN_S, GAP_SIDE_S, FILL_SIDE_S, GAP_P_CAND, GAP_SHIFTS = 0.2e-3, 0.02, 0.02, 1e-3, 50
GAP_MARGIN_S = 2 * GAP_SIDE_S           # searched region = window +- 40 ms, so a gap starting up to 20 ms before
                                        # the window (or ending up to 20 ms after it) is still found
N_FILL = 3
EDGE_TOL = 1e-6                         # a bin edge within 1e-6 bin widths of the window end counts as the end
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]
# step 12 (design doc, step 12; run_precursor_check.py generalised to any window)
MTM_MVT_PER_SIGMA = 0.72                # Gaussian pulse: MVT = 0.72 sigma (injection-recovery test)
MTM_N30_100_MS = 17.54                  # true MVT of the Norris pulse with rise 30 ms, decay 100 ms (same test)
MTM_NORRIS_START = 1 / 6                # Norris pulse starts 1/6 of the window after its start (precursor: at T0)
MTM_N_CAL, MTM_CAL_ITER, MTM_AMP0 = 100, 3, 10.0   # brightness: 3 iterations of 100 fakes from peak = 10 x background
MTM_BRIGHT, MTM_N_TRUTH = 30.0, 300     # true MVT R: median of 300 fakes 30 times brighter (one MVT each)
MTM_K = 20                              # fakes analysed exactly like the real window (300 resamples each)
MTM_COVER_MIN = 10                      # rule (ii): R inside the 16-84% range in at least 10 of the 20
MTM_REALISM_PTS = 15                    # realism flag: fakes' mean valid share > 15 points from the real window's
MTM_CAL_TOL = 0.15                      # calibration reported as off if the fakes' mean SNR is > 15% from the target
MTM_SLICE_S = 1e-3                      # dead-time fakes: removal probability set in 1 ms slices
MTM_NORRIS_GRID_S = 3e-7                # photon-time grid of the Norris pulse (as run_injection_recovery.py)

CSV_COLUMNS = ["label", "mvt_s", "mvt_err_lo_s", "mvt_err_hi_s", "band_keV", "upper_limit", "class", "snr_physical",
               "snr_bala", "snr_physical_linbkg", "detectors", "bin_width_s", "valid_pct", "trusted", "flags", "sys_note",
               "script", "mvt_commit"]
CSV_COLUMNS_DICHIARA = ["label", "mvt_s", "mvt_err_lo_s", "mvt_err_hi_s", "band_keV", "upper_limit", "snr_physical",
                        "detectors", "bin_width_s", "valid_pct", "flags", "dichiara_mvt_s", "dichiara_err_s",
                        "dichiara_upper_limit", "dichiara_diff_sigma", "dichiara_check", "script", "mvt_commit"]
# every column of Iñaki's E_peak tables (his point 21): a clash breaks his merge
FORBIDDEN_COLUMNS = {"t_start", "t_stop", "dichiara_table", "model", "Epeak_keV", "Epeak_err_lo", "Epeak_err_hi",
                     "Epeak_sys", "Epeak_NaI40_keV", "alpha", "alpha_err_lo", "alpha_err_hi", "alpha_sys", "beta",
                     "beta_err_lo", "beta_err_hi", "beta_upper95", "flux_10_1000keV_erg_cm2_s", "flux_err", "cstat",
                     "dof", "Epeak_cpl_keV", "dC_cpl_minus_band", "Epeak_rest_keV", "notes", "notebook", "commit",
                     "run_date", "err_lo", "err_hi", "sys", "Gompertz_Ep", "Gompertz_err", "diff_sigma", "Dichiara_Ep",
                     "Dichiara_err", "source"}
FLAG_ORDER = ["pileup-na", "pileup-b1", "detector-switching", "detector-sensitive", "bkg-sensitive", "gap",
              "gap-mismatch", "deadtime", "no-plateau", "no-valid", "class-LUL", "class-UL", "curve-range", "mtm-fail",
              "realism"]

parser = argparse.ArgumentParser()
parser.add_argument("--burst", required=True)
parser.add_argument("--settings", help="settings file (default settings/<burst>.json)")
parser.add_argument("--part", choices=["windows", "regression", "check", "write"], default="windows")
parser.add_argument("--tier", choices=["1", "2", "3"])
parser.add_argument("--windows", nargs="+", help="window labels")
parser.add_argument("--band", choices=["8-900", "8-1000"], default="8-900")
parser.add_argument("--rows", nargs="+", help="regression items: row names (e.g. C G) and/or pre, gaps, live")
parser.add_argument("--post", action="store_true", help="from saved results only (no new Monte Carlo)")
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

ms = lambda x: float(x) * 1e3
crc = lambda s: zlib.crc32(s.encode()) & 0xFFFFFFFF


class MissingResult(Exception):
    pass


def jdefault(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"cannot save {type(o)}")


def saveJson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=jdefault)
    os.replace(tmp, path)                      # a crash never leaves a half-written file


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bandTag(band):
    return f"{band[0]:g}-{band[1]:g}keV"


def overlaps(a0, a1, b0, b1):
    return a1 > b0 and a0 < b1


# ================================================================ settings, windows, step 0
def loadSettings():
    path = args.settings or os.path.join(SETTINGS_DIR, f"{args.burst}.json")
    S = json.load(open(path))
    if S["burst"] != args.burst:
        sys.exit(f"STOP: {path} is for {S['burst']}, not {args.burst}")
    S["_path"], S["_md5"] = path, md5(path)
    return S


def readWindows(S):
    """Iñaki's windows file: '#' lines are comments, then a header with label,t_start,t_stop (more columns allowed)."""
    path = os.path.join(SETTINGS_DIR, S["windowsFile"])
    lines = [l for l in open(path).read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    W = {}
    for r in rows:
        lab, a, b = r["label"].strip(), float(r["t_start"]), float(r["t_stop"])
        if lab in W:
            sys.exit(f"STOP: label {lab} appears twice in {path}")
        if not b > a:
            sys.exit(f"STOP: window {lab} has t_stop <= t_start")
        W[lab] = (a, b)
    return W, path, md5(path)


def tteFile(S, det):
    files = glob.glob(os.path.join(DATA_ROOT, S["dataFolder"], f"glg_tte_{det}_{S['trigger']}_{S['tteVersion']}.fit"))
    if len(files) != 1:
        sys.exit(f"STOP: expected one TTE file for {det}, found {files}")
    return files[0]


def checkFiles(S):
    """Step 0: same photons as Iñaki (md5) and the same T0 (TRIGTIME in every file header)."""
    from astropy.io import fits
    bad = []
    for d in ALL_NAI:
        f = tteFile(S, d)
        m = md5(f)
        if m != S["md5"][d]:
            bad.append(f"{os.path.basename(f)}: md5 {m}, settings {S['md5'][d]}")
        trig = fits.getheader(f, 0).get("TRIGTIME")
        if trig is None or abs(float(trig) - S["t0_met"]) > 1e-6:
            bad.append(f"{os.path.basename(f)}: TRIGTIME {trig}, settings T0 {S['t0_met']}")
    if bad:
        sys.exit("STOP (step 0): " + "; ".join(bad))
    print(f"step 0: md5 and TRIGTIME of the 12 NaI files match the settings (T0 = MET {S['t0_met']})")


# ================================================================ steps 1-2: photons and background
def loadBurst(S, bands):
    """Every NaI detector once: event times per band (GDT slice_energy), all-channel times and channels, the two
    background fits. The TTE objects stay open for GbmTte.merge (step 9) and GDT live time (step 6)."""
    from gdt.missions.fermi.gbm.tte import GbmTte
    from gdt.core.binning.unbinned import bin_by_time
    from gdt.core.background.fitter import BackgroundFitter
    from gdt.core.background.binned import Polynomial
    B = dict(tte={}, band={b: {} for b in bands}, allT={}, allC={}, bkg={"bala": {}, "inaki": {}})
    t0 = time.time()
    for d in ALL_NAI:
        tte = GbmTte.open(tteFile(S, d))
        for b in bands:
            B["band"][b][d] = np.sort(np.asarray(tte.slice_energy(b).data.times))
        t, c = np.asarray(tte.data.times), np.asarray(tte.data.channels)
        o = np.argsort(t, kind="stable")
        B["allT"][d], B["allC"][d] = t[o], c[o]
        phaii = tte.to_phaii(bin_by_time, BKG_BIN_S, time_ref=0.0)
        for key, ivs, order in (("bala", S["bkgIntervals"], S["bkgOrder"]),
                                ("inaki", S["inakiBkgIntervals"], S["inakiBkgOrder"])):
            fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=[tuple(x) for x in ivs])
            fitter.fit(order=order)
            model = fitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
            stat, dof = np.asarray(fitter.statistic), np.asarray(fitter.dof)
            B["bkg"][key][d] = dict(tstart=np.asarray(model.tstart), tstop=np.asarray(model.tstop),
                                    rates={b: np.asarray(model.integrate_energy(*b).rates).ravel() for b in bands},
                                    ratesAll=np.asarray(model.integrate_energy().rates).ravel(),
                                    redChi2=float(np.nansum(stat) / np.nansum(dof)) if np.nansum(dof) > 0 else np.nan)
        B["tte"][d] = tte
    B["coverage"] = (max(B["allT"][d][0] for d in ALL_NAI), min(B["allT"][d][-1] for d in ALL_NAI))
    print(f"steps 1-2: 12 NaI detectors loaded, backgrounds fitted ({(time.time() - t0) / 60:.1f} min); "
          f"TTE covers {B['coverage'][0]:.1f} to {B['coverage'][1]:.1f} s")
    print("   reduced chi2 (Bala intervals, order 1): " + "  ".join(f"{d} {B['bkg']['bala'][d]['redChi2']:.2f}" for d in ALL_NAI))
    print("   reduced chi2 (Iñaki intervals, order 3): " + "  ".join(f"{d} {B['bkg']['inaki'][d]['redChi2']:.2f}" for d in ALL_NAI))
    return B


def overlapMean(tstart, tstop, rates, window):
    """Mean fitted rate over the window, each 1.024 s bin weighted by its overlap with the window."""
    w = np.clip(np.minimum(tstop, window[1]) - np.maximum(tstart, window[0]), 0.0, None)
    if w.sum() <= 0:
        raise ValueError(f"no background bin overlaps {window}")
    return float(np.sum(w * rates) / w.sum())


def bkgRate(B, key, d, window, band=None):
    m = B["bkg"][key][d]
    return overlapMean(m["tstart"], m["tstop"], m["ratesAll"] if band is None else m["rates"][band], window)


def legacyMeanRate(B, d, window, band=None):
    """run_grb_mvt.py's background average: only the 1.024 s bins fully inside the window (regression only)."""
    m = B["bkg"]["bala"][d]
    inside = (m["tstart"] >= window[0]) & (m["tstop"] <= window[1])
    return float(np.mean((m["ratesAll"] if band is None else m["rates"][band])[inside]))


def inWindow(t, window):
    """Photons with window[0] <= t < window[1] (t sorted)."""
    return t[np.searchsorted(t, window[0], "left"):np.searchsorted(t, window[1], "left")]


def binEdges(tmin, tmax, bw, complete=True, phase=0.0):
    """Edges tmin, tmin + bw, ... as in all earlier scripts (np.arange). complete=True keeps only bins that end
    inside the window: np.arange can add one more edge past tmax (rounding), which would give a last bin that is
    partly or wholly outside the window and therefore emptier than the rest. complete=False is the earlier
    scripts' exact behaviour, used only by the regression test."""
    e = np.arange(tmin + phase * bw, tmax + bw, bw)
    if complete:
        e = e[e <= tmax + EDGE_TOL * bw]
    if e.size < 2:
        raise ValueError(f"window ({tmin}, {tmax}) shorter than one bin of {bw} s")
    return e


def lightCurve(times, bw, tmin, tmax, complete=True, phase=0.0):
    return np.histogram(times, bins=binEdges(tmin, tmax, bw, complete, phase))[0]


def windowTimes(B, band, dets, window, extra=None):
    """All photons of the detectors in the window, added (concatenated and sorted), never merged."""
    src = extra if extra is not None else B["band"][band]
    return np.sort(np.concatenate([inWindow(src[d], window) for d in dets]))


# ================================================================ step 4: detector selection
def selectDetectors(B, band, window, binS, complete=True, quiet=False, bkgKey=WINDOW_BKG):
    """Bala et al.'s iterative SNR maximisation (as run_detector_selection.py / run_precursor_mvt.py)."""
    curves = {d: lightCurve(inWindow(B["band"][band][d], window), binS, *window, complete) for d in ALL_NAI}
    bkg = {d: bkgRate(B, bkgKey, d, window, band) for d in ALL_NAI}
    single = {d: float(curves[d].max() / np.sqrt(bkg[d] * binS)) for d in ALL_NAI}
    ranked = sorted(single, key=single.get, reverse=True)
    rows, combined, bkgSum = [], np.zeros_like(curves[ranked[0]]), 0.0
    for k, det in enumerate(ranked, 1):
        combined = combined + curves[det]
        bkgSum += bkg[det]
        rows.append(dict(k=k, added=det, snr=float(combined.max() / np.sqrt(bkgSum * binS))))
    best = max(rows, key=lambda r: r["snr"])
    chosen = sorted(ranked[:best["k"]], key=ALL_NAI.index)
    if not quiet:
        print(f"   selection, {binS * 1e3:.4g} ms bins: " + "  ".join(f"{d} {single[d]:.1f}" for d in ranked))
        print("      cumulative: " + "  ".join(f"+{r['added']} {r['snr']:.1f}" for r in rows)
              + f"  -> {chosen} (SNR {best['snr']:.1f})")
    return dict(binWidth_s=binS, single=single, ranked=ranked, cumulative=rows, optimal=chosen, optimalSnr=best["snr"])


def combinedSnr(B, band, window, dets, binS):
    c = sum(lightCurve(inWindow(B["band"][band][d], window), binS, *window) for d in dets)
    return float(c.max() / np.sqrt(sum(bkgRate(B, WINDOW_BKG, d, window, band) for d in dets) * binS))


# ================================================================ step 7: Monte Carlo
def loadHaar():
    sys.path.insert(0, REPO)
    here = os.getcwd()
    os.chdir(REPO)
    from loggerSetup import initialiseLogging
    initialiseLogging()
    import analysisTools.haarMethods as haarMethods
    os.chdir(here)
    return haarMethods


def realisationType(mvt, unc):
    if not np.isfinite(mvt) or mvt <= 0.0:
        return "failed"
    if not unc > 0.0:
        return "upper limit"
    return "valid"


def mcCore(ctx, counts, bw):
    """Exactly the call of run_grb_mvt.py / run_precursor_mvt.py."""
    h = ctx["haar"]
    errors = np.sqrt(counts)
    t0 = time.time()
    (_, _, singleMvt, singleUnc, _, _, _) = h.haarPowerMod(counts, errors, minimumBinSizeSeconds=bw)
    r = h.estimateMvtDistribution(counts, errors, h.haarPowerMod, binSizeSeconds=bw, nRealisations=N_REALISATIONS,
                                  nWorkers=args.workers, seed=SEED, resampleErrors=True)
    n = r["nRealisations"]
    limits = r["mvtValues"][r["limitMask"]]
    s = dict(binSize_s=bw, nBins=int(counts.size), totalCounts=int(counts.sum()), meanCountsPerBin=float(counts.mean()),
             singleMvt_ms=ms(singleMvt), singleUnc_ms=ms(singleUnc), singleType=realisationType(singleMvt, singleUnc),
             nRealisations=n, nValid=r["nValid"], nLimit=r["nLimit"], nFailed=r["nFailed"],
             validPercent=100 * r["nValid"] / n, limitPercent=100 * r["nLimit"] / n, failedPercent=100 * r["nFailed"] / n,
             median_ms=ms(r["median"]), p16_ms=ms(r["p16"]), p84_ms=ms(r["p84"]),
             plus_err_ms=ms(r["p84"] - r["median"]), minus_err_ms=ms(r["median"] - r["p16"]),
             limits_median_ms=ms(np.median(limits)) if len(limits) else None,
             limits_p84_ms=ms(np.percentile(limits, 84)) if len(limits) else None,
             seed=SEED, resampleErrors=True, mvttoolsCommit=ctx["commit"], minutes=(time.time() - t0) / 60)
    return s, r


def mc(ctx, path, counts_fn, meta):
    """One Monte Carlo, saved as path (.json, plus the 300 values as .npy); reused if it exists with the same
    meta, refused (stop) if it exists with different meta. counts_fn builds the light curve only when needed."""
    if os.path.exists(path):
        s = json.load(open(path))
        if {k: s.get(k) for k in meta} != json.loads(json.dumps(meta, default=jdefault)):
            sys.exit(f"STOP: {path} exists with different settings {({k: s.get(k) for k in meta})} vs {meta}: "
                     "move it away first")
        return s
    if args.post:
        raise MissingResult(path)
    counts = counts_fn().astype(float)
    s, r = mcCore(ctx, counts, meta["binSize_s"])
    s.update(meta)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path.replace(".json", "_values_s.npy"), r["mvtValues"])
    np.save(path.replace(".json", "_uncertainties_s.npy"), r["mvtUncertainties"])
    saveJson(path, s)
    print(f"      MC {os.path.basename(path)[:-5]:40s} {s['totalCounts']} counts | median {s['median_ms']:.3f} "
          f"(+{s['plus_err_ms']:.3f}/-{s['minus_err_ms']:.3f}) ms | valid {s['validPercent']:.0f}% limit "
          f"{s['limitPercent']:.0f}% failed {s['failedPercent']:.0f}%  [{s['minutes']:.1f} min]", flush=True)
    return s


def setTag(dets):
    return "-".join(dets)


def mvtFor(ctx, B, band, window, dets, bw, folder, tag=None, extra=None):
    meta = dict(detectors=list(dets), binSize_s=bw, window_s=list(window), band_keV=list(band),
                nRealisations=N_REALISATIONS, fill=tag)
    name = f"{setTag(dets)}_{bw * 1e3:g}ms" + (f"_{tag}" if tag else "") + ".json"
    return mc(ctx, os.path.join(folder, "mc", name),
              lambda: lightCurve(windowTimes(B, band, dets, window, extra), bw, *window), meta)


# ================================================================ step 5: data gaps
def gapCandidates(t, det, scan, skip=None):
    """As run_precursor_artefacts.gapCandidates, vectorised (same result, same order): empty intervals between
    consecutive photons (all channels) longer than GAP_MIN_S with chance exp(-local rate x length) below GAP_P_CAND;
    local rate = mean of the counts in the GAP_SIDE_S before and after. skip: the join of a circular shift."""
    if t.size < 2:
        return []
    dt = np.diff(t)
    i = np.nonzero(dt > GAP_MIN_S)[0]
    a, b = t[i], t[i + 1]
    keep = (a >= scan[0] + GAP_SIDE_S) & (b <= scan[1] - GAP_SIDE_S)
    if skip is not None:
        keep &= ~(np.abs(0.5 * (a + b) - skip) < 3 * GAP_SIDE_S)
    nL = np.searchsorted(t, a, "right") - np.searchsorted(t, a - GAP_SIDE_S)
    nR = np.searchsorted(t, b + GAP_SIDE_S, "right") - np.searchsorted(t, b)
    rate = (nL + nR) / (2 * GAP_SIDE_S)
    with np.errstate(over="ignore"):
        keep &= (rate > 0) & (np.exp(-rate * (b - a)) < GAP_P_CAND)
    return [(det, float(x), float(y), float(r)) for x, y, r in zip(a[keep], b[keep], rate[keep])]


def scanSegments(cands):
    """Copied unchanged from run_precursor_artefacts.py."""
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


def findGaps(allT, scan, seedWords):
    """As run_precursor_artefacts.findGaps, for any searched region and seed."""
    times = {d: allT[d][(allT[d] >= scan[0]) & (allT[d] < scan[1])] for d in ALL_NAI}
    segs = scanSegments([c for d in ALL_NAI for c in gapCandidates(times[d], d, scan)])
    L = scan[1] - scan[0]
    rng = np.random.default_rng(np.random.SeedSequence(seedWords))
    nullMin = []
    for k in range(GAP_SHIFTS):
        cands = []
        for d in ALL_NAI:
            o = rng.uniform(0.05, L - 0.05)
            ts = np.sort(scan[0] + np.mod(times[d] - scan[0] + o, L))
            cands += gapCandidates(ts, d, scan, skip=scan[0] + o)
        nullMin.append(min((x["p"] for x in scanSegments(cands)), default=1.0))
    thr = float(min(nullMin))
    acc = {d: [] for d in ALL_NAI}
    for sg in segs:
        if sg["p"] < thr:
            for d in sg["dets"]:
                acc[d].append(dict(sg))
    return acc, segs, thr, nullMin


def fillGaps(t, gaps, rng):
    """As run_precursor_artefacts.correctTimes without dead time: photons added at random in each gap at the
    detector's local rate (the FILL_SIDE_S on each side), for the band times t of one detector."""
    add = [t]
    for g in gaps:
        a, b = g["start"], g["stop"]
        n = (np.searchsorted(t, a) - np.searchsorted(t, a - FILL_SIDE_S)) + \
            (np.searchsorted(t, b + FILL_SIDE_S) - np.searchsorted(t, b, "right"))
        k = rng.poisson(n / (2 * FILL_SIDE_S) * (b - a))
        add.append(rng.uniform(a, b, k))
    return np.sort(np.concatenate(add))


def events(intervals):
    """Overlapping or touching (start, stop) intervals joined into events."""
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def inakiGaps(S, dets, window):
    path = os.path.join(SETTINGS_DIR, S["inakiGapsFile"])
    rows = list(csv.DictReader(open(path)))
    return [dict(det=r["detector"], start=float(r["t_start_s"]), stop=float(r["t_stop_s"])) for r in rows
            if r["burst"] == S["inakiGapsBurst"] and r["detector"] in dets
            and overlaps(float(r["t_start_s"]), float(r["t_stop_s"]), *window)]


def gapStep(S, B, label, window, dets):
    scan = (window[0] - GAP_MARGIN_S, window[1] + GAP_MARGIN_S)
    acc, segs, thr, nullMin = findGaps(B["allT"], scan, [SEED, 930, crc(S["burst"]), crc(label)])
    inWin = [sg for sg in segs if sg["p"] < thr and overlaps(sg["start"], sg["stop"], *window)]
    mine = [sg for sg in inWin if set(sg["dets"]) & set(dets)]
    perDet = {d: [g for g in acc[d] if overlaps(g["start"], g["stop"], *window)] for d in dets}
    his = inakiGaps(S, dets, window)
    evMine, evHis = events([(g["start"], g["stop"]) for g in mine]), events([(g["start"], g["stop"]) for g in his])
    unmatchedMine = [e for e in evMine if not any(overlaps(*e, *f) for f in evHis)]
    unmatchedHis = [e for e in evHis if not any(overlaps(*e, *f) for f in evMine)]
    return dict(scan=scan, threshold=thr, nullMinima=nullMin, acceptedInWindow=inWin, inChosenDetectors=mine,
                perDetector=perDet, inaki=his, eventsMine=evMine, eventsInaki=evHis,
                unmatchedMine=unmatchedMine, unmatchedInaki=unmatchedHis,
                flagGap=bool(mine), flagMismatch=bool(unmatchedMine or unmatchedHis))


# ================================================================ step 6: dead time
def gdtLive(B, d, edges):
    """Iñaki's live_fraction(): GDT PHAII exposure / bin width (all channels, 2.6 us per event, 10 us overflow)."""
    from gdt.core.binning.unbinned import bin_by_edges
    ph = B["tte"][d].to_phaii(bin_by_edges, edges, time_range=(edges[0], edges[-1]))
    return np.asarray(ph.data.exposure) / np.diff(edges)


def ownLive(B, d, edges):
    """The same from the photon counts (cross-check of the GDT number)."""
    t, c = B["allT"][d], B["allC"][d]
    ov = c >= OVERFLOW_CHANNEL
    nN = np.histogram(t[~ov], bins=edges)[0]
    nO = np.histogram(t[ov], bins=edges)[0]
    return 1.0 - (nN * DEAD_S + nO * DEAD_OVERFLOW_S) / np.diff(edges)


def liveStep(B, band, window, dets):
    edges = binEdges(*window, LIVE_BIN_S) if window[1] - window[0] >= 2 * LIVE_BIN_S else np.array(window)
    num = den = 0.0
    per = {}
    for d in dets:
        f = gdtLive(B, d, edges)
        g = ownLive(B, d, edges)
        n = np.histogram(inWindow(B["band"][band][d], window), bins=edges)[0]
        whole = float(gdtLive(B, d, np.array(window, float))[0])
        per[d] = dict(min16ms=float(f.min()), mean16ms=float(f.mean()), whole=whole,
                      maxDiffOwnVsGdt=float(np.max(np.abs(f - g))))
        num += float(np.sum(n * f)); den += float(np.sum(n))
    weighted = num / den if den > 0 else 1.0
    return dict(perDetector=per, photonWeighted=weighted, flag=weighted < LIVE_MIN)


# ================================================================ steps 9-10: SNR and class
def mergedTimes(B, dets, window, band):
    """Bala et al.'s combination (GbmTte.merge, default force_unique=True): used ONLY for the class."""
    from gdt.missions.fermi.gbm.tte import GbmTte
    t = np.array(GbmTte.merge([B["tte"][d].slice_time(window) for d in dets]).slice_energy(band).data.times)
    return np.sort(t[(t >= window[0]) & (t < window[1])])


def snrStep(S, B, band, window, dets, mvtMs, label, complete=True, bkgFn=None, main=WINDOW_BKG):
    """SNR_MVT with the main background model (windows: cubic); also with the other model as a check.
    bkgFn replaces the background average (regression rows: run_grb_mvt.py's average)."""
    mvtS = mvtMs / 1e3
    rate = bkgFn or (lambda key, d, b: bkgRate(B, key, d, window, b))
    times = windowTimes(B, band, dets, window)
    bkgBand, bkgAll = sum(rate(main, d, band) for d in dets), sum(rate(main, d, None) for d in dets)
    counts = lightCurve(times, mvtS, *window, complete)
    phases = [lightCurve(times, mvtS, *window, complete, k / N_PHASES).max() / np.sqrt(bkgBand * mvtS)
              for k in range(N_PHASES)]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 800, crc(S["burst"]), crc(label)]))
    pmax = np.array([rng.poisson(counts).max() for _ in range(N_POISSON)]) / np.sqrt(bkgBand * mvtS)
    merged = mergedTimes(B, dets, window, band)
    mMax = lightCurve(merged, mvtS, *window, complete).max()
    out = dict(mvt_ms=mvtMs, bkgModel=main, bkgCps_band=bkgBand, bkgCps_allChannels=bkgAll, maxCounts=int(counts.max()),
               bkgCountsPerBin=bkgBand * mvtS, physical=float(counts.max() / np.sqrt(bkgBand * mvtS)),
               phaseMin=float(min(phases)), phaseMax=float(max(phases)), poissonMean=float(pmax.mean()),
               poissonStd=float(pmax.std()), bala=float(mMax / np.sqrt(bkgAll * mvtS)),
               mergeLossPct=100.0 * (1 - merged.size / times.size) if times.size else 0.0)
    if bkgFn is None:
        alt = ALT_BKG if main == WINDOW_BKG else WINDOW_BKG
        aBand = sum(bkgRate(B, alt, d, window, band) for d in dets)
        aAll = sum(bkgRate(B, alt, d, window, None) for d in dets)
        out.update(altBkgModel=alt, bkgCps_band_alt=aBand, bkgCps_allChannels_alt=aAll,
                   physical_altBkg=float(counts.max() / np.sqrt(aBand * mvtS)),
                   bala_altBkg=float(mMax / np.sqrt(aAll * mvtS)))
    return out


SHORT_NAMES = {"Below 95% CI (Upper Limit)": "UL", "Within 95% CI (Likely Upper Limit)": "LUL",
               "Above 95% CI (Robust Measurement)": "robust"}


def classify(ctx, mvtMs, snr):
    interpolators, rangeLog = ctx["model"]
    label, bounds = ctx["validation"].classify_mvt_value(mvtMs, snr, interpolators, rangeLog)
    lo, mid, hi = (float(10 ** x) for x in bounds)
    return dict(cls=SHORT_NAMES.get(label, label), band=[lo, mid, hi],
                outsideCurve=not (rangeLog[0] <= np.log10(mvtMs) <= rangeLog[1]))


# ================================================================ step 12: made-to-measure check
_HM = None                                  # haarMethods for the worker processes (inherited when they start)


def mtmShape(key, mvtMs, window):
    """The two fake pulses with the window's MVT (design doc, step 12; as run_precursor_check.py):
    G: Gaussian, sigma = MVT / 0.72, centred in the window;
    N: Norris rise 30 ms / decay 100 ms scaled in time by MVT / 17.54 ms, starting 1/6 of the window after its start."""
    w0, w1 = window
    if key == "G":
        return dict(shape="gaussian", sigma=mvtMs / MTM_MVT_PER_SIGMA / 1e3, centre=0.5 * (w0 + w1))
    f = mvtMs / MTM_N30_100_MS
    return dict(shape="norris", rise=30e-3 * f, decay=100e-3 * f, start=w0 + MTM_NORRIS_START * (w1 - w0))


_norrisCache = {}


def norrisCumulative(p, window):
    """Cumulative Norris rate (peak 1) on a 0.3 us grid from the pulse start to the window end (or 30 decay times,
    beyond which the rate is below exp(-29)); as run_injection_recovery.norrisCumulative."""
    key = (p["rise"], p["decay"], p["start"], window[1])
    if key not in _norrisCache:
        end = min(window[1], p["start"] + 30 * p["decay"])
        g = p["start"] + MTM_NORRIS_GRID_S * np.arange(int(np.ceil((end - p["start"]) / MTM_NORRIS_GRID_S)) + 1)
        x = g - p["start"]
        r = np.zeros_like(g)
        m = x > 0
        r[m] = np.exp(2 * np.sqrt(p["rise"] / p["decay"]) - p["rise"] / x[m] - x[m] / p["decay"])
        cum = np.cumsum(r) * MTM_NORRIS_GRID_S
        i0 = max(int(np.argmax(cum > 0)) - 1, 0)        # one leading zero, so that cum is strictly increasing
        _norrisCache[key] = (g[i0:], cum[i0:])
    return _norrisCache[key]


def mtmSimulate(p, amp, bkg, window, seed):
    """One fake observation (as run_precursor_check.simulate): flat Poisson background at the window's rate plus the
    pulse, peak source rate = amp x background; photons outside the window dropped."""
    rng = np.random.default_rng(seed)
    w0, w1 = window
    nb = rng.poisson(bkg * (w1 - w0))
    tb = rng.uniform(w0, w1, nb)
    if p["shape"] == "gaussian":
        n = rng.poisson(amp * bkg * p["sigma"] * np.sqrt(2 * np.pi))
        ts = rng.normal(p["centre"], p["sigma"], n)
    else:
        g, cum = norrisCumulative(p, window)
        n = rng.poisson(amp * bkg * cum[-1])
        ts = np.interp(rng.uniform(0.0, cum[-1], n), cum, g)
    ts = ts[(ts >= w0) & (ts < w1)]
    return np.sort(np.concatenate([tb, ts]))


def mtmDeadParams(B, band, window, dets):
    """Measured dead time of the real window, in the form the fakes need (as run_precursor_artefacts.py, with the
    GDT live time of step 6): share_d = detector d's share of the window's photons; c_d = dead fraction per unit
    8-900 keV count rate of d (sum over the 16 ms bins of 1 - live, divided by the sum of the rate)."""
    edges = binEdges(*window, LIVE_BIN_S) if window[1] - window[0] >= 2 * LIVE_BIN_S else np.array(window, float)
    n = {d: np.histogram(inWindow(B["band"][band][d], window), bins=edges)[0] for d in dets}
    tot = sum(int(v.sum()) for v in n.values())
    share, c = {}, {}
    for d in dets:
        share[d] = float(n[d].sum() / tot)
        rate = n[d] / np.diff(edges)
        c[d] = float(np.sum(1.0 - gdtLive(B, d, edges)) / np.sum(rate)) if np.sum(rate) > 0 else 0.0
    return dict(share=share, c=c)


def mtmDeadTime(t, dead, window, rng):
    """Dead time applied to a fake: in each 1 ms slice, a photon is removed with probability
    sum_d share_d x min(c_d x share_d x R, 1), R = the fake's own summed rate in the slice (at most 0.9)."""
    w0, w1 = window
    n = max(int(np.ceil((w1 - w0) / MTM_SLICE_S - 1e-6)), 1)
    edges = w0 + MTM_SLICE_S * np.arange(n + 1)
    edges[-1] = w1
    R = np.histogram(t, bins=edges)[0] / np.diff(edges)
    pd = sum(s * np.minimum(dead["c"][d] * s * R, 1.0) for d, s in dead["share"].items())
    idx = np.clip(np.searchsorted(edges, t, "right") - 1, 0, R.size - 1)
    return t[rng.uniform(size=t.size) >= np.clip(pd[idx], 0.0, 0.9)]


def mtmPhotons(p, amp, bkg, window, seed, dead, dseed):
    t = mtmSimulate(p, amp, bkg, window, seed)
    return mtmDeadTime(t, dead, window, np.random.default_rng(dseed)) if dead else t


def mtmSnr(t, window, mvtS, bkg):
    """Physical SNR_MVT exactly as step 9: maximum counts in complete bins of width MVT from the window start, over
    sqrt(background counts per bin), background = the window's fitted rate (the rate the fakes were drawn at)."""
    return float(lightCurve(t, mvtS, *window).max() / np.sqrt(bkg * mvtS))


def _mtmSnrTask(a):
    p, amp, bkg, window, seed, dead, dseed, mvtS = a
    return mtmSnr(mtmPhotons(p, amp, bkg, window, seed, dead, dseed), window, mvtS, bkg)


def _mtmTruthTask(a):
    """One bright fake, one MVT (haarPowerMod, errors sqrt(counts)) at the quoted bin width."""
    global _HM
    p, amp, bkg, window, seed, bw = a
    logging.disable(logging.CRITICAL)
    if _HM is None:
        _HM = loadHaar()
    c = lightCurve(mtmSimulate(p, amp, bkg, window, seed), bw, *window).astype(float)
    try:
        r = _HM.haarPowerMod(c, np.sqrt(c), minimumBinSizeSeconds=bw)
        return float(r[2]), float(r[3])
    except Exception:
        return float("nan"), float("nan")


def mtmInputs(rec):
    q = rec["mc"][f"{BIN_WIDTHS[-1]:g}"]
    return dict(window_s=list(rec["window_s"]), band_keV=list(rec["band_keV"]), detectors=list(rec["detectors"]),
                mvt_ms=q["median_ms"], validPercent=q["validPercent"], snr_physical=rec["snr"]["physical"],
                bkgCps_band=rec["snr"]["bkgCps_band"], deadtime="deadtime" in rec["flags"])


def sameInputs(a, b):
    num = ("mvt_ms", "validPercent", "snr_physical", "bkgCps_band")
    return all(a.get(k) == b.get(k) for k in a if k not in num) and all(
        a.get(k) is not None and b.get(k) is not None and abs(a[k] - b[k]) <= 1e-9 * max(abs(a[k]), abs(b[k]), 1e-300)
        for k in num)


def madeToMeasure(ctx, S, B, label, window, band, informational):
    """Step 12. A real window has no answer sheet, so the method is tested on fakes whose true MVT is known, built in
    the window's own conditions and measured exactly as the window was (design doc, step 12; run_precursor_check.py
    generalised; fixed before running):
      shapes      G and N with the window's MVT (mtmShape); flat Poisson background at the window's fitted rate.
      variants    clean; and, if the window has the deadtime flag, 'dead' (the measured dead time applied: mtmDeadTime).
      brightness  per shape and variant, the peak amplitude for which the fakes' mean physical SNR_MVT (at the window's
                  MVT, as step 9) equals the window's: 3 iterations of 100 fakes, amp x target / mean after each.
      truth R     per shape: median MVT of 300 clean fakes 30 times brighter than the clean match (0.003 ms bins).
      fakes       per shape and variant: 20 fakes at the matched brightness, each analysed as the real window
                  (complete 0.003 ms bins, 300 Poisson resamples, resampleErrors=True).
      rules       (i) the median over the 20 of (median / R) within 1/1.2 - 1.2; (ii) R inside the 16-84% range in at
                  least 10 of 20 (a fake with no valid value counts as a miss). PASS if (i) and (ii) hold for both
                  shapes, in every variant run. Reported only: realism (clean fakes' mean valid share within 15
                  points of the window's, else the realism flag), calibration (mean SNR within 15% of the target).
      seeds       SeedSequence([2026, 98x, crc(burst), crc(label), shape, variant, k]).
    The result is check.json next to record.json; the window's record is then rebuilt (--post) so that the decision
    and the trusted column use it. informational: the window is class UL or LUL (quoted as an upper limit whatever
    the result; the design does not run the check there), run only because it was named with --windows."""
    global _HM
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    folder = os.path.join(RESULTS, "windows", S["burst"], bandTag(band), label)
    rec = json.load(open(os.path.join(folder, "record.json")))
    inp = mtmInputs(rec)
    dets, mvtMs, target, bkg = inp["detectors"], inp["mvt_ms"], inp["snr_physical"], inp["bkgCps_band"]
    mvtS, bw = mvtMs / 1e3, BIN_WIDTHS[-1]
    # the record must describe these data: the window's physical SNR rebuilt from the photons
    again = float(lightCurve(windowTimes(B, band, dets, window), mvtS, *window).max() /
                  np.sqrt(sum(bkgRate(B, WINDOW_BKG, d, window, band) for d in dets) * mvtS))
    if abs(again / target - 1) > 1e-9 or list(window) != inp["window_s"]:
        sys.exit(f"STOP (step 12): record.json of {label} does not match the data (SNR {again} vs {target}); "
                 "run the window again first")
    dead = mtmDeadParams(B, band, window, dets) if inp["deadtime"] else None
    variants = ["clean"] + (["dead"] if dead else [])
    shapes = {k: mtmShape(k, mvtMs, window) for k in ("G", "N")}
    cdir = os.path.join(folder, "check")
    calPath = os.path.join(cdir, "calibration.json")
    print(f"\n========== step 12: {S['burst']} {label}  [{window[0]:g}, {window[1]:g}] s ==========")
    print(f"real window: MVT {mvtMs:.3f} ms, physical SNR {target:.1f}, background {bkg:.0f} c/s, valid "
          f"{inp['validPercent']:.0f}%, detectors {dets}; variants {variants}" + ("; INFORMATIONAL (class "
          f"{rec['class']['cls']}: quoted as an upper limit whatever the result)" if informational else ""))
    g, n = shapes["G"], shapes["N"]
    print(f"shapes: G sigma {g['sigma'] * 1e3:.2f} ms centred at {g['centre']:.4f} s; N rise {n['rise'] * 1e3:.2f} ms, "
          f"decay {n['decay'] * 1e3:.2f} ms from {n['start']:.4f} s")
    if dead:
        print("dead time: " + ", ".join(f"{d} share {dead['share'][d]:.3f}, c {dead['c'][d] * 1e6:.3f} us"
                                         for d in dets))
    _HM = ctx["haar"]
    ss = lambda *w: np.random.SeedSequence([SEED, *w, crc(S["burst"]), crc(label)])
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        # brightness
        if os.path.exists(calPath):
            cal = json.load(open(calPath))
            if not sameInputs(cal["inputs"], inp):
                sys.exit(f"STOP (step 12): {cdir} was made for other inputs ({cal['inputs']}); move it away first")
            print("calibration: saved, reused")
        else:
            cal = dict(inputs=inp, deadParams=dead, shapes=shapes, amp={})
            for si, key in enumerate(shapes):
                for vi, var in enumerate(variants):
                    seeds = ss(980, si, vi).spawn(MTM_N_CAL)
                    dseeds = ss(981, si, vi).spawn(MTM_N_CAL)
                    amp, hist = MTM_AMP0, []
                    for it in range(MTM_CAL_ITER):
                        snr = np.array(list(pool.map(_mtmSnrTask, [(shapes[key], amp, bkg, window, s, dead if var == "dead"
                                                                    else None, d, mvtS) for s, d in zip(seeds, dseeds)],
                                                     chunksize=4)))
                        hist.append(dict(amp=amp, meanSnr=float(snr.mean())))
                        if it < MTM_CAL_ITER - 1:
                            amp *= target / snr.mean()
                    cal["amp"][f"{key}_{var}"] = dict(amp=hist[-1]["amp"], meanSnr=hist[-1]["meanSnr"], history=hist,
                                                      ok=abs(hist[-1]["meanSnr"] / target - 1) <= MTM_CAL_TOL)
                    print(f"calibration {key} {var}: " + ", ".join(f"amp {h['amp']:.4g} -> SNR {h['meanSnr']:.1f}"
                                                                   for h in hist) + f" (target {target:.1f})", flush=True)
            saveJson(calPath, cal)
        # truth R
        truth = {}
        for si, key in enumerate(shapes):
            path = os.path.join(cdir, f"truth_{key}.json")
            if os.path.exists(path):
                truth[key] = json.load(open(path))
                continue
            amp = MTM_BRIGHT * cal["amp"][f"{key}_clean"]["amp"]
            tt = time.time()
            res = list(pool.map(_mtmTruthTask, [(shapes[key], amp, bkg, window, s, bw)
                                                for s in ss(982, si).spawn(MTM_N_TRUTH)], chunksize=2))
            mv = np.array([r[0] for r in res])
            un = np.array([r[1] for r in res])
            ok = np.isfinite(mv) & (mv > 0) & (un > 0)
            truth[key] = dict(amp=amp, R_ms=float(np.median(mv[ok]) * 1e3) if ok.sum() >= 2 else None,
                              p16_ms=float(np.percentile(mv[ok], 16) * 1e3) if ok.sum() >= 2 else None,
                              p84_ms=float(np.percentile(mv[ok], 84) * 1e3) if ok.sum() >= 2 else None,
                              validPercent=100.0 * ok.sum() / mv.size, mvt_ms=(mv * 1e3).tolist(),
                              unc_ms=(un * 1e3).tolist(), minutes=(time.time() - tt) / 60)
            saveJson(path, truth[key])
            print(f"truth {key}: R = {fmt(truth[key]['R_ms'], '.3f')} ms ({truth[key]['validPercent']:.0f}% valid of "
                  f"{MTM_N_TRUTH} at x{MTM_BRIGHT:g})  [{truth[key]['minutes']:.1f} min]", flush=True)

    # the 20 fakes of each shape and variant, analysed as the real window
    h = ctx["haar"]
    fakes = {}
    for si, key in enumerate(shapes):
        for vi, var in enumerate(variants):
            amp = cal["amp"][f"{key}_{var}"]["amp"]
            seeds, dseeds = ss(990, si).spawn(MTM_K), ss(993, si).spawn(MTM_K)   # same photon seeds in both variants
            rows = []
            for k in range(MTM_K):
                path = os.path.join(cdir, "fakes", f"{key}_{var}_{k:02d}.json")
                if os.path.exists(path):
                    rows.append(json.load(open(path)))
                    continue
                tt = time.time()
                t = mtmPhotons(shapes[key], amp, bkg, window, seeds[k], dead if var == "dead" else None, dseeds[k])
                c = lightCurve(t, bw, *window).astype(float)
                r = h.estimateMvtDistribution(c, np.sqrt(c), h.haarPowerMod, binSizeSeconds=bw,
                                              nRealisations=N_REALISATIONS, nWorkers=args.workers,
                                              seed=[SEED, 992, crc(S["burst"]), crc(label), si, vi, k],
                                              resampleErrors=True)
                fin = lambda x: float(x) * 1e3 if np.isfinite(x) else None
                row = dict(k=k, counts=int(c.sum()), snr=mtmSnr(t, window, mvtS, bkg), median_ms=fin(r["median"]),
                           p16_ms=fin(r["p16"]), p84_ms=fin(r["p84"]),
                           validPercent=100.0 * r["nValid"] / r["nRealisations"], minutes=(time.time() - tt) / 60)
                saveJson(path, row)
                print(f"   {key} {var:5s} fake {k:2d}: median {fmt(row['median_ms'], '.3f')} ms "
                      f"({fmt(row['p16_ms'], '.3f')}-{fmt(row['p84_ms'], '.3f')}), valid {row['validPercent']:.0f}%, "
                      f"SNR {row['snr']:.1f}  [{row['minutes']:.1f} min]", flush=True)
                rows.append(row)
            fakes[f"{key}_{var}"] = rows

    # rules
    out = dict(burst=S["burst"], label=label, informational=informational, inputs=inp, shapes=shapes, deadParams=dead,
               calibration=cal["amp"], truth={k: {x: v[x] for x in ("amp", "R_ms", "p16_ms", "p84_ms", "validPercent")}
                                             for k, v in truth.items()}, results={},
               rules=dict(ratioWithin=[1 / ONE_STEP, ONE_STEP], coverMin=MTM_COVER_MIN, of=MTM_K,
                          realismPoints=MTM_REALISM_PTS), scriptMd5=ctx["scriptMd5"], mvttoolsCommit=ctx["commit"])
    allPass, realismOk = True, True
    print(f"\nstep 12 rules ({S['burst']} {label}; real window: MVT {mvtMs:.3f} ms, valid {inp['validPercent']:.0f}%)")
    for name, rows in fakes.items():
        key = name.split("_")[0]
        R = truth[key]["R_ms"]
        med = np.array([r["median_ms"] if r["median_ms"] is not None else np.nan for r in rows], float)
        if R is None:
            res = dict(R_ms=None, ratio=None, cover=0, noValid=int(np.isnan(med).sum()), i=False, ii=False,
                       validMean=float(np.mean([r["validPercent"] for r in rows])))
        else:
            ratio = float(np.nanmedian(med / R)) if np.isfinite(med).any() else None
            cover = int(sum(r["p16_ms"] is not None and r["p16_ms"] <= R <= r["p84_ms"] for r in rows))
            res = dict(R_ms=R, ratio=ratio, ratioSpread=[float(x) for x in np.nanpercentile(med / R, [16, 84])]
                       if np.isfinite(med).any() else None, cover=cover, noValid=int(np.isnan(med).sum()),
                       i=bool(ratio is not None and 1 / ONE_STEP <= ratio <= ONE_STEP), ii=cover >= MTM_COVER_MIN,
                       validMean=float(np.mean([r["validPercent"] for r in rows])))
        res["snrMean"] = float(np.mean([r["snr"] for r in rows]))
        res["realism"] = abs(res["validMean"] - inp["validPercent"]) <= MTM_REALISM_PTS
        out["results"][name] = res
        allPass &= res["i"] and res["ii"]
        if name.endswith("_clean"):
            realismOk &= res["realism"]
        print(f"   {name:8s} R {fmt(R, '.3f')} ms | (i) median of median/R {fmt(res['ratio'], '.3f')} -> "
              f"{'PASS' if res['i'] else 'FAIL'} | (ii) R inside 16-84% in {res['cover']}/{MTM_K} -> "
              f"{'PASS' if res['ii'] else 'FAIL'} | no valid value {res['noValid']} | mean valid "
              f"{res['validMean']:.0f}% (real {inp['validPercent']:.0f}%) | mean SNR {res['snrMean']:.1f} "
              f"(target {target:.1f})")
    out.update({"pass": bool(allPass), "realismOk": bool(realismOk), "minutes": (time.time() - t0) / 60})
    saveJson(os.path.join(folder, "check.json"), out)
    print(f"step 12 verdict: {'PASS' if allPass else 'FAIL'}; realism {'ok' if realismOk else 'FLAG'}"
          + ("  (informational: the decision stays an upper limit)" if informational else "")
          + f"  [{out['minutes']:.1f} min]\nSaved {os.path.join(folder, 'check.json')}")
    return out


# ================================================================ one window
def processWindow(ctx, S, B, label, window, band):
    folder = os.path.join(RESULTS, "windows", S["burst"], bandTag(band), label)
    isMain = tuple(band) == ENERGY_MAIN
    rec = dict(burst=S["burst"], label=label, window_s=list(window), band_keV=list(band), status="measured",
               settings=os.path.basename(S["_path"]), settingsMd5=S["_md5"], windowsMd5=ctx["windowsMd5"],
               script=os.path.basename(__file__), scriptMd5=ctx["scriptMd5"], mvttoolsCommit=ctx["commit"],
               background=BKG_NAMES[WINDOW_BKG], flags=[], notes=[])
    flags = rec["flags"]
    L = window[1] - window[0]
    print(f"\n========== {S['burst']} {label}  [{window[0]:g}, {window[1]:g}] s  ({L:g} s, {bandTag(band)}) ==========")
    if not (B["coverage"][0] < window[0] and window[1] < B["coverage"][1]):
        sys.exit(f"STOP: window {label} is outside the TTE data")

    # step 3
    excl = [e for e in S["excluded"] if overlaps(*window, e["start"], e["stop"])]
    if excl:
        rec.update(status="excluded", excludedBy=excl)
        print(f"step 3: overlaps excluded interval {excl} -> not measured, no row")
        saveJson(os.path.join(folder, "record.json"), rec)
        return rec
    for p in S["pileup"]:
        if overlaps(*window, p["start"], p["stop"]):
            flags.append(f"pileup-{p['det']}")
    for a, b in S["bkgIntervals"] + S["inakiBkgIntervals"]:
        if overlaps(*window, a, b):
            rec["notes"].append(f"window overlaps the background interval [{a}, {b}]")
    print(f"step 3: excluded no; pile-up flags {[f for f in flags if f.startswith('pileup')] or 'none'}")

    # steps 4 and 7
    firstBin = SELECT_BIN_SHORT_S if L < SHORT_WINDOW_S else SELECT_BIN_LONG_S
    print(f"step 4: detector selection, first with {firstBin * 1e3:g} ms bins")
    sel = dict(first=selectDetectors(B, band, window, firstBin), rounds=[])
    cur, tried, converged, switching = sel["first"]["optimal"], [], False, False
    for rnd in range(1, MAX_ROUNDS + 1):
        tried.append(cur)
        print(f"step 7: round {rnd}, detectors {cur}")
        res = {bw: mvtFor(ctx, B, band, window, cur, bw, folder) for bw in BIN_WIDTHS}
        m = res[BIN_WIDTHS[-1]]["median_ms"]
        entry = dict(round=rnd, detectors=cur, mvt_ms=m)
        sel["rounds"].append(entry)
        if not (m and np.isfinite(m) and m > 0):
            rec["notes"].append("no valid MVT at 0.003 ms: detector selection not repeated")
            break
        if L / (m / 1e3) < 2:
            rec["notes"].append("MVT longer than half the window: detector selection not repeated")
            break
        nxt = selectDetectors(B, band, window, m / 1e3)
        entry["reselection"] = nxt
        if nxt["optimal"] == cur:
            converged = True
            print(f"   same set at bins = MVT ({m:.3f} ms): converged")
            break
        if nxt["optimal"] in tried or rnd == MAX_ROUNDS:
            switching = True
            break
        print(f"   set changed at bins = MVT ({m:.3f} ms): {cur} -> {nxt['optimal']}")
        cur = nxt["optimal"]
    if switching:
        lastMvt = sel["rounds"][-1]["mvt_ms"] / 1e3
        snrs = {setTag(t): combinedSnr(B, band, window, t, lastMvt) for t in tried}
        cur = max(tried, key=lambda t: snrs[setTag(t)])
        sel["switchingSnrAtLastMvt"] = snrs
        flags.append("detector-switching")
        print(f"   detector-switching: sets tried {tried}; highest SNR at the last MVT: {cur}")
    dets = cur
    sel.update(used=dets, converged=converged)
    saveJson(os.path.join(folder, "selection.json"), sel)
    main = {bw: mvtFor(ctx, B, band, window, dets, bw, folder) for bw in BIN_WIDTHS}
    q = main[BIN_WIDTHS[-1]]
    valid = bool(q["median_ms"] and np.isfinite(q["median_ms"]) and q["median_ms"] > 0)
    rec.update(detectors=dets, selection=dict(first=sel["first"]["optimal"], rounds=[(r["detectors"], r["mvt_ms"])
               for r in sel["rounds"]], converged=converged), mc={f"{bw:g}": main[bw] for bw in BIN_WIDTHS})

    print("step 4: comparison detector sets at 0.003 ms")
    comp = {}
    for name, cd in S["comparisonSets"].items():
        cd = sorted(cd, key=ALL_NAI.index)
        comp[name] = mvtFor(ctx, B, band, window, cd, BIN_WIDTHS[-1], folder)
    rec["comparison"] = {k: dict(detectors=v["detectors"], median_ms=v["median_ms"], p16_ms=v["p16_ms"],
                                 p84_ms=v["p84_ms"], validPercent=v["validPercent"]) for k, v in comp.items()}

    # step 5
    print("step 5: data gaps")
    gp = gapStep(S, B, label, window, dets)
    rec["gaps"] = {k: v for k, v in gp.items() if k != "nullMinima"}
    for g in gp["inChosenDetectors"]:
        print(f"   gap {g['start']:+.6f} to {g['stop']:+.6f} s ({g['length_ms']:.2f} ms) in {g['dets']}: "
              f"{g['expected']:.1f} photons expected, chance {g['p']:.1e} (threshold {gp['threshold']:.1e})")
    print(f"   accepted gaps in the chosen detectors: {len(gp['inChosenDetectors'])}; Iñaki's list: "
          f"{len(gp['eventsInaki'])} event(s); unmatched ours {gp['unmatchedMine']}, his {gp['unmatchedInaki']}")
    if gp["flagGap"]:
        flags.append("gap")
        fills = []
        for k in range(N_FILL):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 920, crc(S["burst"]), crc(label), k]))
            filled = {d: fillGaps(B["band"][band][d], gp["perDetector"][d], rng) for d in dets}
            fills.append(mvtFor(ctx, B, band, window, dets, BIN_WIDTHS[-1], folder, tag=f"fill{k}", extra=filled))
        fm = np.array([f["median_ms"] for f in fills])
        rec["gapFill"] = dict(medians_ms=fm.tolist(), validPercent=[f["validPercent"] for f in fills],
                              changePct=float(100 * (np.nanmean(fm) / q["median_ms"] - 1)) if valid else None,
                              seedSpreadPct=float(100 * (np.nanmax(fm) - np.nanmin(fm)) / np.nanmean(fm)))
        print(f"   gaps filled ({N_FILL} seeds): medians {', '.join(f'{x:.3f}' for x in fm)} ms vs {q['median_ms']:.3f}")
    if gp["flagMismatch"]:
        flags.append("gap-mismatch")

    # step 6
    lv = liveStep(B, band, window, dets)
    rec["live"] = lv
    print(f"step 6: photon-weighted live fraction {lv['photonWeighted']:.4f}; lowest 16 ms bin "
          + ", ".join(f"{d} {v['min16ms']:.3f}" for d, v in lv["perDetector"].items()))
    if lv["flag"]:
        flags.append("deadtime")

    # step 8
    a, b = main[BIN_WIDTHS[-2]]["median_ms"], q["median_ms"]
    ratio = b / a if (valid and a and np.isfinite(a) and a > 0) else np.nan
    plateau = bool(np.isfinite(ratio) and 1 / ONE_STEP <= ratio <= ONE_STEP)
    rec["plateau"] = dict(median_0p01ms=a, median_0p003ms=b, ratio=ratio, ok=plateau)
    print(f"step 8: medians 0.01 ms {a:.3f}, 0.003 ms {b:.3f} -> ratio {ratio:.3f} -> "
          f"{'plateau' if plateau else 'NO PLATEAU'}")
    if not valid:
        flags.append("no-valid")
    elif not plateau:
        flags.append("no-plateau")

    # steps 9-10 (at the median; with no valid realisation, at the upper-limit value, so the row still has a class)
    snrAt = q["median_ms"] if valid else q.get("limits_p84_ms")
    if snrAt:
        sn = snrStep(S, B, band, window, dets, snrAt, label)
        rec["snr"] = sn
        print(f"step 9: SNR physical {sn['physical']:.1f} (phases {sn['phaseMin']:.1f}-{sn['phaseMax']:.1f}), "
              f"Bala-compatible {sn['bala']:.1f} (background {BKG_NAMES[WINDOW_BKG]}: {sn['bkgCps_band']:.0f} c/s); "
              f"with the {BKG_NAMES[ALT_BKG]} ({sn['bkgCps_band_alt']:.0f} c/s): {sn['physical_altBkg']:.1f} / "
              f"{sn['bala_altBkg']:.1f}; merge removes {sn['mergeLossPct']:.2f}% of the photons")
        if isMain:
            c = classify(ctx, snrAt, sn["bala"])
            ci = classify(ctx, snrAt, sn["bala_altBkg"])
            rec["class"] = dict(c, withAltBkg=ci["cls"])
            print(f"step 10: class {c['cls']} (95% band {c['band'][0]:.1f}-{c['band'][2]:.1f}); with the linear "
                  f"background {ci['cls']}" + ("  (MVT outside the curve's range)" if c["outsideCurve"] else ""))
            if c["cls"] in ("LUL", "UL"):
                flags.append(f"class-{c['cls']}")
            if c["outsideCurve"]:
                flags.append("curve-range")
            if ci["cls"] != c["cls"]:
                flags.append("bkg-sensitive")

    # detector-sensitive (needs the main value)
    if valid:
        spread = {}
        for k, v in comp.items():
            r_ = v["median_ms"] / q["median_ms"] if v["median_ms"] and np.isfinite(v["median_ms"]) else np.nan
            spread[k] = r_
        rec["comparisonRatio"] = spread
        if any(not (np.isfinite(r_) and 1 / ONE_STEP <= r_ <= ONE_STEP) for r_ in spread.values()):
            flags.append("detector-sensitive")
        print("step 4: comparison sets / main: " + ", ".join(f"{k} {v:.3f}" for k, v in spread.items()))

    # step 11
    if not isMain and label in S.get("dichiara", {}):
        rec["dichiara"] = dichiaraCompare(S["dichiara"][label], q, plateau and valid)
        print(f"step 11: Dichiara et al. {rec['dichiara']['text']}")

    # step 12 (made-to-measure) is read if it has been run, and only if it was made for these very values
    chk = os.path.join(folder, "check.json")
    m2m = json.load(open(chk)) if os.path.exists(chk) else None
    if m2m and not (valid and rec.get("snr") and sameInputs(m2m["inputs"], mtmInputs(dict(rec, flags=flags)))):
        rec["notes"].append("check.json was made for other values (MVT, SNR, background, detectors or dead time): "
                            "ignored; run --part check again")
        print("step 12: check.json is for other values: IGNORED (run --part check again)")
        m2m = None
    rec["madeToMeasure"] = m2m
    if m2m is None:
        print("step 12: made-to-measure check not run")
    elif m2m.get("informational"):
        print(f"step 12: made-to-measure {'PASS' if m2m['pass'] else 'FAIL'}, realism "
              f"{'ok' if m2m['realismOk'] else 'differs'} (informational only: class "
              f"{rec.get('class', {}).get('cls')}, the decision rule is unchanged)")
    else:
        print(f"step 12: made-to-measure {'PASS' if m2m['pass'] else 'FAIL'}, realism "
              f"{'ok' if m2m['realismOk'] else 'FLAG'}")
        if not m2m.get("pass"):
            flags.append("mtm-fail")
        if not m2m.get("realismOk", True):
            flags.append("realism")

    # step 13, 14
    rec["systematics"] = systematics(rec)
    rec.update(decide(rec, isMain))
    rec["flags"] = sorted(set(flags), key=lambda f: FLAG_ORDER.index(f) if f in FLAG_ORDER else 99)
    saveJson(os.path.join(folder, "record.json"), rec)
    d = rec["decision"]
    print(f"step 14: {d['kind']}: mvt = {d['mvt_ms']:.3f} ms" + (f" (-{d['err_lo_ms']:.3f}/+{d['err_hi_ms']:.3f})"
          if d["kind"] == "value" else "") + f"; trusted {d['trusted']}; flags {rec['flags'] or 'none'}; "
          f"sys {rec['systematics']['note']}")
    return rec


def dichiaraCompare(ref, q, usable):
    ours, up = q["median_ms"], q["plus_err_ms"]
    if not usable:
        return dict(diffSigma=None, check="ours-upper-limit", text=f"ours is an upper limit ({q['p84_ms']:.2f} ms)")
    if ref["limit"]:
        ok = ours <= ref["mvt_ms"]
        return dict(diffSigma=None, check="consistent" if ok else "inconsistent",
                    text=f"theirs < {ref['mvt_ms']:g} ms, ours {ours:.2f} ms -> {'consistent' if ok else 'NOT consistent'}")
    err = up if ours < ref["mvt_ms"] else q["minus_err_ms"]
    z = (ours - ref["mvt_ms"]) / np.hypot(err, ref["err_ms"])
    return dict(diffSigma=float(z), check="consistent" if abs(z) <= 2 else "inconsistent",
                text=f"{ref['mvt_ms']:g} ± {ref['err_ms']:g} ms vs ours {ours:.2f} ms: {z:+.1f} sigma")


def systematics(rec):
    out, parts = {}, []
    if rec.get("comparisonRatio"):
        out["detectorsPct"] = {k: 100 * (v - 1) for k, v in rec["comparisonRatio"].items()}
        parts.append("det:" + "/".join(f"{v:+.1f}%" if np.isfinite(v) else "n/a" for v in out["detectorsPct"].values()))
    if np.isfinite(rec["plateau"]["ratio"]):
        out["binWidthPct"] = 100 * (rec["plateau"]["ratio"] - 1)
        parts.append(f"bw:{out['binWidthPct']:+.1f}%")
    if rec.get("gapFill") and rec["gapFill"]["changePct"] is not None:
        out["gapPct"] = rec["gapFill"]["changePct"]
        parts.append(f"gap:{out['gapPct']:+.1f}%")
    out["methodPct"] = 100 * METHOD_SYS
    parts.append(f"method:{out['methodPct']:.0f}%")
    if rec.get("snr") and "physical_altBkg" in rec["snr"]:
        out["bkgSnrPct"] = 100 * (rec["snr"]["physical_altBkg"] / rec["snr"]["physical"] - 1)
        parts.append(f"bkgSNR:{out['bkgSnrPct']:+.1f}%")
    out["note"] = ";".join(parts)
    return out


def decide(rec, isMain):
    """Step 14, fixed before running. Upper limit if: no valid MVT, class LUL or UL, no plateau, or the
    made-to-measure check failed. Otherwise a value; trusted = true only once the made-to-measure check passed."""
    q = rec["mc"][f"{BIN_WIDTHS[-1]:g}"]
    f = set(rec["flags"])
    reasons = [x for x in ("no-valid", "class-LUL", "class-UL", "no-plateau", "mtm-fail") if x in f]
    if "no-valid" in f:
        lim = q.get("limits_p84_ms")
        if lim is None:
            return dict(decision=dict(kind="failed", mvt_ms=float("nan"), trusted="false", reasons=reasons))
        return dict(decision=dict(kind="upper limit", mvt_ms=lim, trusted="false", reasons=reasons,
                                  note="no valid realisation: 84th percentile of the upper-limit realisations"))
    if reasons:
        return dict(decision=dict(kind="upper limit", mvt_ms=q["p84_ms"], trusted="false", reasons=reasons))
    if not isMain:
        trusted = "n/a"
    elif rec["madeToMeasure"] is None or rec["madeToMeasure"].get("informational"):
        trusted = "untested"
    else:
        trusted = "true"
    return dict(decision=dict(kind="value", mvt_ms=q["median_ms"], err_lo_ms=q["minus_err_ms"],
                              err_hi_ms=q["plus_err_ms"], trusted=trusted, reasons=[]))


# ================================================================ CSV (step 14) and its checker
def fmt(x, spec):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def csvRow(rec, dichiara=False):
    d, q = rec["decision"], rec["mc"][f"{BIN_WIDTHS[-1]:g}"]
    lim = d["kind"] == "upper limit"
    row = dict(label=rec["label"], mvt_s=fmt(d["mvt_ms"] / 1e3, ".6g"),
               mvt_err_lo_s="" if lim else fmt(d["err_lo_ms"] / 1e3, ".6g"),
               mvt_err_hi_s="" if lim else fmt(d["err_hi_ms"] / 1e3, ".6g"),
               band_keV=f"{rec['band_keV'][0]:g}-{rec['band_keV'][1]:g}", upper_limit="true" if lim else "false",
               snr_physical=fmt(rec.get("snr", {}).get("physical"), ".1f"), detectors=" ".join(rec["detectors"]),
               bin_width_s=f"{BIN_WIDTHS[-1]:g}", valid_pct=fmt(q["validPercent"], ".1f"), flags=";".join(rec["flags"]),
               script=f"{rec['script']}@{rec['scriptMd5'][:8]}", mvt_commit=rec["mvttoolsCommit"].split()[0])
    if dichiara:
        ref = rec.get("dichiaraRef", {})
        dc = rec.get("dichiara", {})
        row.update(dichiara_mvt_s=fmt(ref.get("mvt_ms", np.nan) / 1e3, ".6g"),
                   dichiara_err_s=fmt((ref.get("err_ms") or np.nan) / 1e3, ".6g"),
                   dichiara_upper_limit="true" if ref.get("limit") else "false",
                   dichiara_diff_sigma=fmt(dc.get("diffSigma"), "+.2f"), dichiara_check=dc.get("check", ""))
    else:
        row.update({"class": rec["class"]["cls"], "snr_bala": fmt(rec["snr"]["bala"], ".1f"),
                    "snr_physical_linbkg": fmt(rec["snr"]["physical_altBkg"], ".1f"), "trusted": d["trusted"],
                    "sys_note": rec["systematics"]["note"]})
    return row


def checkCsv(text, columns, dichiara=False):
    """Iñaki's rules (his points 20-22). Returns the list of problems (empty = good)."""
    bad = []
    lines = text.split("\n")
    if lines[-1] != "":
        bad.append("file must end with a newline")
    lines = [l for l in lines if l != ""]
    if not lines or lines[0] != ",".join(columns):
        bad.append("first line is not the expected header")
    if set(columns) & FORBIDDEN_COLUMNS:
        bad.append(f"columns clash with Iñaki's table: {set(columns) & FORBIDDEN_COLUMNS}")
    seen = set()
    for i, l in enumerate(lines, 1):
        if l.lstrip().startswith("#"):
            bad.append(f"line {i}: comment line")
        if ", " in l or " ," in l or '"' in l or "\t" in l:
            bad.append(f"line {i}: space next to a comma, a quote or a tab")
        f = l.split(",")
        if len(f) != len(columns):
            bad.append(f"line {i}: {len(f)} fields, expected {len(columns)}")
            continue
        if i == 1:
            continue
        r = dict(zip(columns, f))
        if any(v != v.strip() for v in f):
            bad.append(f"line {i}: field with a leading or trailing space")
        if not r["label"] or r["label"] in seen:
            bad.append(f"line {i}: empty or repeated label {r['label']}")
        seen.add(r["label"])
        if r["upper_limit"] not in ("true", "false"):
            bad.append(f"line {i}: upper_limit must be true or false")
        try:
            if not float(r["mvt_s"]) > 0:
                bad.append(f"line {i}: mvt_s not positive")
        except ValueError:
            bad.append(f"line {i}: mvt_s not a number")
        errs = (r["mvt_err_lo_s"], r["mvt_err_hi_s"])
        if r["upper_limit"] == "true" and any(errs):
            bad.append(f"line {i}: upper limit with errors")
        if r["upper_limit"] == "false":
            try:
                if not all(float(e) >= 0 and np.isfinite(float(e)) for e in errs):
                    bad.append(f"line {i}: negative or non-finite error")
            except ValueError:
                bad.append(f"line {i}: missing error for a measured value")
        if not re.fullmatch(r"\d+-\d+", r["band_keV"]):
            bad.append(f"line {i}: band_keV {r['band_keV']}")
        if not dichiara:
            if r["class"] not in ("robust", "LUL", "UL"):
                bad.append(f"line {i}: class {r['class']}")
            if r["trusted"] not in ("true", "false", "untested"):
                bad.append(f"line {i}: trusted {r['trusted']}")
        if "nan" in l.lower().split(","):
            bad.append(f"line {i}: nan written out")
    try:                                            # read it exactly as Iñaki's merge does
        import pandas as pd
    except ImportError:
        return bad
    try:
        df = pd.read_csv(io.StringIO(text)).set_index("label")
        if df.index.has_duplicates or df["upper_limit"].dtype != bool:
            bad.append("pandas: duplicated labels or upper_limit not read as true/false")
    except Exception as e:
        bad.append(f"pandas cannot read it as Iñaki's merge does: {type(e).__name__}: {e}")
    return bad


def writeCsv(S, W):
    for band, dich in ((ENERGY_MAIN, False), (tuple(S["dichiaraBand"]) if S.get("dichiaraBand") else None, True)):
        if band is None:
            continue
        base = os.path.join(RESULTS, "windows", S["burst"], bandTag(band))
        cols = CSV_COLUMNS_DICHIARA if dich else CSV_COLUMNS
        rows, skipped = [], []
        for label in W:
            p = os.path.join(base, label, "record.json")
            if not os.path.exists(p):
                continue
            rec = json.load(open(p))
            if rec["status"] != "measured" or rec["decision"]["kind"] == "failed":
                skipped.append(f"{label} ({rec['status'] if rec['status'] != 'measured' else 'no MVT'})")
                continue
            if dich:
                rec["dichiaraRef"] = S["dichiara"].get(label, {})
            rows.append(csvRow(rec, dich))
        if not rows:
            print(f"{bandTag(band)}: no measured windows yet")
            continue
        text = ",".join(cols) + "\n" + "".join(",".join(r[c] for c in cols) + "\n" for r in rows)
        bad = checkCsv(text, cols, dich)
        name = f"mvt_{S['burst'].replace('GRB', '')}" + ("_" + bandTag(band) if dich else "") + ".csv"
        if bad:
            sys.exit(f"STOP: {name} breaks the rules, not saved:\n  " + "\n  ".join(bad))
        out = os.path.join(RESULTS, "windows", S["burst"], "csv", name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write(text)
        print(f"\nSaved {out} ({len(rows)} rows; checked against Iñaki's rules: OK)")
        print(text, end="")
        if skipped:
            print(f"not written (no row by the decision rule): {', '.join(skipped)}")
        if not dich:
            unt = [r["label"] for r in rows if r["trusted"] == "untested"]
            if unt:
                print(f"PROVISIONAL: made-to-measure check not run yet for {', '.join(unt)} (trusted = untested)")


# ================================================================ regression test
def regression(ctx, S, B, W):
    R = S["regression"]
    regDir = os.path.join(RESULTS, "windows", S["burst"], "regression")
    want = set(args.rows) if args.rows else None
    rep = dict(items={}, scriptMd5=ctx["scriptMd5"], mvttoolsCommit=ctx["commit"], settingsMd5=S["_md5"])
    tol = lambda a, b, rel=1e-9: (a is None and b is None) or (a is not None and b is not None and
                                                               abs(a - b) <= rel * max(abs(a), abs(b), 1e-300))
    band = ENERGY_MAIN

    def compareMc(new, old):
        keys = ["totalCounts", "nValid", "nLimit", "nFailed"]
        fkeys = ["median_ms", "p16_ms", "p84_ms", "singleMvt_ms", "singleUnc_ms"]
        diff = [k for k in keys if new[k] != old[k]] + [k for k in fkeys if not tol(new[k], old[k], 1e-12)]
        return diff

    rowsWanted = [r for r in R["rows"] if not want or r in want]
    snrSaved = json.load(open(os.path.join(RESULTS, R["rowsSnr"])))["runs"] if rowsWanted else {}
    for run in rowsWanted:
        row = R["rows"][run]
        old = json.load(open(os.path.join(RESULTS, row["saved"])))
        dets, iv, bw = row["dets"], tuple(row["interval"]), row["bw"]
        print(f"\n--- regression row {run}: {dets}, {bw * 1e3:g} ms, {list(iv)} s")
        if old["detectors"] != dets or old["binSize_s"] != bw or list(old["timeRange_s"]) != list(iv):
            sys.exit(f"STOP: settings for row {run} do not match the saved run")
        edgesL = binEdges(*iv, bw, complete=False)

        def legacyCounts():                                    # run_grb_mvt.binDetectors: whole-file times
            c = np.zeros(len(edgesL) - 1, dtype=float)
            for d in dets:
                c += np.histogram(B["band"][band][d], bins=edgesL)[0]
            return c
        meta = dict(detectors=dets, binSize_s=bw, window_s=list(iv), band_keV=list(band), nRealisations=N_REALISATIONS,
                    fill="regression-legacy")
        new = mc(ctx, os.path.join(regDir, "mc", f"row{run}.json"), legacyCounts, meta)
        diff = compareMc(new, old)
        newBuilder = lightCurve(windowTimes(B, band, dets, iv), bw, *iv)
        lc = legacyCounts()
        nb = min(lc.size, newBuilder.size)
        builder = dict(legacyBins=int(lc.size), completeBins=int(newBuilder.size),
                       differingBins=int(np.sum(lc[:nb] != newBuilder[:nb]) + abs(lc.size - newBuilder.size)))
        s = snrSaved.get(run)
        snrOk = None
        if s is not None:
            sn = snrStep(S, B, band, iv, dets, new["median_ms"], f"row{run}", complete=False,
                         bkgFn=lambda key, d, b: legacyMeanRate(B, d, iv, b))
            c = classify(ctx, new["median_ms"], sn["bala"])
            snrOk = (tol(sn["physical"], s["snrAtOurMedian"]["observed"], 1e-9)
                     and tol(sn["bala"], s["snrBalaDefAtOurMedian"], 1e-9) and c["cls"] == s["classOursBalaDef"])
            pipe = snrStep(S, B, band, iv, dets, new["median_ms"], f"row{run}")      # windows' way: cubic, complete bins
            print(f"   SNR physical {sn['physical']:.3f} (saved {s['snrAtOurMedian']['observed']:.3f}), Bala-compatible "
                  f"{sn['bala']:.3f} (saved {s['snrBalaDefAtOurMedian']:.3f}), class {c['cls']} (saved "
                  f"{s['classOursBalaDef']}); windows' way (cubic background, complete bins) gives {pipe['physical']:.3f} / "
                  f"{pipe['bala']:.3f}")
        ok = not diff and snrOk is not False
        print(f"   MC: median {new['median_ms']:.4f} (saved {old['median_ms']:.4f}), valid {new['nValid']} (saved "
              f"{old['nValid']}) -> {'IDENTICAL' if not diff else 'DIFFERENT in ' + str(diff)}; SNR/class "
              f"{'identical' if snrOk else ('DIFFERENT' if snrOk is False else 'no saved values')}")
        print(f"   bin edges: earlier scripts {builder['legacyBins']} bins, complete bins {builder['completeBins']}; "
              f"bins that differ {builder['differingBins']}")
        rep["items"][run] = dict(ok=ok, mcDiff=diff, snrOk=snrOk, builder=builder, new=new,
                                 snr=dict(physical=sn["physical"], bala=sn["bala"], cls=c["cls"]) if s else None,
                                 bala=row["bala"], interval=list(iv), dets=dets, bw=bw)

    P = R.get("precursor")
    if P and (not want or "pre" in want):
        iv = tuple(P["interval"])
        print(f"\n--- regression precursor {list(iv)} s")
        saved = json.load(open(os.path.join(RESULTS, P["dir"], "summary.json")))
        selSaved = json.load(open(os.path.join(RESULTS, P["dir"], "selection.json")))
        okAll = True
        for oldRun in saved["runs"]:
            dets, bw = oldRun["detectors"], oldRun["binSize_s"]
            old = json.load(open(os.path.join(RESULTS, P["dir"], f"{oldRun['detectorSet']}_{bw * 1e3:g}ms", "summary.json")))
            same = np.array_equal(lightCurve(windowTimes(B, band, dets, iv), bw, *iv, complete=False),
                                  lightCurve(windowTimes(B, band, dets, iv), bw, *iv, complete=True))
            meta = dict(detectors=dets, binSize_s=bw, window_s=list(iv), band_keV=list(band),
                        nRealisations=N_REALISATIONS, fill="regression-legacy")
            new = mc(ctx, os.path.join(regDir, "mc", f"pre_{oldRun['detectorSet']}_{bw * 1e3:g}ms.json"),
                     lambda: lightCurve(windowTimes(B, band, dets, iv), bw, *iv, complete=False), meta)
            diff = compareMc(new, old)
            sv = oldRun.get("snr")
            sn = snrStep(S, B, band, iv, dets, new["median_ms"], "pre-regression", complete=False, main="bala")
            c = classify(ctx, new["median_ms"], sn["bala"])
            snrOk = sv is not None and tol(sn["physical"], sv["physical"]) and tol(sn["bala"], sv["balaCompatible"]) \
                and c["cls"] == sv["cls"]
            ok = not diff and snrOk and same
            okAll &= ok
            print(f"   {oldRun['detectorSet']} {bw * 1e3:g} ms: median {new['median_ms']:.4f} (saved {old['median_ms']:.4f}), "
                  f"valid {new['nValid']} (saved {old['nValid']}); SNR {sn['physical']:.3f}/{sn['bala']:.3f} (saved "
                  f"{sv['physical']:.3f}/{sv['balaCompatible']:.3f}), class {c['cls']} (saved {sv['cls']}); complete "
                  f"bins identical to the earlier ones: {same} -> {'IDENTICAL' if ok else 'DIFFERENT ' + str(diff)}")
            rep["items"][f"pre_{oldRun['detectorSet']}_{bw:g}"] = dict(ok=ok, mcDiff=diff, snrOk=snrOk, sameBins=same)
        sel = selectDetectors(B, band, iv, selSaved["first16ms"]["binWidth_s"], complete=False, bkgKey="bala")
        selOk = sel["optimal"] == selSaved["first16ms"]["optimal"] and all(
            tol(sel["single"][d], selSaved["first16ms"]["single"][d]) for d in ALL_NAI)
        sel2 = selectDetectors(B, band, iv, selSaved["first16ms"]["binWidth_s"], complete=True)
        print(f"   detector selection at 16 ms: {sel['optimal']} (saved {selSaved['first16ms']['optimal']}), single SNRs "
              f"{'identical' if selOk else 'DIFFERENT'}; windows' way (cubic background, complete bins): {sel2['optimal']}")
        rep["items"]["pre_selection"] = dict(ok=selOk)

    G = R.get("gaps")
    if G and (not want or "gaps" in want):
        print("\n--- regression gap search (run_precursor_artefacts.py, -1 to 2 s, seed (2026, 930))")
        params = json.load(open(os.path.join(RESULTS, G["params"])))
        acc, segs, thr, nullMin = findGaps(B["allT"], tuple(G["scan"]), [SEED, 930])
        same = tol(thr, params["gapThreshold"], 1e-12) and all(
            [(g["start"], g["stop"]) for g in acc[d]] == [(g["start"], g["stop"]) for g in params["gapsAll"][d]]
            for d in ALL_NAI)
        print(f"   threshold {thr:.3e} (saved {params['gapThreshold']:.3e}); accepted gaps per detector "
              f"{ {d: len(acc[d]) for d in ALL_NAI} } -> {'IDENTICAL' if same else 'DIFFERENT'}")
        rep["items"]["gaps"] = dict(ok=same, threshold=thr)

    if S.get("inakiLiveFile") and (not want or "live" in want):
        print("\n--- check: GDT live fractions against Iñaki's table (whole window, 12 NaI)")
        rows = list(csv.DictReader(open(os.path.join(SETTINGS_DIR, S["inakiLiveFile"]))))
        worst = 0.0
        for r in rows:
            lab = r["label"]
            if lab in W:
                iv = W[lab]
            elif lab.startswith("bala_"):
                iv = tuple(float(x) for x in lab[5:].split("_"))
            else:
                continue
            for d in ALL_NAI:
                worst = max(worst, abs(gdtLive(B, d, np.array(iv, float))[0] - float(r[d])))
        ok = worst <= LIVE_TABLE_TOL
        print(f"   largest difference {worst:.2e} over {len(rows)} windows x 12 detectors -> "
              f"{'AGREE' if ok else 'DISAGREE'} (tolerance {LIVE_TABLE_TOL:g})")
        rep["items"]["live"] = dict(ok=ok, worst=worst)

    old = os.path.join(regDir, "summary.json")
    prev = json.load(open(old))["items"] if os.path.exists(old) else {}
    prev.update(rep["items"])
    rep["items"] = prev
    rep["passed"] = bool(rep["items"]) and all(v["ok"] for v in rep["items"].values())   # nothing checked = not passed
    saveJson(old, rep)
    print(f"\nREGRESSION: {'PASSED' if rep['passed'] else 'FAILED'} ({len(rep['items'])} items: "
          + ", ".join(f"{k} {'ok' if v['ok'] else 'FAIL'}" for k, v in rep["items"].items()) + f")\nSaved {old}")
    writeBalaRows(S, rep)


def writeBalaRows(S, rep):
    rows = [(k, v) for k, v in rep["items"].items() if "bala" in v]
    if not rows:
        return
    cols = ["row", "interval_s", "detectors", "bin_width_s", "median_ms", "err_lo_ms", "err_hi_ms", "valid_pct",
            "limit_pct", "failed_pct", "snr_physical", "snr_bala", "class", "bala_median_ms", "bala_plus_ms",
            "bala_minus_ms", "bala_success_pct", "bala_snr", "gap_sigma"]
    lines = [",".join(cols)]
    for k, v in rows:
        n, b = v["new"], v["bala"]
        err = np.hypot(n["plus_err_ms"], b["minus"]) if n["median_ms"] < b["median"] else np.hypot(n["minus_err_ms"], b["plus"])
        sn = v["snr"] or {}
        lines.append(",".join([k, f"{v['interval'][0]:g}:{v['interval'][1]:g}", " ".join(v["dets"]), f"{v['bw']:g}",
                               f"{n['median_ms']:.4f}", f"{n['minus_err_ms']:.4f}", f"{n['plus_err_ms']:.4f}",
                               f"{n['validPercent']:.1f}", f"{n['limitPercent']:.1f}", f"{n['failedPercent']:.1f}",
                               fmt(sn.get("physical"), ".1f"), fmt(sn.get("bala"), ".1f"), sn.get("cls", ""),
                               f"{b['median']:g}", f"{b['plus']:g}", f"{b['minus']:g}", f"{b['success']:g}",
                               f"{b['snr']:g}", f"{(n['median_ms'] - b['median']) / err:+.2f}"]))
    out = os.path.join(RESULTS, "windows", S["burst"], "csv", "mvt_bala_rows.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"Saved {out}")


def requireRegression(S):
    p = os.path.join(RESULTS, "windows", S["burst"], "regression", "summary.json")
    if not os.path.exists(p):
        sys.exit(f"STOP: run the regression test first (--part regression); {p} not found")
    r = json.load(open(p))
    if not r["passed"] or not r["items"]:
        sys.exit("STOP: the regression test FAILED; nothing new is measured until the difference is found")
    print(f"regression test passed ({len(r['items'])} items: {', '.join(r['items'])})")


# ================================================================ main
def main():
    t0 = time.time()
    S = loadSettings()
    W, wPath, wMd5 = readWindows(S)
    print(f"{S['burst']}: settings {S['_path']} (md5 {S['_md5'][:8]}), windows {wPath} (md5 {wMd5[:8]}, {len(W)} windows)")
    if args.part == "write":
        writeCsv(S, W)
        return
    band = ENERGY_MAIN if args.band == "8-900" else tuple(S["dichiaraBand"] or sys.exit("no 8-1000 keV run for this burst"))
    if args.windows:
        labels = args.windows
    elif args.tier:
        labels = S["tiers"][args.tier]
    elif args.band == "8-1000":
        labels = list(S["dichiara"])
    else:
        labels = []
    for lab in labels:
        if lab not in W:
            sys.exit(f"STOP: {lab} is not in {wPath}")
    if args.part in ("windows", "check") and not labels:
        sys.exit("give --tier, --windows, or --band 8-1000")
    if args.part == "check":
        if band != ENERGY_MAIN:
            sys.exit("STOP: the made-to-measure check (step 12) is for the 8-900 keV values only")
        todo = []
        for lab in labels:
            p = os.path.join(RESULTS, "windows", S["burst"], bandTag(band), lab, "record.json")
            if not os.path.exists(p):
                sys.exit(f"STOP: {lab} has not been measured yet (no {p}); run the window first")
            r = json.load(open(p))
            if r["status"] != "measured":
                print(f"{lab}: {r['status']}, no check"); continue
            if "no-valid" in r["flags"] or "no-plateau" in r["flags"]:
                print(f"{lab}: no valid MVT or no plateau (an upper limit by the decision rule), no check"); continue
            cls = r.get("class", {}).get("cls")
            if cls in ("UL", "LUL") and not args.windows:
                print(f"{lab}: class {cls} (an upper limit by the decision rule), no check"); continue
            todo.append((lab, cls in ("UL", "LUL")))
        if not todo:
            sys.exit("nothing to check")
        labels = [lab for lab, _ in todo]
    if args.part in ("windows", "check"):
        requireRegression(S)
    checkFiles(S)
    commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
    print(f"MVTTools code in use: {commit}")
    sys.path.insert(1, BALA_CODE)
    import clasify_mvt_value as validation
    validation.MODEL_FILE = os.path.join(BALA_CODE, "mvt_snr_fit_model.npz")
    model = validation.load_model()
    if model is None:
        sys.exit(f"STOP: could not load the validation curve {validation.MODEL_FILE}")
    ctx = dict(haar=loadHaar(), commit=commit, scriptMd5=md5(os.path.abspath(__file__)), windowsMd5=wMd5,
               model=model, validation=validation)
    B = loadBurst(S, [band])
    if args.part == "regression":
        regression(ctx, S, B, W)
    elif args.part == "check":
        for lab, info in todo:
            madeToMeasure(ctx, S, B, lab, W[lab], band, info)
        args.post = True                     # rebuild each record from saved results, now with check.json
        for lab, _ in todo:
            r = processWindow(ctx, S, B, lab, W[lab], band)
            d = r["decision"]
            print(f"record rebuilt: {lab} {d['kind']} {d['mvt_ms']:.3f} ms, trusted {d['trusted']}, flags {r['flags']}")
        print("then: python run_window_mvt.py --burst " + S["burst"] + " --part write")
    else:
        done = []
        for lab in labels:
            try:
                done.append(processWindow(ctx, S, B, lab, W[lab], band))
            except MissingResult as e:
                print(f"{lab}: no saved result {e} (run without --post)")
        print(f"\n===== SUMMARY {S['burst']} {bandTag(band)} =====")
        print(f"{'label':10s}{'decision':>13s}{'MVT ms':>10s}{'-err':>8s}{'+err':>8s}{'valid%':>7s}{'class':>8s}"
              f"{'SNR':>7s}  detectors / flags")
        for r in done:
            if r["status"] != "measured":
                print(f"{r['label']:10s}{r['status']:>13s}")
                continue
            d = r["decision"]
            print(f"{r['label']:10s}{d['kind']:>13s}{d['mvt_ms']:>10.3f}{fmt(d.get('err_lo_ms'), '.3f'):>8s}"
                  f"{fmt(d.get('err_hi_ms'), '.3f'):>8s}{r['mc'][f'{BIN_WIDTHS[-1]:g}']['validPercent']:>7.0f}"
                  f"{r.get('class', {}).get('cls', 'n/a'):>8s}{fmt(r.get('snr', {}).get('physical'), '.1f'):>7s}  "
                  f"{' '.join(r['detectors'])} / {';'.join(r['flags']) or '-'}")
        print("then: python run_window_mvt.py --burst " + S["burst"] + " --part write")
    print(f"Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
