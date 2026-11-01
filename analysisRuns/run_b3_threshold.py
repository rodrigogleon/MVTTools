"""
B3 + B5: from what SNR_MVT can the MVT be trusted? Fine brightness grid, standard and precursor-like conditions.

Question (Dr Tsang): the lowest SNR (or another measure) above which the MVT given by the pipeline can be trusted, as
a minimum for faint precursors. The injection run showed the answer depends on the MVT, and its brightness steps
(x3) only located each pulse's threshold to within a factor of about 3.

Pulses: G3, G10, G30, N3_10, N10_30, N30_100 (MVT about 2 to 21 ms). G1 is left out (MVT <= 0.72 ms, below what
real GBM data can measure).
Bin width: the smallest real-data bin width with a nearly unbiased bright answer (B1): 0.003 ms for G3 and N3_10,
0.01 ms for the others.
Brightness grid: 16 amplitudes per pulse, A = A0 x 1.2^k, k = -9 ... +6 (0.19 A0 to 3.0 A0), where A0 is the lowest
amplitude that worked in the injection run (G3 100, N3_10 100, G10 30, N10_30 30, G30 10, N30_100 10). The grid
reaches the injection run's next amplitude (3 A0) because the definition of "works" below is stricter.
Conditions
  standard (B3): windows and background (1000 counts/s) as in run_injection_recovery.py.
  precursor (B5): the window of the GRB 230307A precursor, 0.384 s (Gaussians centred, -0.192 to 0.192 s; Norris
     pulses starting at 0, -0.064 to 0.32 s as the precursor interval; N30_100 is cut at 0.32 s, where its rate is
     about 4% of the peak), background 5000 counts/s; amplitudes
     divided by sqrt(5) so that SNR_MVT covers the same range (SNR ~ A x sqrt(background x MVT)).
Main Monte Carlo scheme (300 independent simulated observations per setting), 4 rungs per doubling, new seeds.

Reference value R: the bright-run median at the same bin width and in the same condition (standard: the saved
amplitude-3000 run; precursor: a new run at 3000 / sqrt(5)). This separates the effect of faintness (measured here)
from the effect of the bin width (measured in B1). The B1 truth is reported alongside.

Definitions (fixed before running)
  A setting works if: at least 50% valid realisations, R inside its 16-84% range, and |median / R - 1| <= 20%.
  Threshold: the faintest amplitude of the grid from which that amplitude and every brighter one work. Reported:
     its SNR_MVT (measured as in real data, mean over the 300 observations), the SNR_MVT one step fainter (the
     threshold lies between the two), and the class from Bala et al.'s validation curve at the threshold.
  Also (Bala et al.'s rule, for comparison): the lowest SNR_MVT among settings whose 16-84% range contains R.
Decision rules
  S1 (is SNR_MVT enough?): for each pulse, the standard and precursor threshold SNR_MVT agree within one grid step
     (factor 1.2, i.e. their ratio is between 1/1.2 and 1.2): SNR_MVT is a sufficient measure for that pulse.
     If it holds for all pulses: one trust curve for both conditions; otherwise one per condition.
  S2: each threshold SNR_MVT compared with Bala et al.'s 95% band at R: inside, left or right.

Output: results/injection_recovery/b3_threshold/ (settings/, summary.json); plots results/plots/B3_recovery.png
(median / R against SNR_MVT per pulse) and B3_trust_curve.png (threshold SNR_MVT against MVT, with Bala et al.'s
band). Resumable.

Usage (from the scripts folder, next to run_injection_recovery.py)
  python run_b3_threshold.py               simulations, then the analysis
  python run_b3_threshold.py --post        analysis and plots only
"""
import os, sys, json, time, argparse, logging, io, contextlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import run_injection_recovery as ir

PULSES = ["G3", "G10", "G30", "N3_10", "N10_30", "N30_100"]
BIN = {"G3": 3e-6, "N3_10": 3e-6, "G10": 1e-5, "G30": 1e-5, "N10_30": 1e-5, "N30_100": 1e-5}
A0 = {"G3": 100.0, "N3_10": 100.0, "G10": 30.0, "N10_30": 30.0, "G30": 10.0, "N30_100": 10.0}
STEP = 1.2
KS = list(range(-9, 7))
COND = {"standard": dict(bkg=ir.BKG_RATE, scale=1.0), "precursor": dict(bkg=5000.0, scale=1.0 / np.sqrt(5.0))}
PRE_GAUSS = (-0.192, 0.192)
PRE_NORRIS = (-0.064, 0.32)
MIN_VALID = 0.5
TOL = 0.20
OUT = os.path.join(ir.OUT_DIR, "b3_threshold")
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


def windowFor(key, cond):
    if cond == "standard":
        return ir.window(ir.PULSES[key])
    return PRE_GAUSS if ir.PULSES[key]["shape"] == "gaussian" else PRE_NORRIS


def simulate(key, amp, cond, seed):
    """One observation: background at the condition's rate plus the pulse (as ir.simulate, any window)."""
    p = ir.PULSES[key]
    bkg = COND[cond]["bkg"]
    tstart, tstop = windowFor(key, cond)
    rng = np.random.default_rng(seed)
    nb = rng.poisson(bkg * (tstop - tstart))
    tb = rng.uniform(tstart, tstop, nb)
    if p["shape"] == "gaussian":
        n = rng.poisson(amp * bkg * p["sigma"] * np.sqrt(2 * np.pi))
        ts = rng.normal(0.0, p["sigma"], n)
    else:
        grid, cum = ir.norrisCumulative(key)
        n = rng.poisson(amp * bkg / ir.BKG_RATE * cum[-1])
        ts = np.interp(rng.uniform(0.0, cum[-1], n), cum, grid)
    ts = ts[(ts >= tstart) & (ts < tstop)]
    return np.sort(np.concatenate([tb, ts])), int(nb)


def _mvtTask(args):
    global hm
    if hm is None:
        hm = loadHaar()
    key, amp, cond, seed = args
    tstart, tstop = windowFor(key, cond)
    times, nb = simulate(key, amp, cond, seed)
    c = ir.lightCurve(times, tstart, tstop, BIN[key])
    try:
        r = hm.haarPowerMod(c, np.sqrt(c), minimumBinSizeSeconds=BIN[key])
        return float(r[2]), float(r[3])
    except Exception:
        return float("nan"), float("nan")


def _snrTask(args):
    key, amp, cond, seed, medianS = args
    tstart, tstop = windowFor(key, cond)
    times, nb = simulate(key, amp, cond, seed)
    return ir.snrAt(times, nb, tstart, tstop, medianS) if medianS else float("nan")


def amps(key, cond):
    return [A0[key] * STEP ** k * COND[cond]["scale"] for k in KS]


def brightAmp(cond):
    return ir.BRIGHT_AMPLITUDE * COND[cond]["scale"]


def path(key, cond, k):
    return os.path.join(OUT, "settings", f"{cond}_{key}_{'bright' if k is None else f'k{k:+d}'}.json")


def runOne(pool, key, cond, k):
    p = path(key, cond, k)
    if os.path.exists(p):
        print(f"{cond:9s} {key:8s} {'bright' if k is None else f'k={k:+d}':>7s}: already done, skipped")
        return
    t0 = time.time()
    amp = brightAmp(cond) if k is None else A0[key] * STEP ** k * COND[cond]["scale"]
    ci = list(COND).index(cond)
    kk = 99 if k is None else k + 50
    seeds = np.random.SeedSequence([ir.SEED, 300 + ci, list(ir.PULSES).index(key), kk]).spawn(ir.N_REAL)
    res = list(pool.map(_mvtTask, [(key, amp, cond, s) for s in seeds], chunksize=4))
    mvt, unc = [r[0] for r in res], [r[1] for r in res]
    s = ir.summarise(mvt, unc)
    med = s["median"] / 1e3 if np.isfinite(s["median"]) else None
    snrs = np.array(list(pool.map(_snrTask, [(key, amp, cond, sd, med) for sd in seeds], chunksize=4)))
    good = np.isfinite(snrs) & (snrs > 0)
    s["snrMvt"] = float(snrs[good].mean()) if good.any() else float("nan")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(dict(pulse=key, condition=cond, k=k, amplitude=amp, background=COND[cond]["bkg"],
                       window_s=list(windowFor(key, cond)), binWidth_s=BIN[key], nRealisations=ir.N_REAL,
                       mvt_ms=[v * 1e3 for v in mvt], unc_ms=[v * 1e3 for v in unc], **s), f, default=float)
    print(f"{cond:9s} {key:8s} {'bright' if k is None else f'k={k:+d}':>7s} amp {amp:8.3g}: median {s['median']:.4g} ms, "
          f"{s['successPct']:.0f}% valid, SNR {s['snrMvt']:.3g}  [{(time.time() - t0) / 60:.1f} min]", flush=True)


def run(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cond in COND:
            for key in PULSES:
                if cond == "precursor":
                    runOne(pool, key, cond, None)
                for k in KS:
                    runOne(pool, key, cond, k)


# ---------------------------------------------------------------- analysis
def reference(key, cond):
    if cond == "standard":
        saved = json.load(open(os.path.join(ir.OUT_DIR, "settings", f"{key}_{ir.ampTag(ir.BRIGHT_AMPLITUDE)}.json")))
        pb = next(p for p in saved["bins"] if abs(p["binWidth_s"] - BIN[key]) < 1e-15)
        return pb["median"], pb["successPct"]
    p = path(key, cond, None)
    if not os.path.exists(p):
        return float("nan"), float("nan")
    r = json.load(open(p))
    return r["median"], r["successPct"]


def b1Truth(key):
    f = os.path.join(ir.OUT_DIR, "b1_truth", "summary.json")
    if os.path.exists(f):
        t = json.load(open(f)).get("truth", {}).get(key)
        if t:
            return t["new"], t["settled"]
    return float("nan"), None


def post():
    classify, band, _ = ir.loadValidationCurve()
    summary = dict(pulses={})
    print("\n===== B3 + B5. RECOVERY AGAINST BRIGHTNESS =====")
    for key in PULSES:
        b1, b1s = b1Truth(key)
        for cond in COND:
            R, Rvalid = reference(key, cond)
            rows = []
            for k in KS:
                p = path(key, cond, k)
                if not os.path.exists(p):
                    continue
                r = json.load(open(p))
                ok = (r["successPct"] >= 100 * MIN_VALID and np.isfinite(r["median"]) and np.isfinite(R)
                      and r["p16"] <= R <= r["p84"] and abs(r["median"] / R - 1) <= TOL)
                with contextlib.redirect_stdout(io.StringIO()):
                    cls, _ = classify(r["median"], r["snrMvt"])
                rows.append(dict(k=k, amp=r["amplitude"], median=r["median"], p16=r["p16"], p84=r["p84"],
                                 valid=r["successPct"], snr=r["snrMvt"], cls=cls, works=bool(ok),
                                 ratio=r["median"] / R if np.isfinite(R) else float("nan"),
                                 inRange=bool(np.isfinite(R) and np.isfinite(r["median"]) and r["p16"] <= R <= r["p84"])))
            rows.sort(key=lambda x: x["k"])
            print(f"\n{key} ({cond}): bin {BIN[key] * 1e3:g} ms, window {windowFor(key, cond)[1] - windowFor(key, cond)[0]:.3g} s, "
                  f"background {COND[cond]['bkg']:g} c/s; reference R (bright, same bin) = {R:.4g} ms "
                  f"({Rvalid:.0f}% valid); B1 truth {b1:.4g} ms ({'settled' if b1s else 'unsettled' if b1s is False else 'n/a'})")
            print(f"{'k':>4s}{'amp':>9s}{'median':>9s}{'16-84%':>16s}{'valid%':>7s}{'SNR':>8s}{'median/R':>9s}"
                  f"{'R in range':>11s}{'class':>8s}{'works':>7s}")
            for x in rows:
                rr = f"{x['p16']:.3g}-{x['p84']:.3g}"
                print(f"{x['k']:>+4d}{x['amp']:>9.3g}{x['median']:>9.4g}{rr:>16s}{x['valid']:>7.0f}{x['snr']:>8.3g}"
                      f"{x['ratio']:>9.3f}{('yes' if x['inRange'] else 'no'):>11s}{x['cls']:>8s}{('YES' if x['works'] else ''):>7s}")
            # threshold: faintest amplitude from which it and all brighter ones work
            thr = None
            for i in range(len(rows)):
                if all(x["works"] for x in rows[i:]):
                    thr = i
                    break
            bala = [x["snr"] for x in rows if x["inRange"] and np.isfinite(x["snr"])]
            res = dict(reference=R, b1Truth=b1, rows=rows, balaRuleMinSnr=min(bala) if bala else float("nan"))
            if thr is not None and len(rows) == len(KS):
                t, below = rows[thr], (rows[thr - 1] if thr > 0 else None)
                lo, mid, hi = band(R)
                where = "inside" if lo <= t["snr"] <= hi else ("left of band" if t["snr"] < lo else "right of band")
                res.update(threshold=dict(k=t["k"], amp=t["amp"], snr=t["snr"], cls=t["cls"],
                                          snrBelow=below["snr"] if below else float("nan"),
                                          atGridEdge=thr == 0, band=[lo, mid, hi], where=where))
                print(f"THRESHOLD: amplitude {t['amp']:.3g}, SNR_MVT {t['snr']:.3g} (one step fainter: "
                      f"{(below['snr'] if below else float('nan')):.3g}); class there: {t['cls']}; "
                      f"Bala et al.'s band at R: {lo:.3g}-{hi:.3g} -> {where}"
                      + ("  [WARNING: works at the faintest grid amplitude, threshold may be lower]" if thr == 0 else ""))
            else:
                print("THRESHOLD: not found in the grid" + ("" if len(rows) == len(KS) else " (settings missing)"))
            print(f"Bala et al.'s rule (lowest SNR whose 16-84% range contains R): {res['balaRuleMinSnr']:.3g}")
            summary["pulses"].setdefault(key, {})[cond] = res

    print("\n===== S1 / S2. TRUST CURVE: THRESHOLD SNR_MVT AGAINST MVT =====")
    print(f"{'pulse':8s}{'R std (ms)':>11s}{'thr SNR std':>12s}{'class':>8s}{'vs band':>14s}{'R pre (ms)':>11s}"
          f"{'thr SNR pre':>12s}{'class':>8s}{'vs band':>14s}{'ratio pre/std':>14s}{'S1':>6s}")
    allS1 = True
    for key in PULSES:
        d = summary["pulses"].get(key, {})
        s, p = d.get("standard", {}).get("threshold"), d.get("precursor", {}).get("threshold")
        ratio = p["snr"] / s["snr"] if s and p else float("nan")
        edge = bool((s and s["atGridEdge"]) or (p and p["atGridEdge"]))
        s1 = bool(np.isfinite(ratio) and not edge and 1 / STEP <= ratio <= STEP)
        allS1 &= s1
        d["S1"] = dict(ratio=ratio, agree=s1, judged=bool(np.isfinite(ratio) and not edge))
        s1txt = "n/j" if not d["S1"]["judged"] else ("yes" if s1 else "NO")
        f = lambda t, a: (f"{t[a]:.3g}" if t else "-")
        print(f"{key:8s}{d.get('standard', {}).get('reference', float('nan')):>11.4g}{f(s, 'snr'):>12s}"
              f"{(s['cls'] if s else '-'):>8s}{(s['where'] if s else '-'):>14s}"
              f"{d.get('precursor', {}).get('reference', float('nan')):>11.4g}{f(p, 'snr'):>12s}"
              f"{(p['cls'] if p else '-'):>8s}{(p['where'] if p else '-'):>14s}{ratio:>14.3f}{s1txt:>6s}")
    print("(n/j = not judged: a threshold is missing or at the faintest grid amplitude)")
    judged = [summary["pulses"].get(k, {}).get("S1", {}) for k in PULSES]
    nj = sum(1 for j in judged if not j.get("judged"))
    nNo = sum(1 for j in judged if j.get("judged") and not j.get("agree"))
    if nNo:
        verdict = f"conditions matter for {nNo} pulse(s) -> quote the precursor-condition thresholds for precursors"
    elif nj:
        verdict = f"agree where judged, but {nj} pulse(s) not judged -> not decided (extend the grid for those)"
    else:
        verdict = "SNR_MVT is a sufficient measure (one trust curve for both conditions)"
    print("S1 verdict: " + verdict)
    summary["S1all"] = allS1
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"\nSaved {OUT}/summary.json")
    plots(summary, band)


def plots(summary, band):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(ir.PLOT_DIR, exist_ok=True)
    mk = {"standard": ("o", "C0"), "precursor": ("s", "C1")}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, key in zip(axes.ravel(), PULSES):
        for cond in COND:
            d = summary["pulses"].get(key, {}).get(cond)
            if not d or not d["rows"]:
                continue
            R = d["reference"]
            x = np.array([r["snr"] for r in d["rows"]])
            y = np.array([r["ratio"] for r in d["rows"]])
            lo = y - np.array([r["p16"] / R for r in d["rows"]])
            hi = np.array([r["p84"] / R for r in d["rows"]]) - y
            m, c = mk[cond]
            ax.errorbar(x, y, yerr=[lo, hi], fmt=m, color=c, mfc="none", ms=5, capsize=2, lw=0.8, label=cond)
            w = np.array([r["works"] for r in d["rows"]])
            ax.plot(x[w], y[w], m, color=c, ms=5)
            if d.get("threshold"):
                ax.axvline(d["threshold"]["snr"], color=c, ls="--", lw=1)
        ax.axhspan(1 - TOL, 1 + TOL, color="0.9", zorder=0)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("SNR_MVT"); ax.set_ylabel("median MVT / bright reference R")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8, title="filled = works; dashed = threshold", title_fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ir.PLOT_DIR, f"B3_recovery.{ext}"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    grid = np.logspace(np.log10(0.8), np.log10(60), 200)
    b = np.array([band(g) for g in grid])
    ax.fill_betweenx(grid, b[:, 0], b[:, 2], color="0.85", label="Bala et al. 95% band")
    ax.plot(b[:, 1], grid, "k-", lw=1, label="Bala et al. median")
    for cond in COND:
        m, c = mk[cond]
        for key in PULSES:
            d = summary["pulses"].get(key, {}).get(cond)
            if not d or not d.get("threshold"):
                continue
            t = d["threshold"]
            ax.errorbar(t["snr"], d["reference"], xerr=[[t["snr"] - t["snrBelow"]], [0]] if np.isfinite(t["snrBelow"]) else None,
                        fmt=m, color=c, ms=7, capsize=3)
            ax.annotate(key, (t["snr"], d["reference"]), textcoords="offset points", xytext=(6, 4 if cond == "standard" else -10),
                        fontsize=7, color=c)
        ax.plot([], [], m, color=c, label=f"threshold, {cond} (bar: down to one step fainter)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SNR_MVT at the threshold"); ax.set_ylabel("MVT (bright reference R, ms)")
    ax.set_title("Lowest SNR_MVT from which the MVT can be trusted", fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ir.PLOT_DIR, f"B3_trust_curve.{ext}"), dpi=130)
    plt.close(fig)
    print(f"Saved plots: {ir.PLOT_DIR}/B3_recovery.png, B3_trust_curve.png (.pdf too)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
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
