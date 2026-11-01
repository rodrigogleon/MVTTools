"""
Analyses of the saved injection-recovery results (sprint step 6, follow-up A). No simulations are rerun.

A1. Distribution of the 300 MVT values of every setting
    For each pulse, amplitude (including the bright run) and bin width: number of valid / upper-limit / failed
    realisations, number of distinct valid values, and the share of valid values that are exact repeats
    (the same value, to 1e-9 relative, returned by two or more independent realisations).
    Why: haarPowerMod finds the break point with a bounded minimiser (minimize_scalar) on a residual function that
    jumps whenever the break crosses one of its discrete wavelet timescales, so the returned MVT may sit on a
    limited set of values instead of spreading continuously. Independent realisations of a continuous quantity
    almost never give identical values, so exact repeats measure this directly.
    The same check is made on the saved real-data Monte Carlo runs (GRB 230307A and GRB 211211A rows and the
    merged runs), because they feed P1 and the error bars quoted for the bursts.
    Histograms: one figure per pulse (rows: bin widths; columns: amplitudes 3, 30, 300 and the bright run).

A2. Upper limits against failures
    The valid / upper-limit / failed split of every setting, listing the settings with fewer than 90% valid
    realisations at amplitude 100 or more (bright pulses that should not lose realisations).

A3. Criterion 1 again, compared with the bright run at the same bin width (POST HOC: added after seeing the
    results; the pre-registered criterion 1 result, 22/27, is unchanged)
    For every test setting: is the bright-run median at the SAME bin width inside the setting's 16-84% range?
    Reported for the 27 robust settings that are not bin-limited (the settings of criterion 1) and for all
    settings with amplitude 100 or 300.

Decision rules (fixed before running)
  R1. A setting with at least 30 valid realisations is "clumped" if more than 10% of its valid values are exact
      repeats (a continuous output gives about 0%). Over those settings: clumping is a general property of the
      method if more than half are clumped; absent if fewer than 5% are; partial otherwise.
  R2. No pass/fail: the split is reported; the cause of the losses (limit or failure) decides what to look at next.
  R3. Post hoc criterion 1 passes in a setting if the same-bin-width bright median lies inside its 16-84% range.

Output: results/injection_recovery/analysis_A/summary.json and results/plots/IR_hist_<pulse>.png/.pdf

Usage (from the scripts folder, next to run_injection_recovery.py)
  python analyse_injection_saved.py
"""
import os, sys, json, glob
import numpy as np
import run_injection_recovery as ir

REL_TOL = 1e-9
MIN_VALID = 30
CLUMP_SHARE = 0.10
HIST_AMPS = [3.0, 30.0, 300.0, ir.BRIGHT_AMPLITUDE]
REAL_ROOTS = [os.path.join(ir.RESULTS, "bala_rows_8-900keV_rs_haarfix"),
              os.path.join(ir.RESULTS, "bala_rows_211211A"),
              os.path.join(ir.RESULTS, "merged_mc")]
ANALYSIS_DIR = os.path.join(ir.OUT_DIR, "analysis_A")


def split(mvt, unc):
    """Valid / upper-limit / failed masks, exactly as summarise() in run_injection_recovery.py."""
    mvt, unc = np.asarray(mvt, float), np.asarray(unc, float)
    failed = ~np.isfinite(mvt) | (mvt <= 0.0)
    limit = ~failed & ~(unc > 0.0)
    return ~failed & ~limit, limit, failed


def repeats(values):
    """Distinct values, share of values that are exact repeats, and the largest group (value, share)."""
    v = np.sort(np.asarray(values, float))
    if v.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    newGroup = np.ones(v.size, bool)
    newGroup[1:] = np.abs(np.diff(v)) > REL_TOL * np.abs(v[1:])
    groupId = np.cumsum(newGroup) - 1
    sizes = np.bincount(groupId)
    inRepeat = sizes[groupId] >= 2
    top = int(np.argmax(sizes))
    return int(sizes.size), float(inRepeat.mean()), float(v[newGroup][top]), float(sizes[top] / v.size)


def loadSettings():
    recs = {}
    for key in ir.PULSES:
        for amp in ir.AMPLITUDES + [ir.BRIGHT_AMPLITUDE]:
            path = os.path.join(ir.OUT_DIR, "settings", f"{key}_{ir.ampTag(amp)}.json")
            if os.path.exists(path):
                recs[(key, amp)] = json.load(open(path))
            else:
                print(f"WARNING: missing {path}")
    return recs


def truthFrom(recs):
    """True MVT and BW* per pulse, with the same rule as post() in run_injection_recovery.py."""
    truth = {}
    for key in ir.PULSES:
        b = recs.get((key, ir.BRIGHT_AMPLITUDE))
        if b is None:
            continue
        meds = {pb["binWidth_s"]: pb["median"] for pb in b["bins"]}
        anchor = meds[min(ir.BIN_WIDTHS)]
        if not np.isfinite(anchor):
            continue
        ok = [bw for bw in ir.BIN_WIDTHS if np.isfinite(meds[bw]) and abs(meds[bw] - anchor) / anchor <= ir.BIAS_GATE]
        bwStar = max(ok) if ok else min(ir.BIN_WIDTHS)
        truth[key] = dict(tTrue=float(np.mean([meds[bw] for bw in ir.BIN_WIDTHS if bw <= bwStar])), bwStar=bwStar,
                          bright=meds)
    return truth


def a1a2(recs):
    rows = []
    print("\n===== A1/A2. VALID / LIMIT / FAILED AND EXACT REPEATS (all settings, injection) =====")
    print(f"{'pulse':8s}{'amp':>6s}{'BW ms':>7s}{'valid':>6s}{'limit':>6s}{'fail':>5s}{'distinct':>9s}"
          f"{'repeat%':>9s}{'top value ms':>14s}{'top%':>6s}{'lim distinct':>13s}")
    for (key, amp), r in recs.items():
        for pb in r["bins"]:
            valid, limit, failed = split(pb["mvt_ms"], pb["unc_ms"])
            v = np.asarray(pb["mvt_ms"], float)
            nd, rep, topV, topS = repeats(v[valid])
            ndL = repeats(v[limit])[0]
            row = dict(pulse=key, amplitude=amp, binWidth_ms=pb["binWidth_s"] * 1e3, nTotal=int(v.size), nValid=int(valid.sum()),
                       nLimit=int(limit.sum()), nFailed=int(failed.sum()), distinctValid=nd, repeatShare=rep,
                       topValue_ms=topV, topShare=topS, distinctLimit=ndL)
            rows.append(row)
            print(f"{key:8s}{amp:>6g}{row['binWidth_ms']:>7g}{row['nValid']:>6d}{row['nLimit']:>6d}{row['nFailed']:>5d}"
                  f"{nd:>9d}{(100 * rep if np.isfinite(rep) else float('nan')):>9.1f}{topV:>14.5g}"
                  f"{(100 * topS if np.isfinite(topS) else float('nan')):>6.1f}{ndL:>13d}")
    judged = [x for x in rows if x["nValid"] >= MIN_VALID]
    clumped = [x for x in judged if x["repeatShare"] > CLUMP_SHARE]
    frac = len(clumped) / len(judged) if judged else float("nan")
    verdict = ("not judged" if not judged else
               "general" if frac > 0.5 else ("absent" if frac < 0.05 else "partial"))
    print(f"\nR1: settings with >= {MIN_VALID} valid values: {len(judged)}; clumped (> {100 * CLUMP_SHARE:g}% exact "
          f"repeats): {len(clumped)} ({100 * frac:.0f}%) -> clumping {verdict.upper()}")
    for bw in ir.BIN_WIDTHS:
        sub = [x for x in judged if abs(x["binWidth_ms"] - bw * 1e3) < 1e-12]
        if sub:
            print(f"    bin width {bw * 1e3:g} ms: {sum(x['repeatShare'] > CLUMP_SHARE for x in sub)}/{len(sub)} clumped, "
                  f"median repeat share {100 * np.median([x['repeatShare'] for x in sub]):.1f}%")
    print(f"\nA2: bright settings (amplitude >= 100, including the bright run) with fewer than 90% valid:")
    print(f"{'pulse':8s}{'amp':>6s}{'BW ms':>7s}{'valid':>6s}{'limit':>6s}{'fail':>5s}")
    for x in rows:
        if x["amplitude"] >= 100 and x["nValid"] < 0.9 * x["nTotal"]:
            print(f"{x['pulse']:8s}{x['amplitude']:>6g}{x['binWidth_ms']:>7g}{x['nValid']:>6d}{x['nLimit']:>6d}{x['nFailed']:>5d}")
    nl = sum(x["nLimit"] for x in rows if x["amplitude"] >= 100)
    nf = sum(x["nFailed"] for x in rows if x["amplitude"] >= 100)
    print(f"    totals over all settings with amplitude >= 100: upper limits {nl}, failed {nf}")
    return rows, dict(judged=len(judged), clumped=len(clumped), fraction=frac, verdict=verdict)


def a1Real():
    print("\n===== A1 (real data). EXACT REPEATS IN THE SAVED MONTE CARLO RUNS OF THE BURSTS =====")
    print(f"{'run folder':55s}{'valid':>6s}{'limit':>6s}{'fail':>5s}{'distinct':>9s}{'repeat%':>9s}{'top value ms':>14s}{'top%':>6s}")
    out = []
    for root in REAL_ROOTS:
        for f in sorted(glob.glob(os.path.join(root, "*", "mvt_values_seconds.npy"))):
            d = os.path.dirname(f)
            uf = os.path.join(d, "mvt_uncertainties_seconds.npy")
            if not os.path.exists(uf):
                continue
            mvt, unc = np.load(f) * 1e3, np.load(uf) * 1e3
            valid, limit, failed = split(mvt, unc)
            nd, rep, topV, topS = repeats(mvt[valid])
            name = os.path.relpath(d, ir.RESULTS)
            out.append(dict(run=name, nValid=int(valid.sum()), nLimit=int(limit.sum()), nFailed=int(failed.sum()),
                            distinctValid=nd, repeatShare=rep, topValue_ms=topV, topShare=topS))
            print(f"{name:55s}{int(valid.sum()):>6d}{int(limit.sum()):>6d}{int(failed.sum()):>5d}{nd:>9d}"
                  f"{(100 * rep if np.isfinite(rep) else float('nan')):>9.1f}{topV:>14.5g}"
                  f"{(100 * topS if np.isfinite(topS) else float('nan')):>6.1f}")
    if not out:
        print("    no saved real-data runs found under", REAL_ROOTS)
    return out


def a3(recs, truth):
    classify, _, _ = ir.loadValidationCurve()
    print("\n===== A3. CRITERION 1 AGAINST THE BRIGHT RUN AT THE SAME BIN WIDTH (POST HOC) =====")
    print(f"{'pulse':8s}{'amp':>6s}{'BW ms':>7s}{'binlim':>7s}{'class':>8s}{'median':>9s}{'16-84%':>17s}"
          f"{'bright same BW':>15s}{'inside':>7s}{'diff':>8s}{'true (avg)':>11s}{'inside':>7s}")
    rows = []
    for key, t in truth.items():
        for amp in ir.AMPLITUDES:
            r = recs.get((key, amp))
            if r is None:
                continue
            for pb in r["bins"]:
                if not np.isfinite(pb["median"]):
                    continue
                cls, _ = classify(pb["median"], pb["snrMvt"])
                bl = pb["binWidth_s"] > t["bwStar"]
                ref = t["bright"][pb["binWidth_s"]]
                insideSame = bool(np.isfinite(ref) and pb["p16"] <= ref <= pb["p84"])
                insideAvg = bool(pb["p16"] <= t["tTrue"] <= pb["p84"])
                row = dict(pulse=key, amplitude=amp, binWidth_ms=pb["binWidth_s"] * 1e3, binLimited=bl, cls=cls,
                           median_ms=pb["median"], p16_ms=pb["p16"], p84_ms=pb["p84"], brightSameBw_ms=ref,
                           insideSameBw=insideSame, relDiffSameBw=(pb["median"] - ref) / ref if np.isfinite(ref) else float("nan"),
                           tTrue_ms=t["tTrue"], insideTrue=insideAvg)
                rows.append(row)
                if (cls == "robust" and not bl) or amp >= 100:
                    rng = f"{pb['p16']:.3g}-{pb['p84']:.3g}"
                    print(f"{key:8s}{amp:>6g}{row['binWidth_ms']:>7g}{('yes' if bl else ''):>7s}{cls:>8s}{pb['median']:>9.3g}"
                          f"{rng:>17s}"
                          f"{ref:>15.4g}{('yes' if insideSame else 'NO'):>7s}{100 * row['relDiffSameBw']:>+7.0f}%"
                          f"{t['tTrue']:>11.4g}{('yes' if insideAvg else 'NO'):>7s}")
    c1 = [x for x in rows if x["cls"] == "robust" and not x["binLimited"]]
    nSame = sum(x["insideSameBw"] for x in c1)
    nAvg = sum(x["insideTrue"] for x in c1)
    print(f"\nR3 (post hoc): robust, not bin-limited settings containing the bright median at the same bin width: "
          f"{nSame}/{len(c1)}   (pre-registered criterion 1, against the averaged true MVT: {nAvg}/{len(c1)})")
    for x in c1:
        if not x["insideSameBw"]:
            print(f"    not inside: {x['pulse']} amp {x['amplitude']:g} BW {x['binWidth_ms']:g} ms: "
                  f"{x['median_ms']:.3g} ({x['p16_ms']:.3g}-{x['p84_ms']:.3g}), bright same BW {x['brightSameBw_ms']:.4g}")
    return rows, dict(insideSameBw=nSame, insideTrue=nAvg, n=len(c1))


def histograms(recs, truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(ir.PLOT_DIR, exist_ok=True)
    for key in ir.PULSES:
        fig, axes = plt.subplots(len(ir.BIN_WIDTHS), len(HIST_AMPS), figsize=(4 * len(HIST_AMPS), 3 * len(ir.BIN_WIDTHS)))
        for j, amp in enumerate(HIST_AMPS):
            r = recs.get((key, amp))
            for i, bw in enumerate(ir.BIN_WIDTHS):
                ax = axes[i, j]
                pb = next((p for p in r["bins"] if abs(p["binWidth_s"] - bw) < 1e-15), None) if r else None
                if pb is None:
                    ax.set_axis_off()
                    continue
                valid, limit, failed = split(pb["mvt_ms"], pb["unc_ms"])
                v = np.asarray(pb["mvt_ms"], float)[valid]
                if v.size >= 2:
                    lo = min(v.min(), truth[key]["tTrue"]) if key in truth else v.min()
                    hi = max(v.max(), truth[key]["tTrue"]) if key in truth else v.max()
                    ax.hist(v, bins=np.linspace(lo, hi, 100), color="C0")
                    ax.axvline(np.median(v), color="k", lw=1.2)
                    for q in np.percentile(v, [16, 84]):
                        ax.axvline(q, color="k", lw=0.8, ls="--")
                if key in truth:
                    ax.axvline(truth[key]["tTrue"], color="C3", lw=1.2)
                nd, rep, _, topS = repeats(v)
                ax.set_title(f"{key}, amp {amp:g}, {bw * 1e3:g} ms", fontsize=9)
                ax.text(0.02, 0.97, f"valid {int(valid.sum())}  limit {int(limit.sum())}  failed {int(failed.sum())}\n"
                        f"distinct {nd}  repeats {100 * rep:.0f}%  top {100 * topS:.0f}%" if v.size else
                        f"valid {int(valid.sum())}  limit {int(limit.sum())}  failed {int(failed.sum())}",
                        transform=ax.transAxes, va="top", fontsize=7)
                if i == len(ir.BIN_WIDTHS) - 1:
                    ax.set_xlabel("MVT of each valid realisation (ms)")
        axes[0, 0].plot([], [], color="k", lw=1.2, label="median")
        axes[0, 0].plot([], [], color="k", lw=0.8, ls="--", label="16-84%")
        axes[0, 0].plot([], [], color="C3", lw=1.2, label="true MVT")
        axes[0, 0].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(ir.PLOT_DIR, f"IR_hist_{key}.{ext}"), dpi=130)
        plt.close(fig)
    print(f"\nSaved histograms: {ir.PLOT_DIR}/IR_hist_<pulse>.png/.pdf")


if __name__ == "__main__":
    recs = loadSettings()
    truth = truthFrom(recs)
    rowsA12, r1 = a1a2(recs)
    rowsReal = a1Real()
    rowsA3, r3 = a3(recs, truth)
    histograms(recs, truth)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(os.path.join(ANALYSIS_DIR, "summary.json"), "w") as f:
        json.dump(dict(R1=r1, R3=r3, settings=rowsA12, realRuns=rowsReal, criterion1SameBw=rowsA3), f, indent=1,
                  default=float)
    print(f"Saved {ANALYSIS_DIR}/summary.json")
