"""
Detector selection for every window of a burst: Bala et al.'s score (run_window_mvt.py, selectDetectors) and a
background-subtracted ("net") score, side by side. Selection only: no Monte Carlo, nothing in run_window_mvt.py or
its results is changed. Same photons (8-900 keV) and background (cubic on Iñaki's intervals) as the pipeline.

  Bala et al.:  score = brightest bin's counts / sqrt(background counts per bin)
  net:          score = (brightest bin's counts - background counts per bin) / sqrt(background counts per bin)

For each window:
  1. Bala's set at the first selection bin (16 or 64 ms); must equal the first set in the window's record.json.
  2. The net set at the same bin, with its net score; flag faint-selection if that score is below 12.
  3. The pipeline's rounds (reselect with bins = MVT, up to 3) with the net score, using only MVTs already measured
     at 0.003 ms in the window's mc/ folder. A set with no measured MVT stops the rounds: it needs a Monte Carlo run.
Predictions P1 and P2 are checked at the end (fixed before running).

Usage (from analysisRuns/):  python check_selection_rule.py --burst GRB211211A
Output: printed table and RESULTS/selection_rule_check/<burst>.json
"""
import os, sys, json, glob, argparse, hashlib
import numpy as np

PIPELINE_MD5 = "41439b11698c16282c380626408effc8"   # run_window_mvt.py of the Tier 1-3 runs (7650ce7)
FAINT_NET_SNR = 12.0      # faint-selection flag below this net score (fixed 20 Nov, before this run)
BRIGHT_BALA_SNR = 100.0   # P1: windows with Bala's first-bin score >= 100
NEAR_TIE = 0.99           # P1: Bala's set counts as a near-tie if its net score >= 0.99 x the net set's
P2_CORE, P2_MAX_DETS, P2_MIN_CORE = ("n1", "n2", "na"), 5, 2

ap = argparse.ArgumentParser()
ap.add_argument("--burst", required=True)
ap.add_argument("--settings", help="settings file (default settings/<burst>.json)")
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "run_window_mvt.py")
m = hashlib.md5(open(PIPE, "rb").read()).hexdigest()
if m != PIPELINE_MD5:
    sys.exit(f"STOP: run_window_mvt.py md5 {m}, expected {PIPELINE_MD5}; nothing done")
sys.argv = [PIPE, "--burst", a.burst] + (["--settings", a.settings] if a.settings else [])   # read by its parser
sys.path.insert(0, HERE)
import run_window_mvt as P

BAND = P.ENERGY_MAIN


def tag(dets):
    return " ".join(sorted(dets, key=P.ALL_NAI.index))


def curvesAndBkg(B, window, binS):
    curves = {d: P.lightCurve(P.inWindow(B["band"][BAND][d], window), binS, *window) for d in P.ALL_NAI}
    bkg = {d: P.bkgRate(B, P.WINDOW_BKG, d, window, BAND) * binS for d in P.ALL_NAI}     # counts per bin
    return curves, bkg


def selectNet(B, window, binS):
    """selectDetectors (l. 305-322) with the background subtracted in both scores; everything else identical."""
    curves, bkg = curvesAndBkg(B, window, binS)
    single = {d: float((curves[d].max() - bkg[d]) / np.sqrt(bkg[d])) for d in P.ALL_NAI}
    ranked = sorted(single, key=single.get, reverse=True)
    rows, combined, bkgSum = [], np.zeros_like(curves[ranked[0]]), 0.0
    for k, det in enumerate(ranked, 1):
        combined = combined + curves[det]
        bkgSum += bkg[det]
        rows.append(dict(k=k, added=det, snr=float((combined.max() - bkgSum) / np.sqrt(bkgSum))))
    best = max(rows, key=lambda r: r["snr"])
    chosen = sorted(ranked[:best["k"]], key=P.ALL_NAI.index)
    return dict(binWidth_s=binS, single=single, ranked=ranked, cumulative=rows, optimal=chosen, optimalSnr=best["snr"])


def netScore(B, window, dets, binS):
    curves, bkg = curvesAndBkg(B, window, binS)
    c = sum(curves[d] for d in dets)
    b = sum(bkg[d] for d in dets)
    return float((c.max() - b) / np.sqrt(b))


def knownMvts(folder, window):
    """MVT medians already measured at 0.003 ms for this window (any detector set, no gap-fill tag)."""
    out = {}
    for f in glob.glob(os.path.join(folder, "mc", "*.json")):
        s = json.load(open(f))
        if (s.get("fill") is None and abs(s.get("binSize_s", 0) - P.BIN_WIDTHS[-1]) < 1e-12
                and list(s.get("band_keV", [])) == list(BAND) and np.allclose(s.get("window_s", [0, 0]), window)):
            out[tag(s["detectors"])] = s.get("median_ms")
    return out


def netRounds(B, window, firstSet, known):
    """The pipeline's round logic (processWindow, steps 4 and 7) with the net score, on measured MVTs only."""
    L = window[1] - window[0]
    cur, tried, rounds = firstSet, [], []
    for rnd in range(1, P.MAX_ROUNDS + 1):
        tried.append(cur)
        mv = known.get(tag(cur))
        if mv is None:
            return dict(final=cur, mvt_ms=None, status=f"needs Monte Carlo ({tag(cur)})", rounds=rounds)
        rounds.append([cur, mv])
        if not (mv and np.isfinite(mv) and mv > 0):
            return dict(final=cur, mvt_ms=mv, status="no valid MVT", rounds=rounds)
        if L / (mv / 1e3) < 2:
            return dict(final=cur, mvt_ms=mv, status="MVT > half window", rounds=rounds)
        nxt = selectNet(B, window, mv / 1e3)["optimal"]
        if nxt == cur:
            return dict(final=cur, mvt_ms=mv, status="converged", rounds=rounds)
        if nxt in tried or rnd == P.MAX_ROUNDS:
            best = max(tried, key=lambda t: netScore(B, window, t, mv / 1e3))
            return dict(final=best, mvt_ms=known.get(tag(best)), status="switching", rounds=rounds)
        cur = nxt
    return dict(final=cur, mvt_ms=known.get(tag(cur)), status="stopped", rounds=rounds)


def main():
    S = P.loadSettings()
    W, wPath, wMd5 = P.readWindows(S)
    print(f"{S['burst']}: settings {S['_path']} (md5 {S['_md5'][:8]}), windows {wPath} (md5 {wMd5[:8]}, "
          f"{len(W)} windows); run_window_mvt.py md5 {PIPELINE_MD5[:8]}")
    P.checkFiles(S)
    B = P.loadBurst(S, [BAND])
    rows = []
    for lab, window in W.items():
        if any(P.overlaps(*window, e["start"], e["stop"]) for e in S["excluded"]):
            print(f"{lab}: overlaps an excluded interval, skipped (as the pipeline)")
            continue
        L = window[1] - window[0]
        firstBin = P.SELECT_BIN_SHORT_S if L < P.SHORT_WINDOW_S else P.SELECT_BIN_LONG_S
        bala = P.selectDetectors(B, BAND, window, firstBin, quiet=True)
        net = selectNet(B, window, firstBin)
        r = dict(label=lab, window_s=list(window), firstBin_s=firstBin,
                 bala=dict(set=bala["optimal"], score=bala["optimalSnr"],
                           netScore=netScore(B, window, bala["optimal"], firstBin)),
                 net=dict(set=net["optimal"], netScore=net["optimalSnr"], single=net["single"],
                          cumulative=net["cumulative"]),
                 faintSelection=bool(net["optimalSnr"] < FAINT_NET_SNR))
        folder = os.path.join(P.RESULTS, "windows", S["burst"], P.bandTag(BAND), lab)
        recPath = os.path.join(folder, "record.json")
        if os.path.exists(recPath) and json.load(open(recPath)).get("status") == "measured":
            rec = json.load(open(recPath))
            known = knownMvts(folder, window)
            r["record"] = dict(firstSet=rec["selection"]["first"], usedSet=rec["detectors"],
                               mvt_ms=rec["mc"][f"{P.BIN_WIDTHS[-1]:g}"]["median_ms"],
                               decision=rec.get("decision", {}).get("kind"))
            r["selfCheck"] = "ok" if rec["selection"]["first"] == bala["optimal"] else \
                f"MISMATCH: record first set {rec['selection']['first']}"
            r["knownMvts_ms"] = known
            r["netRounds"] = netRounds(B, window, net["optimal"], known)
        else:
            r["record"], r["selfCheck"], r["netRounds"] = None, "no record", None
        rows.append(r)

    # ---- table
    print(f"\n{'label':9s}{'bin':>4s} | {'Bala set (first bin)':38s}{'score':>7s}{'net':>7s} | "
          f"{'net set (first bin)':26s}{'net':>7s}{'faint':>6s} | {'used in run':38s}{'MVT':>9s} | net rounds")
    for r in rows:
        rr, rec = r["netRounds"], r["record"]
        tail = "no record" if rr is None else (f"{tag(rr['final'])}: " + (f"{rr['mvt_ms']:.3f} ms" if rr["mvt_ms"] else "-")
                                               + f" ({rr['status']})")
        used = tag(rec["usedSet"]) if rec else "-"
        mvt = f"{rec['mvt_ms']:.3f}" if rec and rec["mvt_ms"] else "-"
        faint = "YES" if r["faintSelection"] else ""
        print(f"{r['label']:9s}{r['firstBin_s'] * 1e3:4.0f} | {tag(r['bala']['set']):38s}{r['bala']['score']:7.1f}"
              f"{r['bala']['netScore']:7.1f} | {tag(r['net']['set']):26s}{r['net']['netScore']:7.1f}"
              f"{faint:>6s} | {used:38s}{mvt:>9s} | {tail}")
    bad = [r for r in rows if r["selfCheck"].startswith("MISMATCH")]
    print(f"\nself-check (Bala's first set = record.json's): "
          f"{'ok' if not bad else 'MISMATCH in ' + ', '.join(r['label'] for r in bad)}"
          f" ({sum(r['selfCheck'] == 'ok' for r in rows)} ok, {sum(r['selfCheck'] == 'no record' for r in rows)} no record)")

    # ---- predictions (TEST2_PROTOCOL.md)
    p1 = [r for r in rows if r["bala"]["score"] >= BRIGHT_BALA_SNR]
    p1Fail = [r for r in p1 if r["net"]["set"] != r["bala"]["set"]
              and r["bala"]["netScore"] < NEAR_TIE * r["net"]["netScore"]]
    p2 = [r for r in rows if S["burst"] == "GRB211211A" and r["record"] and r["record"]["usedSet"] == P.ALL_NAI]
    p2Fail = [r for r in p2 if not (len(r["net"]["set"]) <= P2_MAX_DETS
                                    and sum(d in r["net"]["set"] for d in P2_CORE) >= P2_MIN_CORE)]
    print(f"P1 (bright windows, Bala score >= {BRIGHT_BALA_SNR:g}: same set or near-tie): {len(p1)} windows, "
          f"{'no such windows' if not p1 else 'PASS' if not p1Fail else 'FAIL: ' + ', '.join(r['label'] for r in p1Fail)}")
    if S["burst"] == "GRB211211A":
        print(f"P2 (windows that used all 12 NaI: net set <= {P2_MAX_DETS} detectors with >= {P2_MIN_CORE} of "
              f"{' '.join(P2_CORE)}): {len(p2)} windows ({', '.join(r['label'] for r in p2)}), "
              f"{'PASS' if p2 and not p2Fail else ('no such windows' if not p2 else 'FAIL: ' + ', '.join(r['label'] for r in p2Fail))}")
    need = [r["label"] for r in rows if r["netRounds"] and r["netRounds"]["mvt_ms"] is None]
    print(f"windows whose net set needs a new Monte Carlo run: {', '.join(need) or 'none'}")
    out = os.path.join(P.RESULTS, "selection_rule_check", f"{S['burst']}.json")
    P.saveJson(out, dict(burst=S["burst"], settingsMd5=S["_md5"], windowsMd5=wMd5, pipelineMd5=PIPELINE_MD5,
                         scriptMd5=hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                         faintNetSnr=FAINT_NET_SNR, rows=rows,
                         P1="no such windows" if not p1 else "PASS" if not p1Fail else "FAIL",
                         P2=(None if S["burst"] != "GRB211211A" else "no such windows" if not p2 else
                             "PASS" if not p2Fail else "FAIL")))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
