"""
Joint MVT + E_peak figure for GRB 230307A and GRB 211211A, after Dichiara et al. (2023) Fig. 2, and E_peak against MVT.
Every number is read from a file; nothing is typed in.

Inputs
  MVT (this work)     RESULTS/windows/GRB230307A/csv/mvt_230307A.csv           8-900 keV (the band of the figure)
                      RESULTS/windows/GRB211211A/csv/mvt_211211A.csv           8-900 keV
                      RESULTS/windows/GRB230307A/csv/mvt_230307A_8-1000keV.csv 8-1000 keV, drawn beside Dichiara's
                      published MVT, which are read from its dichiara_* columns
  E_peak (Iñaki)      INAKI/epeak_230307A.csv, epeak_230307A_instrument_check.csv,
                      INAKI/epeak_211211A.csv, epeak_211211A_systematics.csv   (his repo, ignacio/shared)
                      For GRB 211211A, Epeak_sys comes from the systematics file, and the Band E_peak of f2_01_10
                      (Dichiara's grey point) from the notes column, as in his 12 Nov tables.
  Window edges        INAKI/windows_230307A.csv, windows_211211A.csv; their md5 must match the copies in settings/
  Light curve         GRB 230307A TTE, n6+n8+na, 8-1000 keV, 64 ms (as Iñaki's notebook 13); skipped if not found
E_peak bars: statistical and systematic in quadrature; pale bars add the instrument term (GRB 230307A only).
MVT upper limits (class LUL or UL) are drawn as arrows at the 84th percentile.

Usage (from MVTTools/analysisRuns):
  python plot_joint_figure.py --inaki /mnt/data/rgl36/FYP/inaki_repo/ignacio/shared
"""
import os, re, sys, glob, argparse, hashlib, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--results", default="/mnt/data/rgl36/FYP/results")
parser.add_argument("--inaki", required=True, help="folder with Iñaki's shared CSVs (his repo: ignacio/shared)")
parser.add_argument("--tte", default="/mnt/data/rgl36/FYP/data/fermiGBM/GRB230307A/current")
parser.add_argument("--settings", default=os.path.join(HERE, "settings"))
parser.add_argument("--out", default=None, help="default RESULTS/joint_figure")
args = parser.parse_args()
OUT = args.out or os.path.join(args.results, "joint_figure")
os.makedirs(OUT, exist_ok=True)

BAD_TIME = (3.0, 7.0)                     # s, GRB 230307A TTE packet losses (GCN 33551), as Dichiara and Iñaki
RED, GREY = "tab:red", "0.45"
EPEAK_RED_230307A = ["pre", "m01", "t1_07_18", "m11"]      # Dichiara's four E_peak points (Iñaki, 12 Nov)
EPEAK_SHOWN_211211A = ["pre", "f2_01_10", "f2_27_66"]      # Dichiara's grey points (Iñaki, 12 Nov)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def need(path):
    if not os.path.exists(path):
        sys.exit(f"STOP: {path} not found")
    return path


# ---------------------------------------------------------------- inputs
for b in ("230307A", "211211A"):                          # both sides must use the same window files
    a, s = md5(need(os.path.join(args.inaki, f"windows_{b}.csv"))), md5(need(os.path.join(args.settings, f"windows_{b}.csv")))
    if a != s:
        sys.exit(f"STOP: windows_{b}.csv differs between Iñaki's folder ({a[:8]}) and settings/ ({s[:8]})")
    print(f"windows_{b}.csv: same file on both sides (md5 {a[:8]})")


def readMvt(burst, suffix=""):
    p = need(os.path.join(args.results, "windows", f"GRB{burst}", "csv", f"mvt_{burst}{suffix}.csv"))
    m = pd.read_csv(p).set_index("label")
    if m.index.has_duplicates or m["upper_limit"].dtype != bool:
        sys.exit(f"STOP: {p}: duplicated labels or upper_limit not true/false")
    print(f"MVT {os.path.basename(p)}: {', '.join(f'{l} {1e3 * r.mvt_s:.2f} ms' + (' (limit)' if r.upper_limit else '') for l, r in m.iterrows())}")
    return m


def readEpeak(burst):
    e = pd.read_csv(need(os.path.join(args.inaki, f"epeak_{burst}.csv"))).set_index("label")
    if burst == "230307A":
        ins = pd.read_csv(need(os.path.join(args.inaki, "epeak_230307A_instrument_check.csv"))).set_index("label")
        e["instrument_frac"] = ins["Epeak_instrument_frac"]
    else:
        sysT = pd.read_csv(need(os.path.join(args.inaki, "epeak_211211A_systematics.csv"))).set_index("label")
        e["Epeak_sys"] = sysT["Epeak_sys"]
        e["instrument_frac"] = 0.0                       # no second-instrument check for this burst (Iñaki, 12 Nov)
        band = e["notes"].astype(str).str.extract(r"Band's E_peak is ([0-9.]+) keV")[0].astype(float)
        e["Epeak_band_keV"] = band
    if e[["Epeak_keV", "Epeak_err_lo", "Epeak_err_hi", "Epeak_sys"]].isna().any().any():
        sys.exit(f"STOP: epeak_{burst}: missing E_peak, error or systematic in {list(e.index[e.Epeak_sys.isna()])}")
    e["err"] = np.hypot(0.5 * (e.Epeak_err_lo + e.Epeak_err_hi), e.Epeak_sys)
    e["err_instr"] = np.hypot(e.err, e.Epeak_keV * e.instrument_frac.fillna(0.0))
    return e


mvt = {"230307A": readMvt("230307A"), "211211A": readMvt("211211A")}
mvt1000 = readMvt("230307A", "_8-1000keV")
ep = {b: readEpeak(b) for b in ("230307A", "211211A")}
for b, m in mvt.items():
    missing = sorted(set(m.index) - set(ep[b].index))
    if missing:
        sys.exit(f"STOP: GRB {b}: MVT windows with no E_peak: {missing}")
dich = pd.DataFrame({"mvt_s": mvt1000.dichiara_mvt_s, "mvt_err_lo_s": mvt1000.dichiara_err_s,
                     "mvt_err_hi_s": mvt1000.dichiara_err_s, "upper_limit": mvt1000.dichiara_upper_limit.astype(bool)})
try:
    commit = subprocess.run(["git", "-C", args.inaki, "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
except Exception:
    commit = ""
print(f"E_peak: Iñaki's files in {args.inaki}" + (f" (commit {commit})" if commit else ""))


def lightCurve():
    files = {d: sorted(glob.glob(os.path.join(args.tte, f"glg_tte_{d}_bn230307656_v*.fit"))) for d in ("n6", "n8", "na")}
    if not all(files.values()):
        print("light curve: TTE files not found, drawn without it")
        return None
    from gdt.missions.fermi.gbm.tte import GbmTte
    edges = np.round(np.arange(-2.0 / 0.064, 66.0 / 0.064 + 1) * 0.064, 9)
    lc = sum(np.histogram(GbmTte.open(f[-1]).slice_energy((8.0, 1000.0)).data.times, bins=edges)[0] for f in files.values())
    return 0.5 * (edges[1:] + edges[:-1]), lc / lc.max()


def mid(df):
    return 0.5 * (df.t_start + df.t_stop)


def half(df):
    return 0.5 * (df.t_stop - df.t_start)


def drawMvt(ax, m, e, colour, label, marker="o", filled=True, dx=0.0):
    j = m[["mvt_s", "mvt_err_lo_s", "mvt_err_hi_s", "upper_limit"]].join(e[["t_start", "t_stop"]])
    kw = dict(color=colour, mfc=colour if filled else "white", mec=colour, ms=4.5, lw=1.4)
    val, lim = j[~j.upper_limit], j[j.upper_limit]
    if len(val):
        ax.errorbar(mid(val) + dx, 1e3 * val.mvt_s, xerr=half(val), yerr=[1e3 * val.mvt_err_lo_s, 1e3 * val.mvt_err_hi_s],
                    fmt=marker, label=label, **kw)
    if len(lim):
        ax.errorbar(mid(lim) + dx, 1e3 * lim.mvt_s, xerr=half(lim), yerr=0.15 * 1e3 * lim.mvt_s, uplims=True, fmt=marker,
                    label=None if len(val) else label, **kw)


# ---------------------------------------------------------------- figure 1: Dichiara Fig. 2
fig, (top, bot) = plt.subplots(2, 1, figsize=(7.5, 7.8), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
lc = lightCurve()
if lc is not None:
    top.fill_between(lc[0], 45 * lc[1], step="mid", color=RED, alpha=0.10, lw=0, label="GRB 230307A light curve")
    bot.fill_between(lc[0], 30 * 60 ** lc[1], step="mid", color=RED, alpha=0.10, lw=0)
drawMvt(top, mvt["230307A"], ep["230307A"], RED, "GRB 230307A, 8-900 keV")
drawMvt(top, mvt1000, ep["230307A"], RED, "GRB 230307A, 8-1000 keV", filled=False, dx=-0.7)
drawMvt(top, dich, ep["230307A"], "k", "Dichiara et al. (2023), 8-1000 keV", marker="D", filled=False, dx=0.7)
drawMvt(top, mvt["211211A"], ep["211211A"], GREY, "GRB 211211A, 8-900 keV", marker="s")
top.set_ylabel("Minimum variability timescale [ms]")
top.set_ylim(0, 55)
top.legend(fontsize=7.5, loc="upper right")

e = ep["230307A"]
shown = e.loc[EPEAK_RED_230307A]
others = e[~e.index.isin(EPEAK_RED_230307A) & (e.index != "t1_18_40")]
bot.errorbar(mid(others), others.Epeak_keV, xerr=half(others), yerr=others.err, fmt="none", ecolor=RED, alpha=0.3, lw=1)
bot.errorbar(mid(shown), shown.Epeak_keV, xerr=half(shown), yerr=shown.err_instr, fmt="none", ecolor=RED, alpha=0.3, lw=4)
bot.errorbar(mid(shown), shown.Epeak_keV, xerr=half(shown), yerr=shown.err, fmt="o", color=RED, ms=4.5, lw=1.4,
             label="GRB 230307A")
w = e.loc["t1_18_40"]
bot.hlines(w.Epeak_keV, w.t_start, w.t_stop, colors=RED, linestyles="--", lw=1.2, label="GRB 230307A, 18-40 s")
g = ep["211211A"].loc[EPEAK_SHOWN_211211A]
bot.errorbar(mid(g), g.Epeak_keV, xerr=half(g), yerr=g.err, fmt="s", color=GREY, ms=4.5, lw=1.4, label="GRB 211211A (best model)")
gb = g[g.Epeak_band_keV.notna() & (g.model != "Band")].loc[["f2_01_10"]]
bot.errorbar(mid(gb), gb.Epeak_band_keV, xerr=half(gb), fmt="s", color=GREY, mfc="white", ms=4.5, lw=1.0,
             label="GRB 211211A, 1-10 s, Band")
bot.set_yscale("log")
bot.set_ylim(30, 3000)
bot.set_ylabel(r"$E_{\rm peak}$ [keV]")
bot.set_xlabel("Time since GBM trigger [s]")
bot.set_xlim(-2, 66)
bot.legend(fontsize=7.5, loc="upper right", ncol=2)
for ax in (top, bot):
    ax.axvspan(*BAD_TIME, color="grey", alpha=0.12, lw=0)
    ax.tick_params(direction="in", top=True, right=True)
fig.tight_layout()
p1 = os.path.join(OUT, "joint_dichiara_fig2.png")
fig.savefig(p1, dpi=200)

# ---------------------------------------------------------------- figure 2: E_peak against MVT (8-900 keV)
fig, ax = plt.subplots(figsize=(6.4, 4.9))
for b, colour in (("230307A", RED), ("211211A", GREY)):
    j = mvt[b][["mvt_s", "mvt_err_lo_s", "mvt_err_hi_s", "upper_limit"]].join(ep[b])
    for lab, r in j.iterrows():
        x, y = 1e3 * r.mvt_s, r.Epeak_keV
        mk, ms = ("*", 12) if lab == "pre" else (("o" if b == "230307A" else "s"), 6)
        ax.errorbar(x, y, yerr=r.err_instr, fmt="none", ecolor=colour, alpha=0.3, lw=4)
        if r.upper_limit:
            ax.errorbar(x, y, xerr=0.25 * x, yerr=r.err, xuplims=True, fmt=mk, color=colour, ms=ms)
        else:
            ax.errorbar(x, y, xerr=[[1e3 * r.mvt_err_lo_s], [1e3 * r.mvt_err_hi_s]], yerr=r.err, fmt=mk, color=colour, ms=ms)
        ax.annotate(f"{r.t_start:g} to {r.t_stop:g} s", (x, y), textcoords="offset points", xytext=(7, 5), fontsize=7.5, color=colour)
ax.plot([], [], "o", color=RED, label="GRB 230307A")
ax.plot([], [], "s", color=GREY, label="GRB 211211A")
ax.plot([], [], "*", color="k", ms=10, label="precursor")
ax.set(xscale="log", yscale="log", xlabel="Minimum variability timescale, 8-900 keV [ms]", ylabel=r"$E_{\rm peak}$ [keV]")
ax.set_xlim(1.5, 80)
ax.set_ylim(40, 3000)
ax.legend(fontsize=8, loc="upper right")
ax.tick_params(which="both", direction="in", top=True, right=True)
fig.tight_layout()
p2 = os.path.join(OUT, "joint_epeak_vs_mvt.png")
fig.savefig(p2, dpi=200)

# ---------------------------------------------------------------- the joined table, as plotted
rows = []
for b in ("230307A", "211211A"):
    j = mvt[b].join(ep[b][["t_start", "t_stop", "model", "Epeak_keV", "err", "instrument_frac"]], how="left")
    j.insert(0, "burst", f"GRB{b}")
    rows.append(j[["burst", "t_start", "t_stop", "mvt_s", "mvt_err_lo_s", "mvt_err_hi_s", "upper_limit", "class", "trusted",
                   "model", "Epeak_keV", "err", "instrument_frac"]].rename(columns={"err": "Epeak_err_stat_sys"}))
pd.concat(rows).to_csv(os.path.join(OUT, "joint_table.csv"), float_format="%.6g")
print(f"saved {p1}\n      {p2}\n      {os.path.join(OUT, 'joint_table.csv')}")
