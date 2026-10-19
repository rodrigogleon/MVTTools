"""
Check that run_grb_mvt.py --post reproduces run_snr_classify.py for GRB 230307A.
Compares every number in the runs, the verdict and the background table (old keys mapped to new ones).

Usage
  python compare_snr.py OLD_SUMMARY_JSON NEW_SUMMARY_JSON
"""
import sys, json

old, new = (json.load(open(p)) for p in sys.argv[1:3])
BKG_MAP = {"short": "fit_-1.64_18", "full": "fit_-1.6_45.3",
           "shortAll": "fitAll_-1.64_18", "fullAll": "fitAll_-1.6_45.3",
           "redChi2": "redChi2", "offBurstRaw": "offBurstRaw"}
diffs, checked = [], 0


def walk(a, b, path):
    global checked
    if isinstance(a, dict):
        if not isinstance(b, dict):
            diffs.append(f"{path}: type differs")
            return
        for k in a:
            if k not in b:
                diffs.append(f"{path}.{k}: missing in new")
            else:
                walk(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            diffs.append(f"{path}: list differs")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    else:
        checked += 1
        if a != b:
            diffs.append(f"{path}: old {a} new {b}")


for key in ("seed", "nPoisson", "energyRange_keV", "bkgIntervals_s", "mvttoolsCommits", "verdict", "runs"):
    walk(old[key], new[key], key)
for det, row in old["background"].items():
    for k, v in row.items():
        walk(v, new["background"][det][BKG_MAP[k]], f"background.{det}.{k}")

print(f"values compared: {checked}")
if diffs:
    print(f"DIFFERENCES ({len(diffs)}):")
    for d in diffs[:50]:
        print("  " + d)
else:
    print("IDENTICAL: every value matches")
