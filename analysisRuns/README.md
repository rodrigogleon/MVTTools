# Analysis runs (GRB 230307A, Fermi GBM)

Scripts used to produce the results in the project logbook. They run on the Chandra server
(paths under /mnt/data/rgl36/FYP) and use the MVTTools code in this repository.

| Script | Purpose | Code (commit) | Output folder (results/) |
|---|---|---|---|
| run_baseline.py | Original pipeline, 0.1 ms, 5 detectors | stable (original code) | GRB230307A_baseline |
| run_mcfix.py | Monte Carlo with valid-only median and 16/84 percentiles | e16f408 | GRB230307A_mcfix |
| run_mcfix_v2.py | As above, one random seed per realisation (seed 2026) | 71b1c78 | GRB230307A_mcfix_v2 |
| run_binwidth.py | Bin-width check at 0.01 ms, 8-1000 keV, fixed errors (runs A, B) | 71b1c78 | GRB230307A_bin_* |
| run_bala_compare.py | Single comparison run with Bala et al. Table 3 (old C, 8-1000 keV) | 71b1c78 | GRB230307A_bala_* |
| run_bala_rows.py | Bala et al. (2026) Table 3 rows, 8-900 keV, resampled errors (runs B-H, Ds) | 71b1c78, then 9ad6fbd | bala_rows_8-900keV_rs, bala_rows_8-900keV_rs_haarfix |
| run_code_compare.py | Our Haar code vs Bala et al.'s original Haar code on identical inputs | 9ad6fbd | code_compare |

Notes
- run_bala_rows.py is the current version (output folder bala_rows_8-900keV_rs_haarfix; it also
  records the code commit in each summary.json). The first set (bala_rows_8-900keV_rs) was made
  with the same script before those two changes.
- run_code_compare.py needs Bala et al.'s public code (GBM_MVT_paper, folder MVTfermi).
- All Monte Carlo runs use 300 realisations and seed 2026 unless stated otherwise.
