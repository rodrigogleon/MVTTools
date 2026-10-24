# Analysis runs (GRB 230307A and GRB 211211A, Fermi GBM)

Scripts used to produce the results in the project logbook. They run on the Chandra server
(paths under /mnt/data/rgl36/FYP) and use the MVTTools code in this repository.

| Script | Purpose | Code (commit) | Output folder (results/) |
|---|---|---|---|
| run_baseline.py | Original pipeline, 0.1 ms, 5 detectors | stable (original code) | GRB230307A_baseline |
| run_mcfix.py | Monte Carlo with valid-only median and 16/84 percentiles | e16f408 | GRB230307A_mcfix |
| run_mcfix_v2.py | As above, one random seed per realisation (seed 2026) | 71b1c78 | GRB230307A_mcfix_v2 |
| run_binwidth.py | Bin-width check at 0.01 ms, 8-1000 keV, fixed errors (runs A, B) | 71b1c78 | GRB230307A_bin_* |
| run_bala_compare.py | Single comparison run with Bala et al. Table 3 (old C, 8-1000 keV) | 71b1c78 | GRB230307A_bala_* |
| run_bala_rows.py | Bala et al. (2026) Table 3 rows for GRB 230307A, 8-900 keV, resampled errors (runs B-H, Ds) | 71b1c78, then 9ad6fbd | bala_rows_8-900keV_rs, bala_rows_8-900keV_rs_haarfix |
| run_code_compare.py | Our Haar code vs Bala et al.'s original Haar code on identical inputs | 9ad6fbd | code_compare |
| run_snr_classify.py | Background, SNR_MVT and validation-curve class for every GRB 230307A haar-fix row (post-processing, no Monte Carlo) | reads 9ad6fbd results | snr_classify |
| run_extra_bkg.py | "Cleaner light curve" test: SNR matched to Table 3 by extra background (bkg) or a fainter source (scale), Monte Carlo rerun for runs B and F | 5f7fd0c (Haar code as 9ad6fbd) | extra_bkg (RunB_bkg, RunF_bkg, RunB_scale, RunF_scale) |
| run_detector_selection.py | Detector selection by iterative SNR maximisation (Bala et al. 2026, Section 4.1), at 64 ms and at bin width = MVT; used for both bursts | none (GDT only) | detector_selection |
| run_bala_rows_211211A.py | Bala et al. Table 3 rows for GRB 211211A (runs P-V); same method as run_bala_rows.py | 5add159 (Haar code as 9ad6fbd) | bala_rows_211211A |
| run_grb_mvt.py | Full workflow for any burst listed in it: Monte Carlo (--run), background, SNR_MVT, class, full Table 3 comparison and plateau verdict (--post). Class from a Bala-compatible SNR (detectors merged with GbmTte.merge, all-channel background); physical SNR also reported. Replaces run_bala_rows*.py and run_snr_classify.py | 5add159 (Haar code as 9ad6fbd) | Monte Carlo: the run_bala_rows*.py folders; --post: snr_classify/GRB230307A, snr_classify/GRB211211A |
| run_grb_mvt_v1.py | Previous version of run_grb_mvt.py (class from the physical SNR only) | 5add159 | as above |
| compare_snr.py | Check that run_grb_mvt.py --post reproduces run_snr_classify.py for GRB 230307A value by value | none | none |
| run_merge_test.py | SNR_MVT with detectors combined by GbmTte.merge (force_unique=True) vs all photons, for every Table 3 row of both bursts | none (GDT only) | merge_test |
| run_merged_mc.py | Monte Carlo on merged light curves (GRB 230307A runs B and F), compared with our all-photon runs and Table 3 | 5add159 (Haar code as 9ad6fbd) | merged_mc |
| run_coincidence_test_v1.py | First test of the photons removed by GbmTte.merge against a random-coincidence prediction (tick taken as 1.907 microseconds; superseded) | none (GDT only) | coincidence_test_v1, plots/P7 (v1) |
| run_pair_gaps.py | Pairs of events in two detectors counted against their time gap: excess at zero gap = genuine duplicates (one event recorded by several detectors); burst and background-only windows, energies and detector pairs | none (GDT only) | pair_gaps, plots/P7b |
| run_coincidence_test.py | Photons removed by GbmTte.merge against count rate, compared with random coincidences (2 microsecond tick) plus a steady duplicate rate measured in a background-only window (plot P7) | none (GDT only) | coincidence_test, plots/P7 |
| make_plots.py | Benchmark plots P1-P6 from saved results only (no reruns); --only P4 makes single plots; prints every plotted value and warns if it differs from the saved summaries | none (reads results) | plots |

Notes
- run_bala_rows.py is the current version for GRB 230307A (output folder bala_rows_8-900keV_rs_haarfix; it also
  records the code commit in each summary.json). The first set (bala_rows_8-900keV_rs) was made with the same
  script before those two changes. For new work use run_grb_mvt.py.
- run_code_compare.py, run_snr_classify.py, run_grb_mvt.py and make_plots.py need Bala et al.'s public code
  (GBM_MVT_paper, folder MVTfermi); they use their clasify_mvt_value.py and mvt_snr_fit_model.npz.
- Background (run_snr_classify.py, run_detector_selection.py, run_grb_mvt.py, run_merge_test.py): first-order
  polynomial fitted to the off-burst intervals on 1.024 s bins, as in Bala et al.'s code
  (GRB 230307A: [-68.8, -18.8] and [69.4, 119.4] s; GRB 211211A: [-68.0, -18.0] and [69.7, 119.7] s).
- run_extra_bkg.py needs the outputs of run_bala_rows.py (haar-fix) and run_snr_classify.py, and stops if its
  light curve is not identical to the haar-fix one.
- GBM event times are stored in ticks of 2 microseconds (measured from the data; stored differences of 1.907 or
  2.027 microseconds are floating-point rounding of 2 microseconds). run_coincidence_test_v1.py used 1.907
  microseconds, which made its random prediction about 5% too low.
- Our light curves keep every photon (detectors' counts are added). GbmTte.merge with its default
  force_unique=True, used in Bala et al.'s code, keeps one photon per arrival-time tick, so it deletes coincident
  events from different detectors. run_merge_test.py and run_merged_mc.py show that this, together with an
  all-channel background, reproduces their SNR_MVT and most of their Monte Carlo results. run_pair_gaps.py and
  run_coincidence_test.py show that the deletions are random coincidences of separate photons plus a steady rate
  of genuine duplicates from the background (87-98% of deletions during the bursts, over 99% at the peaks, are
  real photons). We therefore measure MVT and SNR with all photons, and use the merged SNR only to apply Bala et
  al.'s validation curve, which was calibrated with it.
- run_pair_gaps.py reads results/coincidence_test/summary.json for its criterion 3; its logged run used the v1
  output (now in results/coincidence_test_v1). Rerunning it against the v2 output compares with a different
  prediction (v2 includes the duplicates).
- From run_extra_bkg.py onwards, screen output is saved in results/logs/.
- All Monte Carlo runs use 300 realisations and seed 2026 unless stated otherwise.
