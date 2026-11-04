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
| run_injection_recovery.py | Injection-recovery test: 7 synthetic pulses (Gaussian G1-G30, Norris N3_10-N30_100) on a 1000 counts/s Poisson background, amplitudes 0.3-300 plus a bright run (3000), bin widths 0.1, 0.01 and 0.003 ms, 300 independent simulations per setting; true MVT by Bala et al.'s rule, SNR_MVT, class, criteria 1-5, plots P8 and P9 | 37da31f (Haar code as 9ad6fbd) | injection_recovery, plots/P8, P9 |
| analyse_injection_saved.py | Analysis A of the saved injection runs (no reruns): valid / upper-limit / failed split, repeated values and clusters, criterion 1 against the bright median at the same bin width | none (reads results) | injection_recovery/analysis_A, plots/IR_hist_* |
| analyse_rung_positions.py | Analysis A4 (no reruns): position of every MVT between the rungs of the Haar timescale ladder (4 per doubling) and KS test against uniform, for the injection settings and the saved real-burst Monte Carlo runs | none (reads results) | injection_recovery/analysis_A4, plots/IR_rungs_*, IR_rung_position, REAL_rungs, REAL_rung_position |
| run_b0_ladder.py | B0: the injection simulations (same photons) analysed with 8 rungs per doubling (outputBinningRatio = 8) instead of 4 | 37da31f (Haar code as 9ad6fbd) | injection_recovery/b0_ratio8, plots/B0_* |
| run_b1_truth.py | B1: true MVT of the narrow pulses with bin widths down to 0.0001 ms (same photons), settled check, criteria 1-4 recomputed | 37da31f (Haar code as 9ad6fbd) | injection_recovery/b1_truth, plots/B1_* |
| run_b3_threshold.py | B3/B5: lowest SNR_MVT from which the median MVT is within 20% of the bright value at the same bin width; brightness steps of x1.2; standard conditions and precursor-like conditions (0.384 s window, 5000 counts/s) | 37da31f (Haar code as 9ad6fbd) | injection_recovery/b3_threshold, plots/B3_* |
| run_precursor_mvt.py | Step 8: MVT of the GRB 230307A precursor (-0.064 to 0.32 s, 8-900 keV): detector selection by Bala et al.'s rule (repeated at the measured MVT), real-data Monte Carlo at 0.1, 0.01 and 0.003 ms for the optimal set and all 12 NaI, background over a short window (overlap-weighted mean), physical and Bala-compatible SNR_MVT, class, plateau, comparison with Dichiara et al., B5 trust check, background Fano test (P11). Flags --dets, --post | 7d84592 | precursor_mvt/GRB230307A/ |
| run_precursor_check.py | Made-to-measure check: Gaussian and Norris pulses with the precursor's MVT, background, window and SNR (0.5, 1 and 2 times), main and real-data schemes; real Monte Carlo values against the rungs; neighbouring-bin background test; rules C1-C4 (P12) | 7d84592 | precursor_mvt/GRB230307A/check/ |
| run_precursor_artefacts.py | Data gaps (segments across detectors, false-alarm threshold from 50 time-shifted copies), dead time and window edge, on the real precursor and on the step-2 pulses; gap filling and dead-time correction; 20-window grid; rules Q0-Q4 and adoption A1/A2 (P13) | 7d84592 | precursor_mvt/GRB230307A/artefacts/ |
| run_precursor_template.py | Step 7: Haar-denoised templates of known pulses (7a: Gaussian, Norris, double Norris) and of the real precursor (7b gap variants, 7c fading); denoiser threshold factors 1, 1.5, 2, 3 tried in order. Result: stopped at 7a, no factor passes (templates shorten the MVT of Norris-shaped pulses by about 25%) (P14) | 7d84592 | precursor_mvt/GRB230307A/template/ |

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
- The injection-recovery scripts import run_injection_recovery.py, so they must sit in the same folder. They
  simulate photons directly (no merge), so the physical and Bala-compatible SNR_MVT are the same there.
  run_b0_ladder.py and run_b1_truth.py reuse the seeds of run_injection_recovery.py (same photons) and stop if the
  first realisations do not reproduce the saved values; run_b3_threshold.py uses new seeds. All are resumable.
- The precursor scripts read results/precursor_mvt/GRB230307A/summary.json and selection.json, so run_precursor_mvt.py runs first. run_precursor_check.py, run_precursor_artefacts.py and run_precursor_template.py import each other in that order and must sit in the same folder as run_injection_recovery.py and analyse_rung_positions.py. All are resumable and have --post.
- run_precursor_artefacts.py stops if its baseline does not reproduce step 8 exactly (Q0); run_precursor_template.py stops if its window counts do not reproduce artefact runs R0 and R1 (Z0, Z1). From run_precursor_template.py on, trust means: median of fakes within one ladder step (x/1.2 to x1.2) of the truth, and the truth inside the 16-84% range in at least 50% of fakes.
