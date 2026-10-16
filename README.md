# Reproducibility package

Code and result tables for the manuscript "Pipeline choices manufacture
nominally significant directional claims in river discharge-nutrient
monitoring data" (SERRA submission).

## What runs from this package

- **Figures.** `figures/make_manuscript_figs.py` regenerates every manuscript
  figure (600 dpi TIFF/EPS/PNG/PDF) from the shipped result tables and the
  two processed lake series under `data/processed/`. No external downloads
  are needed. The panel-alignment gate and the collision audit write their
  QA records next to the figures.
- **Table checks.** Every number in the manuscript traces to a table under
  `data/results/tables/`; the table-to-script map below names the producer
  of each, or labels it frozen/superseded where the package ships it without
  a producing script.
- **Lake analyses.** `code/lakes_null.py` and `code/taihu_piecewise.py` run
  from `data/processed/` as well (Poyang's permutation null is enumerated
  exactly over all 7! = 5,040 year orders; Taihu uses 1,000 draws plus a
  three-seed sweep).

## What needs the external archives

`code/pipeline.py`, `code/te.py`, `code/strict_layer.py`, `code/shifts_naive.py`,
`code/te_naive_alignment.py`, `code/ccm_nulls.py`, `code/quick_checks.py`,
`code/robustness.py`, `code/cluster_robust.py`, `code/geo_robustness.py`,
`code/interp_halfrl.py`, `code/mixing_counts.py`, `code/ladder.py`,
`code/mccm_grid.py` and `code/mccm_naive_true.py` reconstruct the result
tables from source data. Set `PAPER_BASE` to a directory with:

- `data/raw/` - the public source archives: UNEP GEMS/Water v3 (Zenodo
  concept DOI 10.5281/zenodo.13881899), GSIM indices (PANGAEA
  10.1594/PANGAEA.887477, 10.1594/PANGAEA.887470), the Poyang water-level
  file (Figshare 10.6084/m9.figshare.30913100) and the TPDC Yangtze
  water-quality files (10.11888/Terre.tpdc.303497, 10.11888/Terre.tpdc.303501),
- `data/interim/` - the station-matching outputs (`ccm_qualified_stations*.csv`,
  `matched_river_stations_50km_10yr.csv`) that pair the archives,
- `data/processed/` - the two lake series (also shipped here),
- `data/results/tables/` - where tables are written.

Those raw and interim inputs are not redistributed here; every shipped table
is the frozen output of the scripts, so the tables themselves are verifiable
without downloading anything.

## Layout

- `code/` - the analysis scripts (current layer).
- `code/archive/` - earlier layers kept for the record: the first-generation figure
  scripts, the superseded first naive-TE null, the first lake-null design,
  the 200-realization synthetic run, and the superseded corrected-shift
  script. Not used by the current tables.
- `data/results/tables/` - one CSV per analysis (producer map below).
- `data/processed/` - the Poyang and Taihu series used by the lake scripts
  and Figure 6.
- `figures/` - the manuscript figure sources, the alignment helper they
  import, their QA records (alignment and collision-audit JSONs) and
  QA_NOTES.md.

## Corrections found in our own pipeline (all shipped side by side)

1. M-CCM direction labels. An earlier build passed the two series to the
   bidirectional routine in the wrong order, swapping the labels
   (manuscript Section 2.5). Both tables ship: `mccm_naive_labels_swapped.csv`
   is pre-correction (a swapped share reads 14.1%), `mccm_fixed.csv` is
   repaired (85.9%).
2. The lake null was redesigned once. The first design resampled 200
   year-shifts over an eleven-shift budget and "confirmed" Poyang water
   level -> TN; the year-block permutation with argmax-matched scoring
   reverses that (Section 3.4). Superseded builds: `lakes_backup.csv`,
   `lakes_null_discharge_cal.csv`.
3. The naive TE null was rebuilt under the exact convention of its headline
   statistic (`te_alignments_naive_rerun.csv`, 42.2% anchor). The superseded
   pairwise-statistic version is `naive_null.csv` (archive).
4. The corrected TE / conditional TE family uses ONE estimator definition:
   KSG-1 with the max-norm joint distance and strict marginal counting
   (Kraskov's rule), the definition the manuscript states. An earlier build
   counted marginal points inclusively (that variant still ships as the
   `*_inclusive` columns of `ksg_boundary_check.csv`), and a later strict
   side-check had quietly swapped in a Euclidean joint distance, which
   contradicted the documented geometry. The unified rebuild puts one
   definition in one place: shares, nulls, the strict-months check, the
   leave-one-country-out scan, and the cluster intervals all come from
   `strict_layer.py`.
5. The eleven-shift nulls rotate the PROCESSED series (gap-filled, detrended,
   standardized) by whole years; only the complete-year blocks enter the
   cyclic rotation and a trailing partial year stays in place, so every
   position keeps its calendar month at any record length. The flow record's
   own observation pattern rotates with its values, and each draw's analysis
   support is rebuilt from the rotated record -- the rotated flow pattern
   intersected with the water-quality record's own pattern, with the
   two-month short-gap rule bridging gaps exactly as it does for the
   observation, and the result intersected with the rotated pattern of the
   flow record's own usable months (observed, or a fill across at most two
   months) -- so every analyzed month of a draw carries a flow value of the
   same provenance class the observation draws on; the final intersection
   only bites at the rotation seam and the trailing partial-block junction,
   where the positional short-gap rule alone would let a truncated view of
   a longer gap enter (the second intersection removes 20 of the 671,147
   slot views the positional rule alone would analyse, leaving 671,127 in
   the shipped nulls; the affected months are listed in
   `data/results/tables/rotation_seam_audit.csv`). One helper, `pipeline.rotated_support`, is shared by
   the TE and CCM null scripts; the two naive-layer tables rotate packed rows in
   the naive frame, which carries no gap bookkeeping of its own). Where the
   complete-year count divides k the rotation is an exact identity and the
   draw reproduces the observation; `code/selfcheck.py` verifies that
   identity on the shipped tables, table row by table row. Earlier builds
   had (i) a plain `np.roll` of the raw order, which misaligned the wrapped
   months and rotated trended series in a way that left a saw-tooth behind
   the detrending step, (ii) a block rotation that carried the trailing
   partial year as its own block, which shifted every later block off its
   month phase for records whose length is not a multiple of twelve (235 of
   the 316), and (iii) a rotation scored over the observation's fixed month
   set (`strict_nulls.csv`, `strict_months_null.csv` and
   `ccm_shifts_corrected.csv`), whose rotated-in values could
   land on months the rotated flow record does not itself observe. All
   shift-based nulls (`strict_nulls.csv`, `strict_months_null.csv`,
   `ccm_shifts_corrected.csv`, `null_alignments_naive.csv`,
   `te_alignments_naive_rerun.csv`) were recomputed under the corrected
   rotation, and the month-phase property holds for every shipped
   combination.
6. The concordance tables were recomputed after correction 4 changed the TE
   and conditional-TE votes they are built from: `batch1_pairwise_kappa.csv`
   and `batch1_concperm.csv` now carry the point estimates and 5,000-draw
   station-block permutation bands of the current votes, and
   `batch1_concordance.csv` the naive-to-corrected flip summary.
7. The Taihu piecewise table ships in ONE schema: the piecewise columns are
   joined onto the lake-direction table (`taihu_piecewise.csv`, identical to
   the submission supplement `Supplementary_taihu_piecewise_detrend.csv`),
   and `taihu_piecewise.py` writes exactly that file. An earlier package
   carried an older copy of this table next to the script's different
   output.

## Table-to-script map

In the map below: a row written by a current script is regenerable with the
archives in place; a row labelled *frozen* is a shipped input of the current
producers that this package cannot regenerate; a row labelled *superseded*
is kept for the record only.

| Table | Produced by |
|---|---|
| `combos.csv`, `std_ccm.csv`, `deseason.csv`, `seasonal_r2.csv`, `mccm_naive_labels_swapped.csv` | `pipeline.py` |
| `te_z.csv`, `te.csv`, `ksg_boundary_check.csv`, `strict_nulls.csv`, `strict_months.csv`, `strict_months_null.csv`, `te_loco_strict.csv`, `cluster_robust_strict.csv` | `strict_layer.py` |
| `ablation_ladder_te_z.csv`, `transfer_entropy_491.csv`, `scale_invariance_checks.csv` | `te.py` |
| `ablation_ladder.csv` | `ladder.py` |
| `null_alignments_naive.csv` | `shifts_naive.py` |
| `te_alignments_naive_rerun.csv` | `te_naive_alignment.py` |
| `ccm_shifts_corrected.csv`, `ccm_shifts_n.csv` | `ccm_nulls.py` (the companion table carries each draw's sample size; draws below the 40-pair minimum are recorded as NaN skills and excluded from the draw's share) |
| `mccm_fixed.csv` | `mccm_fixed.py` (the documented label convention applied to `mccm_naive_labels_swapped.csv`; Section 2.5) |
| `synth_grid.csv`, `synth_mixture.csv` | `mccm_grid.py` |
| `mccm_naive_true.csv` | `mccm_naive_true.py` |
| `batch1_libshare.csv`, `batch1_selfflip.csv`, `batch1_flips.csv`, `batch1_concordance.csv`, `batch1_funnel.csv`, `batch1_te_loco.csv`, `batch1_concperm.csv`, `batch1_pairwise_kappa.csv` | `quick_checks.py` |
| `batch2_esweep.csv`, `batch2_aheadscan.csv`, `batch2_strict.csv`, `batch2_cq.csv` | `robustness.py` |
| `cluster_robust.csv`, `cluster_robust_mccm_fixed.csv` | `cluster_robust.py` |
| `geo_robustness.csv` | `geo_robustness.py` |
| `lakes_null.csv`, `dual_lake.csv` | `lakes_null.py`, `lakes.py` |
| `taihu_trend.csv`, `taihu_descriptives.csv` | `taihu_trend.py` (the descriptives table carries the Section 3.4 Chla–TP correlations and the TLI before-2019 vs 2019–2020 comparison) |
| `null_robustness.csv` | `null_robustness.py` (identity-draw exclusion, record-length strata, and library-split seed shares and vote flips for the eleven-shift nulls; the `n_unit` column states whether `n` counts draws or combinations) |
| `rotation_seam_audit.csv` | `null_robustness.py` (the months the provenance intersection removes at the rotation seam and the trailing-block junction; needs the archives) |
| `site_geometry.csv` | `site_geometry.py` (gauge-to-sampling separation record; needs the archives. The manuscript's 90th percentile moved from 12.2 to 12.0 km when this table pinned the linear-interpolation convention, and the candidate-stage median of 15.3 km is its `candidate_separation_median_km` row) |
| `taihu_piecewise.csv` | `taihu_piecewise.py` |
| `tn_aggregation_sensitivity.csv`, `deseasonalized_sametime_full.csv`, `true_xmap_sametime_full.csv`, `ccm_analysis_results.csv` | frozen from the original naive-layer runs (the row-index sampling ledger and its variants); kept as the naive-layer record and not regenerable from this package |
| `transfer_entropy.csv` | frozen from the original naive-layer conditional-TE run (272 combinations, n_months >= 60); read by `te.py` (step-4 cross-reference) and `ladder.py`; not regenerable from this package |
| `interp_and_halfrl.csv` | `interp_halfrl.py` (half-RL stage needs the archives) |
| `mixing_counts.csv` | `mixing_counts.py` (needs the archives) |
| `calendar_scan.csv` | frozen calendar-gap diagnostic consumed by `interp_halfrl.py` (not regenerable from this package; one `combos` entry is absent from the scan and one scanned entry lies outside the final sample -- `interp_halfrl.py` recomputes the missing entry from the archives and records both in `interp_and_halfrl.csv`) |
| `naive_null.csv`, `lakes_backup.csv`, `lakes_null_discharge_cal.csv`, `mccm_synthetic.csv` | superseded builds kept for the record, documented above (no producing script in this package) |

## Manuscript number -> table index (headline numbers)

| Manuscript number | Table |
|---|---|
| naive shares 60.1 / 47.5 / 63.1 / 42.2 / 67.0 | `ablation_ladder.csv`, `mccm_naive_true.csv`, `te_alignments_naive_rerun.csv`, `transfer_entropy_491.csv` |
| naive eleven-shift nulls (CCM and TE) | `null_alignments_naive.csv`, `te_alignments_naive_rerun.csv` |
| corrected CCM 51.9 / deseasonalized 46.8 and their own nulls | `std_ccm.csv`, `deseason.csv`, `ccm_shifts_corrected.csv`, `ccm_shifts_n.csv` |
| corrected M-CCM 85.9 (n = 313) | `mccm_fixed.csv`, `cluster_robust_mccm_fixed.csv` |
| TE / conditional TE (strict counting) and their nulls | `te_z.csv`, `strict_nulls.csv`, `ksg_boundary_check.csv` |
| TE cluster intervals (strict and unstandardized) | `cluster_robust_strict.csv` |
| strict-months TE and its own null | `strict_months.csv`, `strict_months_null.csv` |
| unstandardized TE share and zeros | `te.csv` |
| dose-response grid, mixture expectation | `synth_grid.csv`, `synth_mixture.csv` |
| concordance point estimates, bands and flips | `batch1_pairwise_kappa.csv`, `batch1_concperm.csv`, `batch1_concordance.csv` |
| per-country and leave-one-out scopes (all four methods) | `geo_robustness.csv`, `te_loco_strict.csv` |
| lake directions, three nulls each | `lakes_null.csv`; Taihu piecewise `taihu_piecewise.csv`; Section 3.4 structural descriptives `taihu_descriptives.csv` |
| Section 2.7 null robustness (identity exclusion, strata, split seeds and vote flips, rotation-seam audit) | `null_robustness.csv`, `rotation_seam_audit.csv` |
| gauge-to-sampling geometry sentences (median, 90th percentile, candidate-stage median) | `site_geometry.csv` |
| robustness batch (E x tau, ahead scan, library, c-Q) | `batch2_esweep.csv`, `batch2_aheadscan.csv`, `batch2_strict.csv`, `batch2_cq.csv` |
| funnel, species-mixing screen | `batch1_funnel.csv`, `mixing_counts.csv` |
| interpolation footprint, half-RL share | `interp_and_halfrl.csv` |

## Running

Full rebuild from the archives, in dependency order (producers as in the
map above; every command writes its tables under `data/results/tables/`):

    export PAPER_BASE=/path/to/project
    python3 code/pipeline.py            # combos, standard CCM, deseasonalized, seasonal r2, naive labels
    python3 code/te.py                  # standardized TE layer + ablation rungs
    python3 code/strict_layer.py        # corrected TE/CTE shares, shift nulls, strict-months check, LOCO, cluster CIs
    python3 code/ladder.py              # ablation ladder (reads the te.py outputs)
    python3 code/mccm_grid.py           # synthetic dose-response grid and mixture expectation
    python3 code/mccm_naive_true.py     # naive M-CCM share
    python3 code/quick_checks.py        # concordance batch (reads the corrected votes)
    python3 code/robustness.py          # E x tau sweep, ahead scan, library fraction, c-Q
    python3 code/cluster_robust.py
    python3 code/geo_robustness.py
    python3 code/ccm_nulls.py           # corrected CCM own shift nulls
    python3 code/shifts_naive.py        # naive CCM shift null + power block
    python3 code/te_naive_alignment.py  # naive TE shift null (42.2% anchor gate)
    python3 code/site_geometry.py       # gauge-to-sampling separation record
    python3 code/null_robustness.py     # identity exclusion, strata, split-seed stability
    python3 code/interp_halfrl.py       # interpolation footprint (half-RL stage)
    python3 code/mixing_counts.py       # species-mixing screen
    python3 code/selfcheck.py           # month-phase certificate, table keys, producer map, identity shifts, null-value integrity

Lake analyses and the figures run from the shipped inputs alone, in any
order relative to the block above:

    python3 code/lakes_null.py && python3 code/lakes.py
    python3 code/taihu_trend.py && python3 code/taihu_piecewise.py
    cd figures && PAPER_BASE=$PWD/.. python3 make_manuscript_figs.py

Minimal verification without the archives:

    PAPER_BASE=$PWD python3 code/selfcheck.py   # five of the six checks; the six-check
                                                 # certificate needs data/raw for the archive cross-check

(`selfcheck.py` runs six checks. Two are the month-phase certificate: the pure
operator test over the 166 shipped calendar lengths, which runs from the package
alone, and the archive month-vector cross-check, which re-reads the real
calendars when the archives are supplied and also asserts that every analysed
draw value comes from an observed month or a gap of at most two months of the
base record (never a long-gap interpolation). Four are table checks that run
from the package alone: key integrity (complete keys and anchor rows, strict
draws complete, reduced month-draw coverage printed), producer integrity (every
result table written by exactly one current script or labelled frozen/
superseded, zero undeclared), identity exactness (every identity rotation
reproduces the observed row), and null-value integrity (all values finite; NaN
CCM draws matched to sub-40 sample sizes in the companion table). With the
archives supplied it prints ALL PASS (6 of 6); on the package alone it certifies
five of six, prints the archive cross-check as not executed, and closes with a
partial-certificate line rather than a full pass.)

Python 3.9 with the versions in `requirements.txt`. Every stochastic step
fixes its seed. Tables are written to `data/results/tables/`.
