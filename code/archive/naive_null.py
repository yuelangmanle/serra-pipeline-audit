#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-coupling baselines for the NAIVE pipeline.

The manuscript convicts the naive pipeline's 60.1% and 42.2% against the 50%
line, but its own surrogate nulls proved 50% is not the correct no-coupling
line for M-CCM or conditional TE. This script estimates the naive pipeline's
own whole-year-shift null shares on the 491 naive combinations:
  - naive std CCM (detrended, same-time, mixed species, row-index lags)
  - naive unconditional TE (naive mixed Euclidean-KSG estimator, z-scored)
Surrogate: flow shifted by 12 x k months (k in 1..11), N=100 draws per
combination; naive vote recomputed per surrogate.

Reuses the naive layer of ladder.py (C0 anchor = 60.1% verified
there). Conventions follow the ladder: build_series_rowindex returns
detrended packed-row series; TE rows z-score after detrend.
Output: data/results/tables/naive_null.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
import ladder as lad  # noqa: E402

T = f"{ROOT}/data/results/tables"
N_SURR = 100
SEED = 42


def main():
    qids = set(pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv").gems_id)
    wq_all = lad.naive_build_wq(qids)
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv", low_memory=False)
    res = res[res.n_months >= 60].reset_index(drop=True)
    print(f"naive combos: {len(res)}", flush=True)

    rng = np.random.RandomState(SEED)
    rows = []
    for i, r in res.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        series = {ym: v[param] for ym, v in wq_all.get(gid, {}).items() if param in v}
        built = lad.build_series_rowindex(lad.naive_read_gsim(sid), series)
        if built is None:
            continue
        x, y = built                      # detrended packed-row naive series
        f2w, w2f = lad.naive_xmap_bidir(x, y)
        if np.isnan(f2w) or np.isnan(w2f):
            continue
        # naive TE (z-scored, packed rows)
        xz, yz = lad.zscore(x), lad.zscore(y)
        te_f = lad.uncond_te(xz, yz, lad.naive_kraskov_mi)
        te_w = lad.uncond_te(yz, xz, lad.naive_kraskov_mi)
        shifts = [12 * rng.randint(1, 12) for _ in range(N_SURR)]
        ccm_votes, te_votes = [], []
        for k in shifts:
            xs = np.roll(x, k)
            a, b = lad.naive_xmap_bidir(xs, y)
            if not (np.isnan(a) or np.isnan(b)):
                ccm_votes.append(1 if a > b else 0)
            xs2, ys2 = lad.zscore(xs), yz
            ta = lad.uncond_te(xs2, ys2, lad.naive_kraskov_mi)
            tb = lad.uncond_te(ys2, xs2, lad.naive_kraskov_mi)
            if not (np.isnan(ta) or np.isnan(tb)):
                te_votes.append(1 if ta > tb else 0)
        rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                     "obs_vote_ccm": 1 if f2w > w2f else 0,
                     "obs_vote_te": 1 if te_f > te_w else 0,
                     "null_ccm_mean": float(np.mean(ccm_votes)) if ccm_votes else np.nan,
                     "null_te_mean": float(np.mean(te_votes)) if te_votes else np.nan})
        if (i + 1) % 100 == 0:
            print(f"done {i+1}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{T}/naive_null.csv", index=False)
    print(f"\nn combos with naive CCM: {df.obs_vote_ccm.notna().sum()}")
    print(f"observed naive std-CCM share: {df.obs_vote_ccm.mean():.3f}")
    print(f"null expected flow-share (whole-year shift): {df.null_ccm_mean.mean():.3f}")
    print(f"  per-combo null mean quantiles: "
          f"{df.null_ccm_mean.quantile([.1, .5, .9]).round(3).to_dict()}")
    print(f"observed naive TE share: {df.obs_vote_te.mean():.3f}")
    print(f"null expected flow-share (TE): {df.null_te_mean.mean():.3f}")


if __name__ == "__main__":
    main()
