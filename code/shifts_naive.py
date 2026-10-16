#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Naive-layer own null: the standard-CCM share of the naive pipeline under
whole-year rotations of the flow series.

The naive layer works on packed rows (its own row-index convention), so the
rotation is a twelve-row block rotation of the packed series. The observed
layer is appended at k = 0 for anchoring. The naive TE null lives in
te_naive_alignment.py (rebuilt under the exact convention of the 42.2%
headline); the corrected-layer TE/CTE nulls live in strict_layer.py. The
power block at the end documents the resolution statement of Section 4.1.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
import ladder as lad  # noqa: E402
from pipeline import year_shift  # noqa: E402

T = f"{ROOT}/data/results/tables"


def naive_layer():
    qids = set(pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv").gems_id)
    wq_all = lad.naive_build_wq(qids)
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv", low_memory=False)
    res = res[res.n_months >= 60].reset_index(drop=True)
    out = []
    for _, r in res.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        series = {ym: v[param] for ym, v in wq_all.get(gid, {}).items() if param in v}
        built = lad.build_series_rowindex(lad.naive_read_gsim(sid), series)
        if built is None:
            continue
        out.append((gid, sid, param, built[0], built[1]))
    return out


def main():
    ks = list(range(1, 12))

    # ---- naive std CCM + naive TE (row-index layer) ----
    rows = []
    combos = naive_layer()
    print(f"naive combos: {len(combos)}", flush=True)
    for i, (gid, sid, param, x, y) in enumerate(combos):
        f2w, w2f = lad.naive_xmap_bidir(x, y)
        obs = 1 if f2w > w2f else 0
        for k in ks:
            xs = year_shift(x, k)
            a, b = lad.naive_xmap_bidir(xs, y)
            rows.append({"pipeline": "ccm_naive", "k": k,
                         "vote": 1 if a > b else 0,
                         "gems_id": gid, "gsim_id": sid, "parameter": param})
        rows.append({"pipeline": "ccm_naive", "k": 0, "vote": obs,
                     "gems_id": gid, "gsim_id": sid, "parameter": param})
        if (i + 1) % 100 == 0:
            print(f"naive done {i+1}", flush=True)
    df_n = pd.DataFrame(rows)
    df_n.to_csv(f"{T}/null_alignments_naive.csv", index=False)
    g = df_n[df_n.pipeline == "ccm_naive"]
    obs_share = g[g.k == 0].vote.mean()
    shares = g[g.k > 0].groupby("k").vote.mean()
    exceed = int((shares >= obs_share).sum())
    print(f"\nccm_naive: observed {obs_share:.3f}")
    print(f"  11 alignment shares: mean {shares.mean():.3f} "
          f"range [{shares.min():.3f}, {shares.max():.3f}] "
          f"sd {shares.std():.3f}")
    print(f"  alignments reaching observed share: {exceed}/11 "
          f"-> rank p <= {(exceed+1)/12:.3f}")

    # power analysis for the share test at n = 316
    from scipy import stats as st
    n = 316
    z_a, z_b = st.norm.ppf(0.975), st.norm.ppf(0.80)
    for p_true in (0.55, 0.58, 0.60, 0.65):
        se0 = (0.5 * 0.5 / n) ** 0.5
        se1 = (p_true * (1 - p_true) / n) ** 0.5
        crit = 0.5 + z_a * se0
        power = 1 - st.norm.cdf((crit - p_true) / se1) + st.norm.cdf((-crit - p_true) / se1)
        print(f"share {p_true:.2f}: power = {power:.2f}")


if __name__ == "__main__":
    main()
