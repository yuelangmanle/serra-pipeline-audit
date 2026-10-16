#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Naive-layer M-CCM share (same-time, all-species, row-index lags).

The naive number in the manuscript comes from here. An earlier build had
reused the corrected-layer output with labels mirrored as the "naive"
value; that file is kept only for the label-convention discussion.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"
sys.path.insert(0, HERE)
import ladder as lad  # noqa: E402
import pipeline as gp  # noqa: E402


def main():
    t0 = time.time()
    qids = set(pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations_v2.csv").gems_id)
    wq_all = lad.naive_build_wq(qids)
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv", low_memory=False)
    res = res[res.n_months >= 60].reset_index(drop=True)

    rows = []
    for i, r in res.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        series = {ym: v[param] for ym, v in wq_all.get(gid, {}).items() if param in v}
        built = lad.build_series_rowindex(lad.naive_read_gsim(sid), series)
        if built is None:
            continue
        x, y = built[0], built[1]
        if len(x) < 60:
            continue
        months = np.arange(len(x)) % 12
        z = np.column_stack([np.sin(2 * np.pi * months / 12),
                             np.cos(2 * np.pi * months / 12)])
        valid = np.isfinite(x) & np.isfinite(y)
        f2w, nf, _ = gp.ccm_run(y, x, valid, ahead=0, controls=z)
        w2f, nw, _ = gp.ccm_run(x, y, valid, ahead=0, controls=z)
        if np.isnan(f2w) or np.isnan(w2f):
            continue
        rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                     "mccm_f2w": f2w, "mccm_w2f": w2f})
        if (i + 1) % 100 == 0:
            g = pd.DataFrame(rows)
            k = int((g.mccm_f2w > g.mccm_w2f).sum())
            print(f"{i+1} combos, interim share {100*k/len(g):.1f}% "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{T}/mccm_naive_true.csv", index=False)
    n = len(out)
    k = int((out.mccm_f2w > out.mccm_w2f).sum())
    from scipy import stats
    print(f"\nTRUE naive M-CCM: n={n} share={100*k/n:.1f}% ({k}/{n}) "
          f"one-sided binom p={stats.binomtest(k, n, 0.5, alternative='greater').pvalue:.3g}")
    print(f"written {T}/mccm_naive_true.csv ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
