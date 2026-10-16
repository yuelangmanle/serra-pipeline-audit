#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The naive TE eleven-alignment null, rebuilt under the exact convention of
the 42.2% headline (ablation ladder rung T0).

The naive layer keeps its own estimator here on purpose: detrend, naive
z-score, Schreiber-form TE with the naive (Euclidean-joint) Kraskov
estimate -- the same convention as the headline number this null is read
against. Whole-year rotations use the shared year-block rotation, applied
to the raw packed series before detrending, the order the corrected-layer
build uses.

Anchor gate: k = 0 must reproduce share = 207/491 = 0.4216 before any
alignment row is trusted.
Writes data/results/tables/te_alignments_naive_rerun.csv.
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

from te import (naive_read_gsim, naive_build_wq, naive_merge_rows,  # noqa: E402
                legacy_transfer_entropy, zscore)
from pipeline import detrend, year_shift  # noqa: E402

KS = list(range(1, 12))


def main():
    t0 = time.time()
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv", low_memory=False)
    res = res[res.n_months >= 60].reset_index(drop=True)
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations_v2.csv")
    qids = set(qual.gems_id)
    wq_all = naive_build_wq(qids)

    combos = []
    for _, r in res.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        series = {ym: v[param] for ym, v in wq_all.get(gid, {}).items()
                  if param in v}
        flow_df = naive_read_gsim(sid)
        if flow_df is None or len(series) < 30:
            continue
        mg = naive_merge_rows(flow_df, series)
        if len(mg) < 60:
            continue
        combos.append((gid, sid, param,
                       mg.flow.values.astype(float),
                       mg.wq.values.astype(float)))
    print(f"naive combos: {len(combos)}", flush=True)

    def te_vote(xp_, yp_):
        te_f = legacy_transfer_entropy(xp_, yp_, lag=1)
        te_w = legacy_transfer_entropy(yp_, xp_, lag=1)
        return 1 if te_f > te_w else 0

    rows = []
    for i, (gid, sid, param, x, y) in enumerate(combos):
        # processed first (detrend + naive z-score), then whole-year rotation:
        # rotating the raw trended series would leave a saw-tooth behind
        sx = zscore(detrend(x))
        sy = zscore(detrend(y))
        votes = {0: te_vote(sx, sy)}
        for k in KS:
            votes[k] = te_vote(year_shift(sx, k), sy)
        for k, v in votes.items():
            rows.append({"pipeline": "te_naive_s6", "k": k, "vote": v,
                         "gems_id": gid, "gsim_id": sid, "parameter": param})
        if (i + 1) % 50 == 0:
            el = (time.time() - t0) / 60
            g = pd.DataFrame(rows)
            k0 = g[g.k == 0].vote
            print(f"{i+1}/{len(combos)} ({el:.1f} min) "
                  f"k=0 interim share {k0.mean():.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{T}/te_alignments_naive_rerun.csv", index=False)

    # ---- anchor gate + summary ----
    g = df[df.pipeline == "te_naive_s6"]
    k0_share = g[g.k == 0].vote.mean()
    n0 = g[g.k == 0].vote.count()
    expected = 207 / 491
    print(f"\nk=0 anchor: {int(round(k0_share * n0))}/{n0} = {k0_share:.4f} "
          f"(expected 207/491 = {expected:.4f})", flush=True)
    anchor_ok = abs(k0_share - expected) < 1e-9
    print(f"ANCHOR {'OK' if anchor_ok else 'MISMATCH -- DO NOT USE'}", flush=True)

    shares = g[g.k > 0].groupby("k").vote.mean()
    obs_share = k0_share
    exceed = int((shares >= obs_share).sum())
    below = int((shares <= obs_share).sum())
    print(f"observed (k=0, exact headline convention) share = {100*obs_share:.1f}%")
    print(f"11 alignment shares: mean {100*shares.mean():.1f} "
          f"range [{100*shares.min():.1f}, {100*shares.max():.1f}] "
          f"sd {100*shares.std():.1f}")
    print(f"alignments >= observed: {exceed}/11; alignments <= observed: {below}/11")
    print(f"upper-tail rank p <= {(exceed+1)/12:.3f}; "
          f"lower-tail rank p <= {(below+1)/12:.3f}")
    print(f"\nwritten: {T}/te_alignments_naive_rerun.csv "
          f"({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
