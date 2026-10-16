#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Station-level cluster bootstrap for the corrected CCM family.

Combos sharing a gauging station (gsim_id) or a water-quality station
(gems_id) are pseudo-replicated, so combo-level binomial p-values overstate
the effective sample size. Station-cluster bootstrap: resample stations with
replacement (B = 2000, seed 42), keep every combo of the sampled stations,
report the percentile CI and a null-centred two-sided bootstrap p with the
(b + 1) / (B + 1) convention.

The TE / conditional-TE cluster intervals live in strict_layer.py's
cluster_robust_strict.csv (they share this design); this script owns the
standard-CCM, deseasonalized-CCM and M-CCM intervals.

Writes data/results/tables/cluster_robust.csv and
       data/results/tables/cluster_robust_mccm_fixed.csv.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"

B = 2000
SEED = 42
LIB_HEADLINE = 0.7
CLUSTER_KEYS = ["gsim_id", "gems_id"]


def load(df_file, fwd, bwd, dropna=False):
    df = pd.read_csv(f"{T}/{df_file}")
    if "lib" in df.columns:
        df = df[df["lib"] == LIB_HEADLINE].copy()
    if dropna:
        df = df.dropna(subset=[fwd, bwd]).copy()
    df["dom"] = (df[fwd] > df[bwd]).astype(int)   # ties are not flow-dominant
    return df


def bootstrap_rows(name, df):
    rows = []
    dom = df["dom"].to_numpy()
    for key in CLUSTER_KEYS:
        codes, clusters = np.unique(df[key].astype(str).to_numpy(),
                                    return_inverse=True)
        k = len(codes)
        obs = dom.mean()

        rng = np.random.default_rng(SEED)     # fresh stream per method x key
        sums = np.bincount(clusters, weights=dom, minlength=k)
        cnts = np.bincount(clusters, minlength=k)
        idx = rng.integers(0, k, size=(B, k))
        boot = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)

        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        boot_h0 = boot - obs + 0.5
        dev = np.abs(boot_h0 - 0.5) >= abs(obs - 0.5)
        p_two = (dev.sum() + 1) / (B + 1)

        rows.append({"method": name, "cluster_key": key,
                     "n_combos": int(len(df)), "n_clusters": int(k),
                     "obs_share": round(float(obs), 4),
                     "ci_low": round(float(ci_low), 4),
                     "ci_high": round(float(ci_high), 4),
                     "p_two_sided": round(float(p_two), 4)})
    return rows


def main():
    methods = {
        "std_ccm": load("std_ccm.csv", "f2w", "w2f"),
        "deseason": load("deseason.csv", "f2w", "w2f"),
    }
    mccm = load("mccm_fixed.csv", "mccm_f2w", "mccm_w2f", dropna=True)

    anchors = {"std_ccm": 51.9, "deseason": 46.8, "mccm": 85.9}
    for name, df in list(methods.items()) + [("mccm", mccm)]:
        obs = 100.0 * df["dom"].mean()
        flag = "OK" if abs(obs - anchors[name]) < 0.05 else "MISMATCH"
        print(f"  {name:9s}  n={len(df):3d}  share={obs:5.1f}%  "
              f"anchor={anchors[name]:5.1f}%  [{flag}]")

    rows = []
    for name, df in methods.items():
        rows += bootstrap_rows(name, df)
    res = pd.DataFrame(rows)
    res.to_csv(f"{T}/cluster_robust.csv", index=False)
    print()
    print(res.to_string(index=False))

    mc_rows = bootstrap_rows("mccm", mccm)
    mc = pd.DataFrame(mc_rows)[["cluster_key", "n_clusters", "obs_share",
                                "ci_low", "ci_high", "p_two_sided"]]
    mc.to_csv(f"{T}/cluster_robust_mccm_fixed.csv", index=False)
    print()
    print(mc.to_string(index=False))
    print(f"\nwritten: cluster_robust.csv, cluster_robust_mccm_fixed.csv")


if __name__ == "__main__":
    main()
