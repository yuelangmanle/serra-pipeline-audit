#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geographic stratification and leave-one-country-out robustness.

The 316 corrected combos come from five countries by GEMStat station prefix
(USA 99 / IND 86 / CAN 83 / NOR 32 / SVN 16), so coverage is much narrower
than "global". This script reports, for the four corrected methods, the
full-sample share, each country stratum, and each leave-one-out scope, with
a two-sided exact binomial p against the 50% line. The per-country TE
column is the table behind the manuscript's Section 4.5 numbers.

Writes data/results/tables/geo_robustness.csv.
"""
import os
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"

LIB_HEADLINE = 0.7

METHODS = {
    "std_ccm":  ("std_ccm.csv", "f2w", "w2f", False),
    "deseason": ("deseason.csv", "f2w", "w2f", False),
    "te":       ("te_z.csv", "te_f2w", "te_w2f", False),
    "mccm":     ("mccm_fixed.csv", "mccm_f2w", "mccm_w2f", True),
}

ANCHORS = {"std_ccm": 51.9, "deseason": 46.8, "te": 45.6, "mccm": 85.9}

COUNTRY_MAP = {
    "USA": "United States",
    "IND": "India",
    "CAN": "Canada",
    "NOR": "Norway",
    "SVN": "Slovenia",
}


def main():
    combos = pd.read_csv(f"{T}/combos.csv")
    combos["country"] = combos["gems_id"].str.slice(0, 3).map(COUNTRY_MAP)

    methods = {}
    for name, (fname, fwd, bwd, dropna) in METHODS.items():
        df = pd.read_csv(f"{T}/{fname}")
        if "lib" in df.columns:
            df = df[df["lib"] == LIB_HEADLINE].copy()
        if dropna:
            df = df.dropna(subset=[fwd, bwd]).copy()
        df["dom"] = (df[fwd] > df[bwd]).astype(int)   # ties are not dominant
        df = df.merge(combos[["gems_id", "gsim_id", "parameter", "country"]],
                      on=["gems_id", "gsim_id", "parameter"], how="left")
        assert df["country"].notna().all(), f"unmapped country in {name}"
        methods[name] = df

    for name, df in methods.items():
        obs = 100.0 * df["dom"].mean()
        flag = "OK" if abs(obs - ANCHORS[name]) < 0.05 else "MISMATCH"
        print(f"  {name:9s}  n={len(df):3d}  share={obs:5.1f}%  "
              f"anchor={ANCHORS[name]:5.1f}%  [{flag}]")

    rows = []
    for name, df in methods.items():
        def emit(scope, sub):
            n = len(sub)
            k = int(sub["dom"].sum())
            p = binomtest(k, n, 0.5).pvalue if n > 0 else float("nan")
            rows.append({"method": name, "scope": scope, "n": int(n),
                         "flow_dom_share": round(k / n, 4) if n else float("nan"),
                         "p_two_sided": round(float(p), 4) if n else float("nan")})
        # full sample, every country stratum, every leave-one-out scope
        emit("full", df)
        for ctry in COUNTRY_MAP.values():
            emit(ctry, df[df["country"] == ctry])
        for ctry in COUNTRY_MAP.values():
            emit(f"leave_out_{ctry.replace(' ', '_')}", df[df["country"] != ctry])

    res = pd.DataFrame(rows)
    res.to_csv(f"{T}/geo_robustness.csv", index=False)
    with pd.option_context("display.width", 120):
        print(res.to_string(index=False))
    print(f"\nwritten: {T}/geo_robustness.csv")


if __name__ == "__main__":
    main()
