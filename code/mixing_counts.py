#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Species-mixing screen: the two counts quoted in the Introduction.

"32% of station-months" and "65 of 172 nitrogen stations" -- recomputed from
the naive WQ archive layer. Needs the external GEMStat archive under
data/raw/ to rerun; the shipped table is the frozen output.

Writes data/results/tables/mixing_counts.csv.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"

TOTAL_CODES = {"TN", "TOTN"}
FRACTIONAL_CODES = {"NO3+NO2", "NO3N", "NO2N", "NH3N", "NH4N"}


def main():
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv")
    tn = res[(res.parameter == "TN") & (res.n_months >= 60)].reset_index(drop=True)
    gids = set(tn.gems_id)

    raw = pd.read_csv(f"{ROOT}/data/raw/global/gemstat/extracted/Other_Nitrogen.csv",
                      encoding="latin-1", low_memory=False)
    raw = raw[raw["GEMS Station Number"].isin(gids)].copy()
    raw["date"] = pd.to_datetime(raw["Sample Date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw["ym"] = raw.date.dt.to_period("M")
    raw["value"] = pd.to_numeric(raw["Value"], errors="coerce")
    raw.loc[raw["Value Flags"] == "<", "value"] = raw.loc[raw["Value Flags"] == "<", "value"] / 2
    raw = raw[(raw.value >= 0) & (raw.value <= 100)]

    n_stations = len(gids)
    n_mixed = 0
    n_months = 0
    n_multi = 0
    for gid in sorted(gids):
        sub = raw[raw["GEMS Station Number"] == gid]
        if len(sub) == 0:
            continue
        codes = set(sub["Parameter Code"])
        if (codes & TOTAL_CODES) and (codes & FRACTIONAL_CODES):
            n_mixed += 1
        per = sub.groupby("ym")["Parameter Code"].nunique()
        n_months += len(per)
        n_multi += int((per > 1).sum())

    out = pd.DataFrame([
        {"quantity": "TN stations entering the mixing screen", "value": n_stations},
        {"quantity": "stations mixing total and fractional N codes", "value": n_mixed},
        {"quantity": "station-months with N records", "value": n_months},
        {"quantity": "station-months with >1 N code co-occurring", "value": n_multi},
        {"quantity": "share of station-months with co-occurring N codes",
         "value": round(n_multi / n_months, 4)},
    ])
    out.to_csv(f"{T}/mixing_counts.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
