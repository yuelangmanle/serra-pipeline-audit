#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gauge-to-sampling separation of the analysed combinations.

The manuscript's data-quality sentence (median, 90th percentile, maximum
separation, pairs beyond 25 km, pairs crossing a national border) gets its
own record: data/results/tables/site_geometry.csv, computed on the final
316-combination sample joined to the nearest-neighbour matching table
(data/interim/matched_river_stations_50km_10yr.csv; archives needed).

Percentiles are numpy's default (linear interpolation). The border count
compares the GEMS station's country with the country of the matched GSIM
station's identifier prefix.

Run with datasets in place: PAPER_BASE=/path python3 code/site_geometry.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"

GSIM_PREFIX = {"CAN": "CA", "IND": "IN", "NOR": "NO", "SVN": "SI", "USA": "US"}


def main():
    combos = pd.read_csv(f"{T}/combos.csv")
    combos = combos[combos.n_valid >= 60]
    match = pd.read_csv(f"{ROOT}/data/interim/matched_river_stations_50km_10yr.csv")
    match = match.drop_duplicates("gems_id")
    d = combos.merge(match[["gems_id", "distance_km"]], on="gems_id", how="left")
    missing = int(d.distance_km.isna().sum())
    dist = d.distance_km.dropna()
    cross = sum(
        1 for gid, sid in zip(d.gems_id, d.gsim_id)
        if sid[:2] != GSIM_PREFIX.get(gid[:3], gid[:3]))
    rows = [
        {"metric": "combinations", "value": int(len(d)),
         "note": "n_valid >= 60; distance joined from the matching table"},
        {"metric": "distance_missing", "value": missing,
         "note": "combinations without a matching-table row"},
        {"metric": "separation_median_km", "value": round(float(np.median(dist)), 4),
         "note": "median gauge-to-sampling separation"},
        {"metric": "separation_p90_km", "value": round(float(np.percentile(dist, 90)), 4),
         "note": "90th percentile (linear interpolation)"},
        {"metric": "separation_max_km", "value": round(float(dist.max()), 4),
         "note": "maximum"},
        {"metric": "pairs_beyond_25km", "value": int((dist > 25).sum()),
         "note": "combinations with separation above 25 km"},
        {"metric": "pairs_crossing_border", "value": int(cross),
         "note": "GEMS country differs from the matched GSIM station prefix"},
    ]
    cand = match  # unique paired stations (deduplicated above)
    rows.append({"metric": "candidate_pairs",
                 "value": int(len(cand)),
                 "note": "unique paired stations in the matching table (before the calendar/overlap gates)"})
    rows.append({"metric": "candidate_separation_median_km",
                 "value": round(float(np.median(cand.distance_km.dropna())), 4),
                 "note": "median gauge-to-sampling separation of the candidate pairs"})
    out = pd.DataFrame(rows, columns=["metric", "value", "note"])
    out.to_csv(f"{T}/site_geometry.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwritten: {T}/site_geometry.csv")


if __name__ == "__main__":
    main()
