#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpolation footprint and half-detection-limit share.

Interpolation share for the Methods section, from calendar_scan.csv (the
calendar-gap diagnostic: strict3 / gap1_3 / gap2_3 are E=3 embedding-window
counts on the strict-observed, gap<=1-filled and gap<=2-filled masks)
joined to combos.csv (n_valid = months entering the analysis after gap<=2
filling). The scan's column semantics are re-verified from scratch
(merged_calendar + fill_short_gaps_pair + valid_targets) on sample combos.

Half-detection-limit share in the global GEMStat input, replicating
build_wq_total: rows with Value Flags == '<' enter as value/2. Reported at
three stages: rows passing the range filter, rows surviving the 3xIQR
filter, and station-months whose monthly mean involved >=1 half-RL row.
The half-RL stage needs the external GEMStat archive under data/raw/.

Writes data/results/tables/interp_and_halfrl.csv.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from pipeline import (read_gsim, merged_calendar,  # noqa: E402
                      fill_short_gaps_pair, valid_targets, build_wq_total)

VERIFY_COMBOS = [("CAN00018", "CA_0003930", "TP"),
                 ("CAN00027", "CA_0004406", "TP"),
                 ("CAN00023", "CA_0005241", "TN")]


def window_count(mask, E=3):
    """Number of E-lag embedding targets on a validity mask (valid_targets)."""
    return len(valid_targets(mask.astype(bool), E, 1, 0))


def task5_interp():
    cs = pd.read_csv(f"{T}/calendar_scan.csv")
    cb = pd.read_csv(f"{T}/combos.csv")
    m = cb.merge(cs, on=["gems_id", "parameter"], how="left")

    csk = set(zip(cs.gems_id, cs.parameter))
    cbk = set(zip(cb.gems_id, cb.parameter))
    orphan_scan = sorted(csk - cbk)
    missing_scan = sorted(cbk - csk)

    # any combo missing from the scan gets its counts recomputed from raw
    if missing_scan:
        qual0 = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv")
        wq_fill, _ = build_wq_total(set(qual0.gems_id))
        rec_rows = []
        for gid, param in missing_scan:
            sid = cb.loc[(cb.gems_id == gid) & (cb.parameter == param),
                         "gsim_id"].iloc[0]
            flow = read_gsim(sid)
            series = wq_fill.get(gid, {}).get(param, {})
            x, y, valid_obs, cal = merged_calendar(flow, series)
            _, _, v1 = fill_short_gaps_pair(x, y, valid_obs.copy(), max_gap=1)
            _, _, v2 = fill_short_gaps_pair(x, y, valid_obs.copy(), max_gap=2)
            rec_rows.append({"gems_id": gid, "parameter": param,
                             "n_cal": len(cal), "n_obs": int(valid_obs.sum()),
                             "strict3": window_count(valid_obs),
                             "gap1_3": window_count(v1),
                             "gap2_3": window_count(v2)})
        m = m.merge(pd.DataFrame(rec_rows), on=["gems_id", "parameter"],
                    how="left", suffixes=("", "_rec"))
        for c in ["n_cal", "n_obs", "strict3", "gap1_3", "gap2_3"]:
            m[c] = m[c].fillna(m[f"{c}_rec"])
        m = m.drop(columns=[f"{c}_rec" for c in
                            ["n_cal", "n_obs", "strict3", "gap1_3", "gap2_3"]])
    assert m[["strict3", "gap1_3", "gap2_3", "n_obs"]].notna().all().all()
    m["n_interp"] = m["n_valid"] - m["n_obs"]

    n_combos = len(m)
    n_combos_any_interp = int((m.n_interp > 0).sum())
    n_combos_window_interp = int((m.strict3 < m.gap2_3).sum())
    tot_interp = int(m.n_interp.sum())
    tot_obs = int(m.n_obs.sum())
    tot_valid = int(m.n_valid.sum())

    # ---- verify calendar_scan semantics from raw data on sample combos
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv")
    wq_small, _ = build_wq_total(set(qual.gems_id))
    checks = []
    for gid, sid, param in VERIFY_COMBOS:
        flow = read_gsim(sid)
        series = wq_small.get(gid, {}).get(param, {})
        x, y, valid_obs, cal = merged_calendar(flow, series)
        strict_n = window_count(valid_obs)
        _, _, v1 = fill_short_gaps_pair(x, y, valid_obs.copy(), max_gap=1)
        _, _, v2 = fill_short_gaps_pair(x, y, valid_obs.copy(), max_gap=2)
        row = cs[(cs.gems_id == gid) & (cs.parameter == param)].iloc[0]
        checks.append({
            "gems_id": gid, "gsim_id": sid, "parameter": param,
            "n_cal_recompute": len(cal), "n_cal_table": int(row.n_cal),
            "n_obs_recompute": int(valid_obs.sum()), "n_obs_table": int(row.n_obs),
            "strict3_recompute": strict_n, "strict3_table": int(row.strict3),
            "gap1_3_recompute": window_count(v1), "gap1_3_table": int(row.gap1_3),
            "gap2_3_recompute": window_count(v2), "gap2_3_table": int(row.gap2_3),
            "n_valid_table": int(cb[(cb.gems_id == gid) & (cb.parameter == param)].n_valid.iloc[0]),
            "n_valid_recompute": int(v2.sum()),
        })
    chk = pd.DataFrame(checks)
    ok = all((chk[f"{c}_recompute"] == chk[f"{c}_table"]).all()
             for c in ["n_cal", "n_obs", "strict3", "gap1_3", "gap2_3", "n_valid"])
    print(chk.to_string(index=False))
    print(f"  calendar_scan semantics verified from raw data on {len(chk)} combos: {ok}")

    rows = [
        {"section": "interp", "metric": "n_combos", "value": n_combos},
        {"section": "interp", "metric": "n_combos_with_interp_month",
         "value": n_combos_any_interp},
        {"section": "interp", "metric": "pct_combos_with_interp_month",
         "value": 100 * n_combos_any_interp / n_combos},
        {"section": "interp", "metric": "n_combos_interp_in_embedding_window",
         "value": n_combos_window_interp},
        {"section": "interp", "metric": "pct_combos_interp_in_embedding_window",
         "value": 100 * n_combos_window_interp / n_combos},
        {"section": "interp", "metric": "n_interp_months_total", "value": tot_interp},
        {"section": "interp", "metric": "n_obs_months_total", "value": tot_obs},
        {"section": "interp", "metric": "n_analysed_months_total", "value": tot_valid},
        {"section": "interp", "metric": "pct_interp_of_observed",
         "value": 100 * tot_interp / tot_obs},
        {"section": "interp", "metric": "pct_interp_of_analysed",
         "value": 100 * tot_interp / tot_valid},
        {"section": "interp", "metric": "calendar_scan_semantics_verified",
         "value": float(ok)},
        {"section": "interp", "metric": "calendar_scan_orphan_keys",
         "value": ";".join(f"{a}|{b}" for a, b in orphan_scan)},
        {"section": "interp", "metric": "combos_recomputed_missing_scan",
         "value": ";".join(f"{a}|{b}" for a, b in missing_scan)},
    ]
    return rows


def task6_halfrl():
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations_v2.csv")
    qids = set(qual.gems_id)
    rows = []
    for param, fn, codes, hi in [
        ("TN", "Other_Nitrogen.csv", ["TN", "TOTN"], 100.0),
        ("TP", "Phosphorus.csv", ["TP", "TOTP"], 10.0),
    ]:
        df = pd.read_csv(f"{ROOT}/data/raw/global/gemstat/extracted/{fn}",
                         encoding="latin-1", low_memory=False)
        df = df[df["GEMS Station Number"].isin(qids) & df["Parameter Code"].isin(codes)]
        df["date"] = pd.to_datetime(df["Sample Date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df["ym"] = df.date.dt.to_period("M")
        df["raw_value"] = pd.to_numeric(df["Value"], errors="coerce")
        df["flag_lt"] = (df["Value Flags"].fillna("") == "<")
        # build_wq_total: value = raw/2 where flag '<'; then range filter
        df["value"] = df["raw_value"] / 2
        in_range = df[(df.value >= 0) & (df.value <= hi)].copy()

        def iqr_ok(g):
            med = g.median()
            q1, q3 = g.quantile(0.25), g.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                return pd.Series(True, index=g.index)
            return (g >= med - 3 * iqr) & (g <= med + 3 * iqr)

        kept = in_range[in_range.groupby("GEMS Station Number")["value"]
                        .transform(iqr_ok).fillna(False)].copy()
        kept["is_flag"] = kept["flag_lt"]
        g = kept.groupby(["GEMS Station Number", "ym"]).agg(
            n=("is_flag", "size"), n_flag=("is_flag", "sum"))
        months_with_flag = g[(g.n_flag > 0)]
        rows += [
            {"section": "halfrl", "metric": f"{param}_rows_in_range",
             "value": len(in_range)},
            {"section": "halfrl", "metric": f"{param}_rows_flag_lt",
             "value": int(in_range.flag_lt.sum())},
            {"section": "halfrl", "metric": f"{param}_pct_rows_flag_lt_in_range",
             "value": 100 * in_range.flag_lt.mean()},
            {"section": "halfrl", "metric": f"{param}_rows_after_iqr",
             "value": len(kept)},
            {"section": "halfrl", "metric": f"{param}_pct_rows_flag_lt_after_iqr",
             "value": 100 * kept.is_flag.mean()},
            {"section": "halfrl", "metric": f"{param}_station_months", "value": len(g)},
            {"section": "halfrl",
             "metric": f"{param}_pct_station_months_with_halfrl",
             "value": 100 * len(months_with_flag) / len(g)},
        ]
        print(f"  {param}: rows in range={len(in_range)}, half-RL rows="
              f"{int(in_range.flag_lt.sum())} ({100*in_range.flag_lt.mean():.2f}%)")
        print(f"       after 3xIQR filter: {len(kept)} rows, half-RL "
              f"{100*kept.is_flag.mean():.2f}%; station-months with >=1 half-RL "
              f"value: {len(months_with_flag)}/{len(g)} "
              f"({100*len(months_with_flag)/len(g):.2f}%)")
    return rows


def main():
    rows = task5_interp() + task6_halfrl()
    out = pd.DataFrame(rows)
    out.to_csv(f"{T}/interp_and_halfrl.csv", index=False)
    print(f"-> {T}/interp_and_halfrl.csv")


if __name__ == "__main__":
    main()
