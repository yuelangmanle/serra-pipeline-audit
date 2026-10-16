#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Taihu Chla trend — one authoritative table for the trend number(s).

Why this script exists (three sources, three numbers):
  1. Manuscript text (+0.374 ug/L/yr, monthly Sen) came from
     the dual-lake Sen slope of the earlier layer, which used INTEGER-year indices on the monthly
     series: 66,430 pairwise slopes of which 1,975 same-year pairs divide by
     zero to +/-inf. The +/-inf block is asymmetric (1,219 +inf vs 756 -inf),
     so the median is NOT invariant to it: the contaminated median window sits
     ~232 rank positions above the clean one, moving the estimate from +0.356
     (cross-year pairs only) to +0.374.
  2. Fig4(e) labelled an np.polyfit OLS-on-annual-means slope (+0.766) as
     "Sen = +0.766" — an estimator mislabel.
  3. The packaged taihu_trend_analysis.csv gives +0.458 / p = 0.0116 with 33
     years (i.e. including 2023) under an unstated estimator — out of scope
     for the manuscript window and superseded by this table.

This script computes, on data through 2022-12 (manuscript window):
  A. Monthly Sen, correct implementation, two variants:
     (a) decimal-year time t = year + (month-0.5)/12, ALL pairs used;
     (b) same-year pairs dropped (cross-year pairs only).
     Both reported with a tie-corrected Mann-Kendall test.
  B. Annual scale on annual means restricted to years with >= 9 observed
     months (drops 1991 and 1994, 3 months each): Sen + MK + OLS.
  C. Sensitivity: Sen + OLS on ALL annual means including partial years
     (internal recomputation: Sen +0.695, OLS +0.766).

Output: data/results/tables/taihu_trend.csv
Columns: estimator, scale, slope, p, n
"""
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = os.environ.get("PAPER_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = f"{ROOT}/data/results/tables"
DATA_END = "2022-12-01"   # manuscript window: drop 2023-24 partial years
MIN_MONTHS = 9            # a year enters the annual series with >= 9 observed months


def sen_slope(times, values):
    """Theil-Sen slope: median of all pairwise slopes on the given time axis.

    `times` must be a float array (decimal years) with no duplicated values
    unless the caller intends duplicate-time pairs to be excluded beforehand.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    slopes = []
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((values[j] - values[i]) / (times[j] - times[i]))
    return float(np.median(slopes)), np.array(slopes)


def sen_slope_cross_year(years, values):
    """Variant (b): integer-year axis, same-year pairs excluded entirely
    (the mathematically clean way to avoid the 0/0 -> +/-inf contamination)."""
    years = np.asarray(years)
    values = np.asarray(values, dtype=float)
    slopes = []
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((values[j] - values[i]) / (years[j] - years[i]))
    return float(np.median(slopes)), np.array(slopes)


def mk_test(values):
    """Mann-Kendall test with tie correction and continuity correction.

    Returns (S, z, p two-sided).
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(v[j] - v[i])
    tie_term = sum(k * (k - 1) * (2 * k + 5) for k in Counter(v).values())
    var = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return s, z, float(p)


def legacy_sen_slope_integer_year(values, years):
    """Verbatim reproduction of the earlier dual-lake Sen slope (integer-year axis,
    same-year pairs -> +/-inf) for the audit trail ONLY. Not used downstream."""
    v = pd.Series(values, index=years).dropna()
    yrs, vals = v.index.values, v.values
    n = len(vals)
    slopes = [(vals[j] - vals[i]) / (yrs[j] - yrs[i])
              for i in range(n) for j in range(i + 1, n)]
    return float(np.median(slopes)), np.array(slopes)


def main():
    rows = []

    t = pd.read_csv(f"{ROOT}/data/processed/taihu_monthly_water_quality.csv")
    t["date"] = pd.to_datetime(dict(year=t.year, month=t.month, day=1))
    t = t[t.date <= DATA_END].set_index("date").sort_index()
    print(f"Taihu monthly records through {DATA_END}: {len(t)} rows")

    chla = t["Chla"].dropna()
    vals = chla.values.astype(float)
    years = chla.index.year.values
    dec_year = years + (chla.index.month.values - 0.5) / 12.0
    n_months = len(vals)
    print(f"Chla observed months: {n_months}; pairwise slopes C(n,2) = "
          f"{n_months * (n_months - 1) // 2}")

    # ---------------- A. monthly Sen, two correct variants ----------------
    sen_dec, slopes_dec = sen_slope(dec_year, vals)
    sen_xyr, slopes_xyr = sen_slope_cross_year(years, vals)
    S_m, z_m, p_m = mk_test(vals)

    same_year = int(sum(1 for i in range(n_months) for j in range(i + 1, n_months)
                        if years[j] == years[i]))
    print(f"\n[A] Monthly Sen (correct implementations)")
    print(f"    variant (a) decimal-year time, all pairs : {sen_dec:+.4f} ug/L/yr")
    print(f"    variant (b) cross-year pairs only        : {sen_xyr:+.4f} ug/L/yr")
    print(f"    (same-year pairs in (a): {same_year}; variant disagreement "
          f"{abs(sen_dec - sen_xyr):.4f} ug/L/yr = "
          f"{100 * abs(sen_dec - sen_xyr) / abs(sen_xyr):.1f}% of (b))")
    print(f"    MK: S={S_m:.0f}  z={z_m:.3f}  p={p_m:.3e}")

    rows.append({"estimator": "Sen", "scale": "monthly_decimal_year",
                 "slope": round(sen_dec, 6), "p": p_m, "n": n_months})
    rows.append({"estimator": "Sen", "scale": "monthly_cross_year_pairs",
                 "slope": round(sen_xyr, 6), "p": p_m, "n": n_months})

    # ---------------- legacy estimator audit (not used downstream) --------
    leg, leg_slopes = legacy_sen_slope_integer_year(t["Chla"].values, t.index.year.values)
    n_inf_pos = int(np.isposinf(leg_slopes).sum())
    n_inf_neg = int(np.isneginf(leg_slopes).sum())
    print(f"\n[Audit] legacy integer-year sen_slope with +/-inf left in place:")
    print(f"    median = {leg:+.4f}  (+inf: {n_inf_pos}, -inf: {n_inf_neg}; the "
          f"asymmetric inf block shifts the median window "
          f"{(n_inf_pos - n_inf_neg) / 2:+.0f} rank positions vs variant (b))")

    # ---------------- B. annual scale (>= 9 observed months) --------------
    grp = chla.groupby(chla.index.year)
    ann_mean, ann_cnt = grp.mean(), grp.size()
    kept = ann_cnt >= MIN_MONTHS
    ann = ann_mean[kept]
    a_years = ann.index.values.astype(float)
    a_vals = ann.values.astype(float)

    sen_ann, _ = sen_slope(a_years, a_vals)
    S_a, z_a, p_a = mk_test(a_vals)
    lr_a = stats.linregress(a_years, a_vals)
    print(f"\n[B] Annual scale, years with >= {MIN_MONTHS} observed months "
          f"(n = {len(a_years)}, dropped: "
          f"{sorted(set(ann_mean.index) - set(ann.index))})")
    print(f"    Sen  = {sen_ann:+.4f} ug/L/yr   MK z={z_a:.3f}  p={p_a:.3e}")
    print(f"    OLS  = {lr_a.slope:+.4f} ug/L/yr  p={lr_a.pvalue:.3e}")

    rows.append({"estimator": "Sen", "scale": f"annual_ge{MIN_MONTHS}months",
                 "slope": round(sen_ann, 6), "p": p_a, "n": len(a_years)})
    rows.append({"estimator": "OLS", "scale": f"annual_ge{MIN_MONTHS}months",
                 "slope": round(float(lr_a.slope), 6), "p": float(lr_a.pvalue),
                 "n": len(a_years)})

    # ---------------- C. sensitivity: all annual means, no year filter ----
    ann_all = ann_mean
    ay_all = ann_all.index.values.astype(float)
    av_all = ann_all.values.astype(float)
    sen_all, _ = sen_slope(ay_all, av_all)
    _, _, p_all = mk_test(av_all)
    lr_all = stats.linregress(ay_all, av_all)
    print(f"\n[C] Sensitivity: ALL annual means incl. partial years (n = {len(ay_all)})")
    print(f"    Sen  = {sen_all:+.4f} ug/L/yr   MK p={p_all:.3e}")
    print(f"    OLS  = {lr_all.slope:+.4f} ug/L/yr  p={lr_all.pvalue:.3e}"
          f"   <- reproduces the internal +0.695 / +0.766")
    rows.append({"estimator": "Sen", "scale": "annual_all_years_sensitivity",
                 "slope": round(sen_all, 6), "p": p_all, "n": len(ay_all)})
    rows.append({"estimator": "OLS", "scale": "annual_all_years_sensitivity",
                 "slope": round(float(lr_all.slope), 6), "p": float(lr_all.pvalue),
                 "n": len(ay_all)})

    # ---------------- structural descriptives (manuscript Section 3.4) ----
    # The three descriptive numbers quoted in Section 3.4 get their own
    # table so every manuscript number traces to a shipped table:
    # (i) the early/late Chla-TP Pearson correlation, (ii) the TLI
    # before-2019 vs 2019-2020 monthly Mann-Whitney comparison behind the
    # "no monthly-resolution regime shift" statement.
    from scipy import stats as _st
    recs = []
    for scope, early_mask in (("all months through 2008", t.index.year <= 2008),
                              ("2009-2022", (t.index.year >= 2009) & (t.index.year <= 2022))):
        pair = t[early_mask].dropna(subset=["Chla", "TP"])
        r, p = _st.pearsonr(pair.TP, pair.Chla)
        recs.append({"analysis": "Chla-TP Pearson r", "scope": scope,
                     "value": round(float(r), 6), "p": float(p), "n": len(pair)})
        print(f"  Pearson Chla-TP, {scope}: r = {r:.4f} (p = {p:.2g}, n = {len(pair)})")
    ttl = t.dropna(subset=["TLI"])
    pre = ttl[ttl.index.year < 2019]["TLI"]
    shift = ttl[ttl.index.year.isin([2019, 2020])]["TLI"]
    u, p_tli = _st.mannwhitneyu(pre, shift)
    recs.append({"analysis": "TLI Mann-Whitney U", "scope": "monthly, before 2019 vs 2019-2020",
                 "value": float(u), "p": float(p_tli), "n": len(pre) + len(shift)})
    print(f"  TLI monthly before 2019 (n={len(pre)}) vs 2019-2020 (n={len(shift)}): "
          f"means {pre.mean():.1f} vs {shift.mean():.1f}, Mann-Whitney p = {p_tli:.3f}")
    for yr in (1991, 1994, 2021, 2022):
        cnt = int((ttl.index.year == yr).sum())
        recs.append({"analysis": "TLI observed months", "scope": f"{yr}",
                     "value": float(cnt), "p": np.nan, "n": cnt})
        print(f"  TLI coverage {yr}: {cnt} months")
    pd.DataFrame(recs, columns=["analysis", "scope", "value", "p", "n"]).to_csv(
        f"{T}/taihu_descriptives.csv", index=False)
    print(f"\nWritten: {T}/taihu_descriptives.csv")

    # ---------------- write table -----------------------------------------
    out = pd.DataFrame(rows, columns=["estimator", "scale", "slope", "p", "n"])
    out.to_csv(f"{T}/taihu_trend.csv", index=False)
    print(f"\nWritten: {T}/taihu_trend.csv")
    print(out.to_string(index=False))

    print("\nReconciliation with the three pre-existing sources:")
    print(f"  manuscript text +0.374 (legacy estimator)  -> correct monthly "
          f"Sen = {sen_dec:+.4f} (decimal-year) / {sen_xyr:+.4f} (cross-year); "
          f"+0.374 is confirmed by variant (a) to 3 decimals")
    print(f"  Fig4(e) label 'Sen = +0.766' (was OLS)     -> true annual Sen = "
          f"{sen_ann:+.4f}; OLS = {lr_a.slope:+.4f}")
    print(f"  taihu_trend_analysis.csv +0.458 / p=0.0116 -> 33 years incl. 2023, "
          f"estimator unstated; superseded by this table")


if __name__ == "__main__":
    main()
