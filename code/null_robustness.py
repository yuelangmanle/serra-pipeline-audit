#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness records for the eleven-shift nulls (manuscript Section 2.7).

Three checks, one row-group each, written to
data/results/tables/null_robustness.csv:

* identity_exclusion - the family null shares recomputed with the identity
  draws (k a multiple of the complete-year count) removed, against the
  shipped aggregates; documents that the identity draws stay in the
  aggregates by design and that removing them changes no verdict.
* strata - observed share against its own null range, splitting the sample
  into records with more than eleven complete years and records with fewer.
* seed_stability (needs the archives) - the corrected standard and
  deseasonalized CCM observed shares under three library-split seeds,
  giving the share's sensitivity to the random 70/30 split.

Strata and identity-exclusion rows need only the shipped tables; the seed
block additionally needs data/raw and data/interim via PAPER_BASE.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"
sys.path.insert(0, HERE)

KS = list(range(1, 12))


def load_years():
    combos = pd.read_csv(f"{T}/combos.csv")
    combos["y"] = combos.n_calendar // 12
    return combos.set_index(["gems_id", "gsim_id", "parameter"])["y"].to_dict()


def aug(df, yk):
    df = df.copy()
    df["y"] = [yk.get(t, 0) for t in
               df[["gems_id", "gsim_id", "parameter"]].itertuples(
                   index=False, name=None)]
    return df


def main():
    t0 = time.time()
    yk = load_years()
    rows = []

    def add(check, family, scope, vm, vmin, vmax, n, n_unit):
        rows.append({"check": check, "family": family, "scope": scope,
                     "value_mean": vm, "value_min": vmin, "value_max": vmax,
                     "n": n, "n_unit": n_unit})

    # ---------------- identity-draw exclusion -----------------------------
    n = aug(pd.read_csv(f"{T}/strict_nulls.csv"), yk)
    m = aug(pd.read_csv(f"{T}/strict_months_null.csv"), yk)
    n["isid"] = (n.y > 0) & ((n.k % n.y) == 0)
    m["isid"] = (m.y > 0) & ((m.k % m.y) == 0)

    def fam_shares(df, f, w=None):
        if w is None:                      # vote table
            return np.array([100 * df[df.k == k].vote.mean() for k in KS])
        return np.array([100 * (df[df.k == k][f] > df[df.k == k][w]).mean()
                         for k in KS])

    for fam, df, f, w in (("TE", n, "te_f", "te_w"), ("CTE", n, "cte_f", "cte_w"),
                          ("strict_months", m, None, None)):
        full = fam_shares(df, f, w)
        excl = fam_shares(df[~df.isid], f, w)
        add("identity_exclusion", fam, "full",
            round(float(full.mean()), 4), round(float(full.min()), 4),
            round(float(full.max()), 4), len(df), "draws")
        add("identity_exclusion", fam, "identity draws removed",
            round(float(excl.mean()), 4), round(float(excl.min()), 4),
            round(float(excl.max()), 4), int((~df.isid).sum()), "draws")
    c = aug(pd.read_csv(f"{T}/ccm_shifts_corrected.csv"), yk)
    for fam, f, w in (("std_ccm", "std_f", "std_w"), ("des_ccm", "des_f", "des_w")):
        full, excl = [], []
        for k in KS:
            pair = c[[f"{f}_{k}", f"{w}_{k}"]].dropna()
            full.append(100 * (pair.iloc[:, 0] > pair.iloc[:, 1]).mean())
            keep = np.array([not (y > 0 and k % y == 0) for y in c.y])
            pair2 = c.loc[keep, [f"{f}_{k}", f"{w}_{k}"]].dropna()
            excl.append(100 * (pair2.iloc[:, 0] > pair2.iloc[:, 1]).mean())
        for scope, sh in (("full", full), ("identity draws removed", excl)):
            add("identity_exclusion", fam, scope,
                round(float(np.mean(sh)), 4), round(float(np.min(sh)), 4),
                round(float(np.max(sh)), 4), len(c), "combinations")

    # ---------------- strata: y > 11 vs y <= 11 ---------------------------
    te_z = aug(pd.read_csv(f"{T}/te_z.csv"), yk)

    for fam, df, f, w in (("TE", n, "te_f", "te_w"), ("CTE", n, "cte_f", "cte_w")):
        for label, sub_o, sub_n in (
                (">11 complete years", te_z[te_z.y > 11], n[n.y > 11]),
                ("<=11 complete years", te_z[te_z.y <= 11], n[n.y <= 11])):
            obs = 100 * ((sub_o.te_f2w > sub_o.te_w2f).mean() if fam == "TE"
                         else (sub_o.cte_f2w > sub_o.cte_w2f).mean())
            sh = np.array([100 * (sub_n[sub_n.k == k][f] > sub_n[sub_n.k == k][w]).mean()
                           for k in KS])
            add("strata", fam, label, round(obs, 4),
                round(float(sh.min()), 4), round(float(sh.max()), 4),
                len(sub_o), "combinations")
    for label, sub in ((">11 complete years", m[m.y > 11]), ("<=11 complete years", m[m.y <= 11])):
        obs = 100 * sub[sub.k == 0].vote.mean()
        sh = np.array([100 * sub[sub.k == k].vote.mean() for k in KS])
        n_combos = sub[["gems_id", "gsim_id", "parameter"]].drop_duplicates().shape[0]
        add("strata", "strict_months", label, round(obs, 4),
            round(float(sh.min()), 4), round(float(sh.max()), 4),
            n_combos, "combinations")
    for fam, f, w in (("std_ccm", "std_f", "std_w"), ("des_ccm", "des_f", "des_w")):
        for label, sub in ((">11 complete years", c[c.y > 11]),
                           ("<=11 complete years", c[c.y <= 11])):
            obs = 100 * (sub[f] > sub[w]).mean()
            sh = []
            for k in KS:
                pair = sub[[f"{f}_{k}", f"{w}_{k}"]].dropna()
                sh.append(100 * (pair.iloc[:, 0] > pair.iloc[:, 1]).mean())
            add("strata", fam, label, round(obs, 4),
                round(float(np.min(sh)), 4), round(float(np.max(sh)), 4),
                len(sub), "combinations")

    # ---------------- seed stability (needs the archives) -----------------
    raw = f"{ROOT}/data/raw"
    if os.path.isdir(raw):
        from pipeline import (ccm_run, read_gsim, merged_calendar,
                              fill_short_gaps_pair, detrend)
        from te import load_wq_total
        wq, _ = load_wq_total()
        combos = pd.read_csv(f"{T}/combos.csv")
        combos = combos[combos.n_valid >= 60]
        seeds = (42, 1, 2)
        votes = {"std": {s: [] for s in seeds}, "des": {s: [] for s in seeds}}
        for i, r in combos.iterrows():
            flow = read_gsim(r.gsim_id)
            series = wq.get(r.gems_id, {}).get(r.parameter, {})
            if flow is None or len(series) < 60:
                continue
            x_raw, y_raw, vobs, cal = merged_calendar(flow, series)
            if vobs.sum() < 60:
                continue
            x, y, valid = fill_short_gaps_pair(x_raw, y_raw, vobs, max_gap=2)
            if np.nanstd(y) < 1e-6 or len(np.unique(np.round(y, 6))) < 10:
                continue
            months = np.array([p.month for p in cal])
            da = detrend(x - np.array([np.nanmean(x[(months == mm) & valid])
                                       if ((months == mm) & valid).any() else 0 for mm in months]))
            db = detrend(y - np.array([np.nanmean(y[(months == mm) & valid])
                                       if ((months == mm) & valid).any() else 0 for mm in months]))
            for s in seeds:
                fs, _, _ = ccm_run(y, x, valid, 0, None, seed=s)
                ws, _, _ = ccm_run(x, y, valid, 0, None, seed=s)
                votes["std"][s].append(1 if fs > ws else 0)
                gs, _, _ = ccm_run(db, da, valid, 0, None, seed=s)
                hs, _, _ = ccm_run(da, db, valid, 0, None, seed=s)
                votes["des"][s].append(1 if gs > hs else 0)
            if (i + 1) % 100 == 0:
                print(f"  seeds {i+1} ({(time.time()-t0)/60:.1f} min)", flush=True)
        for fam in ("std", "des"):
            for s in seeds:
                share = 100 * np.mean(votes[fam][s])
                add("seed_stability", f"{fam}_ccm", f"library-split seed {s}",
                    round(float(share), 4), np.nan, np.nan,
                    len(votes[fam][s]), "combinations")
            base = np.asarray(votes[fam][42])
            flips = sum(int(np.sum(np.asarray(votes[fam][s]) != base))
                        for s in seeds if s != 42)
            add("seed_stability", f"{fam}_ccm",
                "combinations changing vote vs seed 42 (summed over seeds 1, 2)",
                float(flips), np.nan, np.nan, len(base), "combinations")
            sh = [100 * np.mean(votes[fam][s]) for s in seeds]
            print(f"seed stability {fam}: {[round(x,1) for x in sh]} "
                  f"(spread {max(sh)-min(sh):.1f} pp; flips vs 42: {flips})")
    else:
        print("seed block skipped: no data/raw (needs the archives)")

    # ---------------- rotation-seam audit (needs the archives) ------------
    raw = f"{ROOT}/data/raw"
    if os.path.isdir(raw):
        from pipeline import read_gsim, merged_calendar, year_shift, extend_short_gaps
        from te import load_wq_total
        wq, _ = load_wq_total()
        combos = pd.read_csv(f"{T}/combos.csv")
        combos = combos[combos.n_valid >= 60]
        seam_rows = []
        for _, r in combos.iterrows():
            flow = read_gsim(r.gsim_id)
            series = wq.get(r.gems_id, {}).get(r.parameter, {})
            if flow is None or len(series) < 60:
                continue
            x_raw, y_raw, vobs, cal = merged_calendar(flow, series)
            if vobs.sum() < 60:
                continue
            obs_x = ~np.isnan(x_raw)
            obs_y = ~np.isnan(y_raw)
            usable = extend_short_gaps(obs_x)
            for k in KS:
                old = extend_short_gaps(year_shift(obs_x, k) & obs_y)
                new = old & year_shift(usable, k)
                drop = np.where(old & ~new)[0]
                if len(drop):
                    seam_rows.append({
                        "gems_id": r.gems_id, "gsim_id": r.gsim_id,
                        "parameter": r.parameter, "k": k,
                        "n_dropped": int(len(drop)),
                        "months": ";".join(str(cal[p]) for p in drop)})
        pd.DataFrame(seam_rows, columns=["gems_id", "gsim_id", "parameter",
                                         "k", "n_dropped", "months"]).to_csv(
            f"{T}/rotation_seam_audit.csv", index=False)
        print(f"rotation seam audit: {len(seam_rows)} rows, "
              f"{sum(r['n_dropped'] for r in seam_rows)} months removed")

    out = pd.DataFrame(rows, columns=["check", "family", "scope", "value_mean",
                                      "value_min", "value_max", "n", "n_unit"])
    out.to_csv(f"{T}/null_robustness.csv", index=False)
    print(f"\nwritten: {T}/null_robustness.csv ({len(out)} rows, "
          f"{(time.time()-t0)/60:.1f} min)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
