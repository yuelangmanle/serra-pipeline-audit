#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Earlier pass: year-block nulls for the lakes, before the
argmax-matched redesign in lakes_null.py. Kept for the record.

REFUSES TO RUN: it would overwrite the current lakes_null.csv; the
current null lives in code/lakes_null.py.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
from lakes import prep, deseason_z, simplex_ccm  # noqa: E402

T = f"{ROOT}/data/results/tables"
N_SURR = 1000
SEED = 42


def year_perm_p(base, target, valid, years, rho_obs, E, n_surr=N_SURR, seed=SEED):
    yr_counts = pd.Series(years).value_counts()
    full = sorted(yr_counts[yr_counts == 12].index)
    mask = np.isin(years, full)
    idx = np.where(mask)[0]
    if len(full) < 4:
        return np.nan, len(full)
    blocks = [idx[i * 12:(i + 1) * 12] for i in range(len(full))]
    tgt = target[idx]
    rng = np.random.RandomState(seed)
    b = 0
    for _ in range(n_surr):
        order = rng.permutation(len(blocks))
        bs = np.concatenate([base[blocks[o]] for o in order])
        vs = np.concatenate([valid[blocks[o]] for o in order])
        r, _ = simplex_ccm(bs, tgt, vs, E)
        if not np.isnan(r) and r >= rho_obs:
            b += 1
    return (b + 1) / (n_surr + 1), len(full)


def detrend_valid(a, valid):
    t = np.arange(len(a))
    sl, ic, _, _, _ = stats.linregress(t[valid], a[valid])
    return a - (sl * t + ic)


def load_pairs():
    """Replicates lakes.main() series construction."""
    pairs = {}
    p = pd.read_csv(f"{ROOT}/data/processed/poyang_merged_hydrology_wq.csv")
    p["date"] = pd.to_datetime(dict(year=p.year, month=p.month, day=1))
    p = p.set_index("date").sort_index()
    cal = None
    series = {}
    for name, col in [("WL", "water_level"), ("TN", "TN"), ("TP", "TP"), ("DIS", "discharge")]:
        v, c, _ = prep(p[col].dropna() if col == "discharge" else p[col])
        series[name] = pd.Series(v, index=c)
        cal = c
    idx = pd.period_range(cal.min(), cal.max(), freq="M")
    for name, s in series.items():
        series[name] = s.reindex(idx)
    years = np.array([p_.year for p_ in idx])
    for a, b in [("WL", "TN"), ("WL", "TP"), ("DIS", "TN"), ("DIS", "TP")]:
        pairs[("Poyang", a, b)] = (series[a].values, series[b].values,
                                   idx, years)
    t = pd.read_csv(f"{ROOT}/data/processed/taihu_monthly_water_quality.csv")
    t["date"] = pd.to_datetime(dict(year=t.year, month=t.month, day=1))
    t = t[t.date <= "2022-12-01"].set_index("date").sort_index()
    cal_t = pd.period_range(t.index.min(), t.index.max(), freq="M")
    series_t = {c: pd.Series(prep(t[c])[0], index=cal_t) for c in ["TN", "TP", "Chla"]}
    years_t = np.array([p_.year for p_ in cal_t])
    for a, b in [("TN", "TP"), ("TN", "Chla"), ("TP", "Chla")]:
        pairs[("Taihu", a, b)] = (series_t[a].values, series_t[b].values, cal_t, years_t)
    return pairs


def main():
    raise SystemExit("archived script kept for the record; the current lake "
                     "null is produced by code/lakes_null.py")
    old = pd.read_csv(f"{T}/dual_lake.csv")
    pairs = load_pairs()
    rows = []
    for (lake, a, b), (sa, sb, cal, years) in pairs.items():
        for (x, y, dname) in [(sa, sb, f"{a}->{b}"), (sb, sa, f"{b}->{a}")]:
            sel = old[(old.lake == lake) & (old.direction == dname)]
            if sel.empty:
                continue
            E = int(sel.E.iloc[0])
            rho_old = float(sel.rho.iloc[0])
            p_old = float(sel.p.iloc[0])
            # original (deseason+z) variant
            va = deseason_z(x, cal, ~np.isnan(x))
            vb = deseason_z(y, cal, ~np.isnan(y))
            valid = ~np.isnan(va) & ~np.isnan(vb)
            r, _ = simplex_ccm(vb, va, valid, E)      # x->y: estimate x from y manifold
            p_new, n_full = year_perm_p(vb, va, valid, years, r, E)
            # detrended variant
            vad = detrend_valid(x, ~np.isnan(x))
            vbd = detrend_valid(y, ~np.isnan(y))
            vadz = deseason_z(vad, cal, ~np.isnan(vad))
            vbdz = deseason_z(vbd, cal, ~np.isnan(vbd))
            validd = ~np.isnan(vadz) & ~np.isnan(vbdz)
            rd, _ = simplex_ccm(vbdz, vadz, validd, E)
            pd_new, _ = year_perm_p(vbdz, vadz, validd, years, rd, E, seed=SEED + 1)
            rows.append({"lake": lake, "direction": dname, "E": E,
                         "rho": round(r, 3), "p_shift11_old": p_old,
                         "p_yearperm": round(p_new, 4),
                         "rho_detrended": round(rd, 3),
                         "p_yearperm_detrended": round(pd_new, 4),
                         "n": int(valid.sum()), "n_full_years": n_full})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{T}/lakes_null.csv", index=False)
    print("saved", f"{T}/lakes_null.csv")


if __name__ == "__main__":
    main()
