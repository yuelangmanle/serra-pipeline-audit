#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poyang / Taihu CCM layer (kept as the reference implementation).

Sentinel cleaning first (physically impossible negatives -> NaN), then the
calendar-aligned, deseasonalized cross mappings. lakes_null.py builds the
permutation nulls on top of this.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

ROOT = os.environ.get("PAPER_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = f"{ROOT}/data/results/tables"
N_SURR = 200
SEED = 42


def prep(series, max_gap=2):
    """Sentinel-safe preparation: NaN for impossible values, convert to monthly
    periods, reindex to the true monthly calendar, linear interpolation of
    interior gaps <= max_gap."""
    s = series.copy()
    s[s < 0] = np.nan
    s.index = pd.PeriodIndex(s.index, freq="M")
    cal = pd.period_range(s.index.min(), s.index.max(), freq="M")
    s = s.reindex(cal)
    filled = s.interpolate(limit=max_gap, limit_area="inside")
    n_gaps = int(s.isna().sum() - filled.isna().sum())
    return filled.values.astype(float), cal, n_gaps


def deseason_z(v, cal, valid):
    months = np.array([p.month for p in cal])
    out = v.copy()
    for m in np.unique(months[valid]):
        sel = (months == m) & valid
        out[sel] -= out[sel].mean()
    sd = out[valid].std()
    out = (out - out[valid].mean()) / (sd + 1e-12)
    return out


def simplex_ccm(base, target, valid, E, tau=1, lib_fraction=0.7, seed=SEED, ahead=0):
    n = len(base)
    idx = []
    for t in range((E - 1) * tau, n):
        tt = t + ahead
        if tt >= n or not valid[tt] or np.isnan(target[tt]):
            continue
        if all(valid[t - j * tau] for j in range(E)) and not np.isnan(base[t - (E - 1) * tau]):
            idx.append(t)
    idx = np.array(sorted(set(idx)), dtype=int)
    if len(idx) < 30:
        return np.nan, len(idx)
    Mv = np.array([[base[t - j * tau] for j in range(E)] for t in idx])
    tgt = np.array([target[t + ahead] for t in idx])
    rng = np.random.RandomState(seed)
    pos = np.arange(len(idx)); rng.shuffle(pos)
    L = int(len(pos) * lib_fraction)
    lib_m, lib_t = Mv[pos[:L]], tgt[pos[:L]]
    pm, pt = Mv[pos[L:]], tgt[pos[L:]]
    if len(lib_m) < E + 2 or len(pm) < 8:
        return np.nan, len(idx)
    tree = cKDTree(lib_m)
    d, ix = tree.query(pm, k=E + 1)
    w = np.exp(-d / np.maximum(d[:, [0]], 1e-10))
    w /= w.sum(axis=1, keepdims=True)
    P = np.sum(w * lib_t[ix], axis=1)
    return stats.pearsonr(P, pt)[0], len(idx)


def surrogate_p(base, target, valid, rho_obs, E, n_surr=N_SURR, seed=SEED):
    rng = np.random.RandomState(seed)
    b = 0
    for _ in range(n_surr):
        k12 = rng.randint(1, 12) * 12
        bs = np.roll(base, k12)
        vs = np.roll(valid, k12)  # valid mask rolls with the values
        r, _ = simplex_ccm(bs, target, vs, E)
        if not np.isnan(r) and r >= rho_obs:
            b += 1
    return (b + 1) / (n_surr + 1)


def ccm_direction(x, y, valid, n_surr=N_SURR, seed=SEED):
    """Return dict of E, rho, p for x->y (estimate x from y's manifold)."""
    best = {"rho": np.nan, "E": np.nan, "p": np.nan}
    for E in range(2, 7):
        r, _ = simplex_ccm(y, x, valid, E, seed=seed)
        if np.isnan(r):
            continue
        if np.isnan(best["rho"]) or r > best["rho"]:
            p = surrogate_p(y, x, valid, r, E, seed=seed + E)
            best = {"rho": r, "E": E, "p": p}
    return best


def sen_slope(v, years):
    v = pd.Series(v, index=years).dropna()
    slopes = []
    yrs = v.index.values
    vals = v.values
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((vals[j] - vals[i]) / (yrs[j] - yrs[i]))
    return float(np.median(slopes))


def mk_p(v, years):
    v = pd.Series(v, index=years).dropna()
    n = len(v); vals = v.values
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(vals[j] - vals[i])
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(abs(z)))


def main():
    results = []

    # ---------------- Poyang ----------------
    p = pd.read_csv(f"{ROOT}/data/processed/poyang_merged_hydrology_wq.csv")
    p["date"] = pd.to_datetime(dict(year=p.year, month=p.month, day=1))
    p = p.set_index("date").sort_index()
    sent = {c: int((p[c] < 0).sum()) for c in ["TP", "Chla", "SD"]}

    wl, cal_wl, _ = prep(p.water_level)
    tn, cal_tn, _ = prep(p.TN)
    tp, cal_tp, _ = prep(p.TP)
    dis, cal_dis, _ = prep(p.discharge.dropna())
    # common calendar
    cal = cal_tn
    WL = pd.Series(wl, index=cal_wl).reindex(cal).values
    TN = pd.Series(tn, index=cal_tn).reindex(cal).values
    TP = pd.Series(tp, index=cal_tp).reindex(cal).values
    DIS = pd.Series(dis, index=cal_dis).reindex(cal).values

    def run_pair(a, b, label, cal):
        va = deseason_z(a, cal, ~np.isnan(a))
        vb = deseason_z(b, cal, ~np.isnan(b))
        valid = ~np.isnan(va) & ~np.isnan(vb)
        fwd = ccm_direction(va, vb, valid)   # a->b: estimate a from b manifold
        bwd = ccm_direction(vb, va, valid)
        r_lin = stats.pearsonr(a[valid], b[valid])
        results.append({"lake": "Poyang", "pair": label, "direction": f"{label.split('~')[0]}->{label.split('~')[1]}",
                        "E": fwd["E"], "rho": fwd["rho"], "p": fwd["p"], "n": int(valid.sum()),
                        "pearson_r": r_lin[0]})
        results.append({"lake": "Poyang", "pair": label, "direction": f"{label.split('~')[1]}->{label.split('~')[0]}",
                        "E": bwd["E"], "rho": bwd["rho"], "p": bwd["p"], "n": int(valid.sum()),
                        "pearson_r": r_lin[0]})

    run_pair(WL, TN, "WL~TN", cal)
    run_pair(WL, TP, "WL~TP", cal)
    run_pair(DIS, TN, "DIS~TN", cal)
    run_pair(DIS, TP, "DIS~TP", cal)

    # Poyang WL->TN E-sensitivity (cleaned), honest report
    va = deseason_z(WL, cal, ~np.isnan(WL))
    vb = deseason_z(TN, cal, ~np.isnan(TN))

    # ---------------- Taihu (through 2022-12) ----------------
    t = pd.read_csv(f"{ROOT}/data/processed/taihu_monthly_water_quality.csv")
    t["date"] = pd.to_datetime(dict(year=t.year, month=t.month, day=1))
    t = t[t.date <= "2022-12-01"].set_index("date").sort_index()
    taihu_sent = {c: int((t[c] < 0).sum()) for c in ["TP", "Chla", "SD"]}

    def prep_t(col):
        v, cal, ngaps = prep(t[col])
        return v, cal, ngaps
    ttp, calT, g1 = prep_t("TP")
    ttn, calT, g2 = prep_t("TN")
    tch, calT, g3 = prep_t("Chla")

    def run_pair_t(a, b, label):
        va = deseason_z(a, calT, ~np.isnan(a))
        vb = deseason_z(b, calT, ~np.isnan(b))
        valid = ~np.isnan(va) & ~np.isnan(vb)
        fwd = ccm_direction(va, vb, valid)
        bwd = ccm_direction(vb, va, valid)
        results.append({"lake": "Taihu", "pair": label,
                        "direction": f"{label.split('~')[0]}->{label.split('~')[1]}",
                        "E": fwd["E"], "rho": fwd["rho"], "p": fwd["p"], "n": int(valid.sum()),
                        "pearson_r": stats.pearsonr(a[valid], b[valid])[0]})
        results.append({"lake": "Taihu", "pair": label,
                        "direction": f"{label.split('~')[1]}->{label.split('~')[0]}",
                        "E": bwd["E"], "rho": bwd["rho"], "p": bwd["p"], "n": int(valid.sum()),
                        "pearson_r": stats.pearsonr(a[valid], b[valid])[0]})

    run_pair_t(ttp, tch, "TP~Chla")
    run_pair_t(ttn, tch, "TN~Chla")
    run_pair_t(ttn, ttp, "TN~TP")

    # Taihu supporting stats (through 2022)
    yrs = t.index.year.values
    tn_trend = (sen_slope(t.TN.values, yrs), mk_p(t.TN.values, yrs))
    ch_trend = (sen_slope(t.Chla.values, yrs), mk_p(t.Chla.values, yrs))
    sub_early = t[t.year <= 2008][["TP", "Chla"]].dropna()
    r_early = stats.pearsonr(sub_early.TP, sub_early.Chla)[0]
    sub = t[t.year >= 2009][["TP", "Chla"]].dropna()
    r_late = stats.pearsonr(sub.TP, sub.Chla)[0]

    res = pd.DataFrame(results)
    res.to_csv(f"{T}/dual_lake.csv", index=False)
    print(f"Poyang sentinel months cleaned: {sent}; Taihu: {taihu_sent}")
    print(f"Taihu TN trend: Sen={tn_trend[0]:+.4f} mg/L/yr MK p={tn_trend[1]:.4f}")
    print(f"Taihu Chla trend: Sen={ch_trend[0]:+.4f} ug/L/yr MK p={ch_trend[1]:.4f}")
    print(f"Taihu Chla-TP r: {r_early:.3f} (1992-2008) -> {r_late:.3f} (2009-2022)")
    print("\n=== dual-lake directions ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
