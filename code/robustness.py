#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E x tau sweep, ahead scan, strict months, library fractions, c-Q slopes.

One script because they share the data loading; each block prints its own
summary and writes its own table.
"""
import os
import sys
import zipfile
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
from pipeline import (read_gsim, build_wq_total, merged_calendar,  # noqa: E402
                                 fill_short_gaps_pair, detrend, ccm_run,
                                 seasonal_controls, ksg_mi)

T = f"{ROOT}/data/results/tables"
K, SEED, LIBF = 4, 42, 0.7


def load_pairs():
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv")
    wq, _ = build_wq_total(set(qual.gems_id))
    combos = pd.read_csv(f"{T}/combos.csv")
    cache = {}
    for _, r in combos.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        flow = read_gsim(sid)
        series = wq.get(gid, {}).get(param, {})
        if flow is None or len(series) < 60:
            continue
        x, y, valid_obs, cal = merged_calendar(flow, series)
        if valid_obs.sum() < 60:
            continue
        cache[(gid, sid, param)] = (x, y, valid_obs, cal)
    return cache


def share(votes):
    v = np.asarray(votes, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, 0
    p = stats.binomtest(int(v.sum()), len(v), 0.5).pvalue
    return v.mean(), p


def main():
    cache = load_pairs()
    print(f"loaded {len(cache)} pairs", flush=True)

    esw, ahead_rows, strict, cq = [], [], [], []
    for i, ((gid, sid, param), (x_raw, y_raw, valid_obs, cal)) in enumerate(cache.items()):
        x, y, valid = fill_short_gaps_pair(x_raw, y_raw, valid_obs, max_gap=2)
        dx, dy = detrend(x), detrend(y)

        # ---- 1. E x tau sweep (gap-filled, same-time) ----
        for E in (2, 3, 4):
            for tau in (1, 2):
                f2w, _, _ = ccm_run(y, x, valid, 0, None, LIBF, SEED, E, tau, K)
                w2f, _, _ = ccm_run(x, y, valid, 0, None, LIBF, SEED, E, tau, K)
                if not (np.isnan(f2w) or np.isnan(w2f)):
                    esw.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                                "E": E, "tau": tau, "vote": 1 if f2w > w2f else 0})

        # ---- 2. ahead scan (E=3, tau=1, gap-filled) ----
        for ah in range(7):
            f2w, _, _ = ccm_run(y, x, valid, ah, None, LIBF, SEED, 3, 1, K)
            w2f, _, _ = ccm_run(x, y, valid, ah, None, LIBF, SEED, 3, 1, K)
            if not (np.isnan(f2w) or np.isnan(w2f)):
                ahead_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                                   "ahead": ah, "f2w": f2w, "w2f": w2f,
                                   "vote": 1 if f2w > w2f else 0,
                                   "diff": f2w - w2f})

        # ---- 3. strict observed months ----
        v2 = valid_obs & ~np.isnan(x_raw) & ~np.isnan(y_raw)
        if v2.sum() >= 60:
            def detrend_obs(a):
                t = np.arange(len(a))
                sl, ic, _, _, _ = stats.linregress(t[v2], a[v2])
                return a - (sl * t + ic)
            dxo, dyo = detrend_obs(x_raw), detrend_obs(y_raw)
            f2w_s, _, _ = ccm_run(y_raw, x_raw, v2, 0, None, LIBF, SEED, 3, 1, K)
            w2f_s, _, _ = ccm_run(x_raw, y_raw, v2, 0, None, LIBF, SEED, 3, 1, K)
            if not (np.isnan(f2w_s) or np.isnan(w2f_s)):
                strict.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                               "method": "std_ccm",
                               "vote": 1 if f2w_s > w2f_s else 0})
            # z-score on observed values only
            xm = dxo[v2].mean(); xs = dxo[v2].std()
            ym = dyo[v2].mean(); ys = dyo[v2].std()
            if xs > 1e-9 and ys > 1e-9:
                xn = (dxo - xm) / xs; yn = (dyo - ym) / ys
                # proper Schreiber form under the strict counting rule
                idx_ok = np.array([q for q in range(1, len(xn))
                                   if v2[q] and v2[q - 1]], dtype=int)
                xp_b, yp_b = xn[idx_ok - 1], yn[idx_ok - 1]
                te_f = float(ksg_mi(np.column_stack([xp_b, yp_b]), yn[idx_ok])
                             - ksg_mi(yp_b, yn[idx_ok]))
                te_w = float(ksg_mi(np.column_stack([yp_b, xp_b]), xn[idx_ok])
                             - ksg_mi(xp_b, xn[idx_ok]))
                if len(idx_ok) >= 40:
                    strict.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                                   "method": "te_z",
                                   "vote": 1 if te_f > te_w else 0})

        # ---- 4. c-Q slope (observed months) ----
        q = x_raw[valid_obs]; c = y_raw[valid_obs]
        ok = (q > 0) & (c > 0)
        if ok.sum() >= 30:
            sl, _, _, _, _ = stats.linregress(np.log10(q[ok]), np.log10(c[ok]))
            cq.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                       "cq_slope": sl, "n_months": int(ok.sum())})

        if (i + 1) % 50 == 0:
            print(f"done {i+1}", flush=True)

    pd.DataFrame(esw).to_csv(f"{T}/batch2_esweep.csv", index=False)
    pd.DataFrame(ahead_rows).to_csv(f"{T}/batch2_aheadscan.csv", index=False)
    pd.DataFrame(strict).to_csv(f"{T}/batch2_strict.csv", index=False)
    pd.DataFrame(cq).to_csv(f"{T}/batch2_cq.csv", index=False)

    # summaries
    print("\n=== E x tau sweep ===")
    e = pd.DataFrame(esw)
    for (E, tau), g in e.groupby(["E", "tau"]):
        s, p = share(g.vote)
        print(f"E={E} tau={tau}: share={s:.3f} (p={p:.3f}, n={len(g)})")
    print("\n=== ahead scan ===")
    a = pd.DataFrame(ahead_rows)
    for ah, g in a.groupby("ahead"):
        s, p = share(g.vote)
        print(f"ahead={ah}: share={s:.3f} (p={p:.3f}, n={len(g)}), mean diff={g['diff'].mean():+.4f}")
    print("\n=== strict months ===")
    s2 = pd.DataFrame(strict)
    for meth, g in s2.groupby("method"):
        s, p = share(g.vote)
        print(f"{meth}: share={s:.3f} (p={p:.4f}, n={len(g)})")
    print("\n=== c-Q slopes ===")
    c = pd.DataFrame(cq)
    for param, g in c.groupby("parameter"):
        print(f"{param}: median slope={g.cq_slope.median():+.3f} "
              f"(IQR {g.cq_slope.quantile(.25):+.3f}..{g.cq_slope.quantile(.75):+.3f}, "
              f"n={len(g)}, %negative={(g.cq_slope < 0).mean():.0%})")
    print("saved batch2 tables")


if __name__ == "__main__":
    main()
