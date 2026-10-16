#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standardized transfer entropy and conditional TE, Schreiber form.

Grew out of the exact-zero problem: with unstandardized series the max-norm
joint distances coincide and the two KSG terms cancel bitwise (the zeros in
the unstandardized table). z-scoring each series on its valid months fixes
that; the strict shares themselves live in strict_layer.py, and this
file keeps the shared helpers plus the ablation ladder rungs, the naive-491
companion table, and the estimator-calibration checks. Conditional TE
conditions on the sin/cos calendar controls.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"
sys.path.insert(0, HERE)

from pipeline import (  # noqa: E402
    read_gsim, build_wq_total, merged_calendar, fill_short_gaps_pair,
    detrend, seasonal_controls, ksg_mi, valid_targets,
)
import zipfile  # noqa: E402  (naive readers below)
from scipy.spatial import cKDTree as KDTree  # noqa: E402
from scipy.special import digamma  # noqa: E402

N_SURR = 200
SEED = 42
LAG = 1
MIN_PAIRS = 40


# =================================================================== helpers

def zscore_valid(a, valid):
    """z-score with mean/SD over VALID observations only (ddof=0)."""
    mu = a[valid].mean()
    sd = a[valid].std()
    return (a - mu) / (sd + 1e-12)


def zscore(a):
    """Naive z-score (naive s6 convention, all packed rows)."""
    return (a - a.mean()) / (a.std() + 1e-10)


def te_cte_proper(xp, yp, zp, yt, k=3):
    """Proper Schreiber TE and conditional TE from lag-aligned rows.
    TE(X->Y)      = I(Yt; [Xp,Yp]) - I(Yt; Yp)
    CTE(X->Y; Z)  = I(Yt; [Xp,Yp,Zp]) - I(Yt; [Yp,Zp])
    (ksg_mi imported from pipeline, reused verbatim.)"""
    te = ksg_mi(np.column_stack([xp, yp]), yt, k=k) - ksg_mi(yp, yt, k=k)
    cte = (ksg_mi(np.column_stack([xp, yp, zp]), yt, k=k)
           - ksg_mi(np.column_stack([yp, zp]), yt, k=k))
    return te, cte


def process_pair(x_raw, y_raw, valid_obs, cal):
    """The corrected recipe on one raw pair: fill short gaps, detrend,
    z-score on valid months. Returns (dzx, dzy, valid, cal) or None."""
    x, y, valid = fill_short_gaps_pair(x_raw, y_raw, valid_obs, max_gap=2)
    if np.nanstd(y) < 1e-6 or len(np.unique(np.round(y, 6))) < 10:
        return None
    dx, dy = detrend(x), detrend(y)
    return zscore_valid(dx, valid), zscore_valid(dy, valid), valid, cal


def build_corrected(sid, series_dict, min_valid=60):
    """Corrected pipeline series + the z-score fix (single-combination entry).
    Returns (dzx, dzy, valid, cal) with dz* = z-scored detrended series,
    or None."""
    flow = read_gsim(sid)
    if flow is None or len(series_dict) < 60:
        return None
    x_raw, y_raw, valid_obs, cal = merged_calendar(flow, series_dict)
    if valid_obs.sum() < min_valid:
        return None
    return process_pair(x_raw, y_raw, valid_obs, cal)


def lag_pairs(dzx, dzy, valid, lag=LAG):
    idx = np.array([t for t in range(lag, len(dzx))
                    if valid[t] and valid[t - lag]], dtype=int)
    return idx, dzx[idx - lag], dzy[idx - lag], dzy[idx]


def binom_p(k, n):
    return float(stats.binomtest(k, n, 0.5).pvalue) if n else np.nan


def load_wq_total():
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv")
    wq, n_frac = build_wq_total(set(qual.gems_id))
    return wq, n_frac


# ------------------------------------------------------- naive layer (legacy)

def naive_read_gsim(gsim_id):
    """Legacy naive GSIM reader (original semantics preserved verbatim)."""
    z = zipfile.ZipFile(f"{ROOT}/data/raw/global/gsim/indices/GSIM_indices.zip")
    try:
        with z.open(f"GSIM_indices/TIMESERIES/monthly/{gsim_id}.mon") as f:
            lines = f.read().decode("utf-8", "ignore").split("\n")
    except KeyError:
        return None
    dl = [l for l in lines if not l.startswith("#") and l.strip()]
    dates, vals = [], []
    for line in dl[1:]:
        p = line.replace(",", "").split()
        if len(p) >= 2:
            dates.append(p[0])
            try:
                v = float(p[1])
                vals.append(np.nan if v <= -900 or abs(v) > 1e6 else v)
            except Exception:
                vals.append(np.nan)
    return (pd.DataFrame({"ym": pd.to_datetime(dates, errors="coerce"), "flow": vals})
            .dropna().assign(ym=lambda d: d.ym.dt.to_period("M")))


def naive_build_wq(qids):
    """Legacy naive composite WQ builder: ALL species codes (preserved verbatim)."""
    wq = {}
    for param, fn, codes in [
            ("TN", "Other_Nitrogen.csv",
             ["TN", "TOTN", "NO3+NO2", "NO3N", "NO2N", "NH3N", "NH4N"]),
            ("TP", "Phosphorus.csv",
             ["TP", "TOTP", "PO4P", "DRP", "ORTHOP"])]:
        df = pd.read_csv(f"{ROOT}/data/raw/global/gemstat/extracted/{fn}",
                         encoding="latin-1", low_memory=False)
        df = df[df["GEMS Station Number"].isin(qids) & df["Parameter Code"].isin(codes)]
        df["date"] = pd.to_datetime(df["Sample Date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["ym"] = df.date.dt.to_period("M")
        df["value"] = pd.to_numeric(df["Value"], errors="coerce")
        df.loc[df["Value Flags"] == "<", "value"] = df.loc[df["Value Flags"] == "<", "value"] / 2
        df = df[(df.value >= 0) & (df.value <= (100 if param == "TN" else 10))]
        m = df.groupby(["GEMS Station Number", "ym"])["value"].mean().reset_index()
        for _, r in m.iterrows():
            wq.setdefault(r["GEMS Station Number"], {}).setdefault(str(r["ym"]), {})[param] = r["value"]
    return wq


def naive_merge_rows(flow_df, series_dict):
    """Row-index pairing: inner join on observed months (naive lag = row arithmetic)."""
    rec = [(ym, v) for ym, v in series_dict.items()]
    wqd = pd.DataFrame(rec, columns=["ym", "wq"])
    wqd["ym"] = pd.PeriodIndex(wqd.ym, freq="M")
    return pd.merge(flow_df, wqd, on="ym", how="inner").sort_values("ym")


def naive_kraskov_mi(x, y, k=3):
    """Naive (biased) Kraskov estimator, preserved verbatim from the
    original transfer-entropy code (Euclidean joint k-NN distance +
    KSG counting formulas)."""
    n = len(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    xy = np.hstack([x, y])
    tree_xy = KDTree(xy)
    tree_x = KDTree(x)
    tree_y = KDTree(y)
    dists, _ = tree_xy.query(xy, k=k + 1)
    eps = dists[:, k]
    nx = np.array([len(tree_x.query_ball_point(xi, eps_i)) - 1 for xi, eps_i in zip(x, eps)])
    ny = np.array([len(tree_y.query_ball_point(yi, eps_i)) - 1 for yi, eps_i in zip(y, eps)])
    return digamma(k) + digamma(n) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))


def legacy_transfer_entropy(x, y, lag=1, k=3):
    """Original transfer-entropy implementation, verbatim (already Schreiber proper)."""
    y_t = y[lag:].reshape(-1, 1)
    y_past = y[:-lag].reshape(-1, 1)
    x_past = x[:-lag].reshape(-1, 1)
    mi_joint = naive_kraskov_mi(y_t, np.hstack([x_past, y_past]), k=k)
    mi_y = naive_kraskov_mi(y_t, y_past, k=k)
    return mi_joint - mi_y


def legacy_conditional_te(x, y, z, lag=1, k=3):
    """Original conditional-TE implementation, verbatim."""
    y_t = y[lag:].reshape(-1, 1)
    y_past = y[:-lag].reshape(-1, 1)
    x_past = x[:-lag].reshape(-1, 1)
    z_past = z[:-lag]
    if z_past.ndim == 1:
        z_past = z_past.reshape(-1, 1)
    mi_full = naive_kraskov_mi(y_t, np.hstack([x_past, y_past, z_past]), k=k)
    mi_cond = naive_kraskov_mi(y_t, np.hstack([y_past, z_past]), k=k)
    return mi_full - mi_cond


# =================================================================== step 1

def step1():
    t0 = time.time()
    combos = pd.read_csv(f"{T}/combos.csv")
    wq, _ = load_wq_total()
    rows = []
    for i, c in combos.iterrows():
        gid, sid, param = c.gems_id, c.gsim_id, c.parameter
        b = build_corrected(sid, wq.get(gid, {}).get(param, {}))
        if b is None:
            print(f"  WARNING: combo {gid}/{sid}/{param} failed rebuild", flush=True)
            continue
        dzx, dzy, valid, cal = b
        idx, xp, yp, yt_w = lag_pairs(dzx, dzy, valid)
        if len(idx) < MIN_PAIRS:
            continue
        zp = seasonal_controls(cal)[idx - LAG]
        xt = dzx[idx]
        te_f2w, cte_f2w = te_cte_proper(xp, yp, zp, yt_w)     # flow -> WQ
        te_w2f, cte_w2f = te_cte_proper(yp, xp, zp, xt)       # WQ -> flow
        rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                     "te_f2w": te_f2w, "te_w2f": te_w2f,
                     "cte_f2w": cte_f2w, "cte_w2f": cte_w2f, "n_pairs": len(idx)})
        if (i + 1) % 50 == 0:
            print(f"step1: {i + 1}/{len(combos)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    out = pd.DataFrame(rows)
    # the shares themselves live in strict_layer.py; this step is kept for the
    # interactive report only and deliberately writes no table
    report_step1(out)
    return out


def report_step1(out):
    n = len(out)
    k_te = int((out.te_f2w > out.te_w2f).sum())
    k_cte = int((out.cte_f2w > out.cte_w2f).sum())
    d_te = out.te_f2w - out.te_w2f
    d_cte = out.cte_f2w - out.cte_w2f
    old = pd.read_csv(f"{T}/te.csv")
    print("\n=== STEP 1: corrected TE/CTE, z-scored + proper Schreiber form "
          f"(n={n}) ===")
    print(f"TE  flow-dom share {100*k_te/n:.1f}% ({k_te}/{n}), "
          f"two-sided binom p = {binom_p(k_te, n):.4g}; "
          f"mean diff {d_te.mean():+.4f}, median {d_te.median():+.4f}")
    print(f"CTE flow-dom share {100*k_cte/n:.1f}% ({k_cte}/{n}), "
          f"two-sided binom p = {binom_p(k_cte, n):.4g}; "
          f"mean diff {d_cte.mean():+.4f}, median {d_cte.median():+.4f}")
    print(f"CTE exact zeros: f2w {int((out.cte_f2w == 0).sum())}/{n}, "
          f"w2f {int((out.cte_w2f == 0).sum())}/{n}")
    print(f"CTE f2w: mean {out.cte_f2w.mean():+.4f}, median {out.cte_f2w.median():+.4f}; "
          f"CTE w2f: mean {out.cte_w2f.mean():+.4f}, median {out.cte_w2f.median():+.4f}")
    try:
        wp = stats.wilcoxon(out.cte_f2w, out.cte_w2f).pvalue
        print(f"CTE f2w vs w2f two-sided Wilcoxon p = {wp:.4g}")
    except ValueError:
        print("CTE f2w vs w2f two-sided Wilcoxon: undefined (all-zero diffs)")
    print(f"[old te.csv for contrast: TE share {100*(old.te_f2w > old.te_w2f).mean():.1f}%, "
          f"CTE share {100*(old.cte_f2w > old.cte_w2f).mean():.1f}%, "
          f"zeros f2w {int((old.cte_f2w == 0).sum())}/w2f {int((old.cte_w2f == 0).sum())}, "
          f"cte_f2w mean {old.cte_f2w.mean():+.4f}, cte_w2f mean {old.cte_w2f.mean():+.4f}]")


# =================================================================== step 3

def step3():
    t0 = time.time()
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv")
    res = res[res.n_months >= 60].reset_index(drop=True)
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv")
    qids = set(qual.gems_id)
    combos316 = pd.read_csv(f"{T}/combos.csv")
    keys316 = list(map(tuple, combos316[["gems_id", "gsim_id", "parameter"]].values))

    wq_all = naive_build_wq(qids)
    wq_tot, _ = load_wq_total()

    rows = []

    # ---------- naive series caches (row-index, detrended; z-scored at use) ----
    cache_all, cache_tot = {}, {}   # SEPARATE caches: all-species vs total-species

    def get_naive(cache, key, series, flow_df):
        if key not in cache:
            if flow_df is None or len(series) < 30:
                cache[key] = None
            else:
                mg = naive_merge_rows(flow_df, series)
                cache[key] = None if len(mg) < 60 else (
                    detrend(mg.flow.values.astype(float)),
                    detrend(mg.wq.values.astype(float)), len(mg))
        return cache[key]

    # ---- T0: naive estimator (s6 verbatim, proper form) + naive data, n=491
    print("\n[T0] all-species + row-index + naive s6 TE (proper form) ...", flush=True)
    t0_rows = []
    done = 0
    for _, r in res.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        sc = get_naive(cache_all, (gid, sid, param),
                       {ym: v[param] for ym, v in wq_all.get(gid, {}).items() if param in v},
                       naive_read_gsim(sid))
        if sc is None:
            continue
        x, y, n_m = sc
        x, y = zscore(x), zscore(y)
        try:
            t0_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                            "te_f2w": legacy_transfer_entropy(x, y, lag=LAG),
                            "te_w2f": legacy_transfer_entropy(y, x, lag=LAG)})
        except Exception as e:
            print(f"  T0 error {gid}: {e}", flush=True)
        done += 1
        if done % 100 == 0:
            print(f"  T0 done {done} ({(time.time()-t0)/60:.1f} min)", flush=True)
    t0df = pd.DataFrame(t0_rows)
    n0 = len(t0df)
    k0 = int((t0df.te_f2w > t0df.te_w2f).sum())
    s0 = k0 / n0
    print(f"  T0: n={n0} share={100*s0:.1f}% binom 2p={binom_p(k0, n0):.3g}", flush=True)
    rows.append({"stage": "T0", "n": n0, "share": round(s0, 4),
                 "p_two_sided": binom_p(k0, n0),
                 "note": "naive estimator = original transfer-entropy code verbatim "
                         "proper Schreiber form, Euclidean-joint naive Kraskov) on naive data "
                         "(all-species codes + row-index lags + naive z-score); binom_two_sided"})

    # ---- T1: corrected estimator + same naive data (n=491)
    print("\n[T1] all-species + row-index + corrected KSG-1, proper form ...", flush=True)
    t1_rows = []
    done = 0
    for _, r in t0df.iterrows():
        key = (r.gems_id, r.gsim_id, r.parameter)
        x, y, n_m = cache_all[key]
        x, y = zscore(x), zscore(y)
        xp, yp, yt = x[:-LAG], y[:-LAG], y[LAG:]
        xt = x[LAG:]
        t1_rows.append({"gems_id": r.gems_id, "gsim_id": r.gsim_id,
                        "parameter": r.parameter,
                        "te_f2w": float(ksg_mi(np.column_stack([xp, yp]), yt)
                                        - ksg_mi(yp, yt)),
                        "te_w2f": float(ksg_mi(np.column_stack([yp, xp]), xt)
                                        - ksg_mi(xp, xt))})
        done += 1
        if done % 100 == 0:
            print(f"  T1 done {done} ({(time.time()-t0)/60:.1f} min)", flush=True)
    t1df = pd.DataFrame(t1_rows)
    n1 = len(t1df)
    k1 = int((t1df.te_f2w > t1df.te_w2f).sum())
    s1 = k1 / n1
    print(f"  T1: n={n1} share={100*s1:.1f}% binom 2p={binom_p(k1, n1):.3g}", flush=True)
    rows.append({"stage": "T1", "n": n1, "share": round(s1, 4),
                 "p_two_sided": binom_p(k1, n1),
                 "note": f"estimator isolated vs T0 (identical sample & series): "
                         f"{100*(s1-s0):+.1f} pp; binom_two_sided"})

    # ---- T2: total-species + row-index + corrected estimator (n=316)
    print("\n[T2] total-species + row-index + corrected KSG-1, proper form ...", flush=True)
    t2_rows = []
    done = 0
    for gid, sid, param in keys316:
        key = (gid, sid, param)
        sc = get_naive(cache_tot, key, wq_tot.get(gid, {}).get(param, {}),
                       naive_read_gsim(sid))
        if sc is None:
            continue
        x, y, n_m = sc
        x, y = zscore(x), zscore(y)
        xp, yp, yt = x[:-LAG], y[:-LAG], y[LAG:]
        xt = x[LAG:]
        t2_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                        "te_f2w": float(ksg_mi(np.column_stack([xp, yp]), yt)
                                        - ksg_mi(yp, yt)),
                        "te_w2f": float(ksg_mi(np.column_stack([yp, xp]), xt)
                                        - ksg_mi(xp, xt))})
        done += 1
        if done % 100 == 0:
            print(f"  T2 done {done} ({(time.time()-t0)/60:.1f} min)", flush=True)
    t2df = pd.DataFrame(t2_rows)
    n2 = len(t2df)
    k2 = int((t2df.te_f2w > t2df.te_w2f).sum())
    s2 = k2 / n2
    print(f"  T2: n={n2} share={100*s2:.1f}% binom 2p={binom_p(k2, n2):.3g}", flush=True)
    rows.append({"stage": "T2", "n": n2, "share": round(s2, 4),
                 "p_two_sided": binom_p(k2, n2),
                 "note": f"species isolated vs T1 (all-species -> total-only; "
                         f"491 -> 316 part of the correction): {100*(s2-s1):+.1f} pp; "
                         f"binom_two_sided"})

    # ---- T3: total-species + calendar + corrected estimator = step 1 table
    print("\n[T3] total-species + calendar + corrected KSG-1 (= step-1 table) ...", flush=True)
    te_z = pd.read_csv(f"{T}/te_z.csv")
    n3 = len(te_z)
    k3 = int((te_z.te_f2w > te_z.te_w2f).sum())
    s3 = k3 / n3
    print(f"  T3: n={n3} share={100*s3:.1f}% binom 2p={binom_p(k3, n3):.3g}", flush=True)
    rows.append({"stage": "T3", "n": n3, "share": round(s3, 4),
                 "p_two_sided": binom_p(k3, n3),
                 "note": "anchor check: T3 = step-1 te_z.csv TE share (computed on the "
                         f"calendar-true pipeline; {n3}/{len(keys316)} combos); "
                         f"indexing isolated vs T2 (row -> calendar-true): {100*(s3-s2):+.1f} pp; "
                         f"binom_two_sided"})

    out = pd.DataFrame(rows)[["stage", "n", "share", "p_two_sided", "note"]]
    out.to_csv(f"{T}/ablation_ladder_te_z.csv", index=False)

    print("\n=== NEW TE ABLATION LADDER (proper form + z-score at every rung) ===", flush=True)
    for _, r in out.iterrows():
        print(f"  {r.stage}: n={r.n} share={100*r.share:.1f}% p={r.p_two_sided:.3g}", flush=True)
    old = pd.read_csv(f"{T}/ablation_ladder.csv")
    oldt = old[old.stage.str.startswith("T")][["stage", "flow_dom_share"]]
    print("old ladder (contrast only, NOT written to the new table):", flush=True)
    for _, r in oldt.iterrows():
        print(f"  {r.stage}: {100*r.flow_dom_share:.1f}%", flush=True)
    print(f"written: {T}/ablation_ladder_te_z.csv", flush=True)


# =================================================================== step 4

def step4():
    t0 = time.time()
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv")
    res = res[res.n_months >= 60].reset_index(drop=True)
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv")
    wq_all = naive_build_wq(set(qual.gems_id))

    rows, errors = [], 0
    done = 0
    for _, r in res.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        flow = naive_read_gsim(sid)
        series = {ym: v[param] for ym, v in wq_all.get(gid, {}).items() if param in v}
        ok = flow is not None and len(series) >= 30
        if ok:
            mg = naive_merge_rows(flow, series)
            ok = len(mg) >= 60
        if not ok:
            continue
        ctrl = np.column_stack([np.sin(2 * np.pi * mg.ym.dt.month.values / 12),
                                np.cos(2 * np.pi * mg.ym.dt.month.values / 12)])
        x = zscore(detrend(mg.flow.values.astype(float)))
        y = zscore(detrend(mg.wq.values.astype(float)))
        try:
            rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                         "n_months": len(mg),
                         "te_f2w": legacy_transfer_entropy(x, y, lag=LAG),
                         "te_w2f": legacy_transfer_entropy(y, x, lag=LAG),
                         "cte_f2w": legacy_conditional_te(x, y, ctrl, lag=LAG),
                         "cte_w2f": legacy_conditional_te(y, x, ctrl, lag=LAG)})
        except Exception as e:
            errors += 1
            print(f"  error {gid}/{param}: {e}", flush=True)
        done += 1
        if done % 50 == 0:
            print(f"step4: {done} combos ({(time.time()-t0)/60:.1f} min)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{T}/transfer_entropy_491.csv", index=False)

    v26 = pd.read_csv(f"{T}/transfer_entropy.csv")
    cte_keys = set(map(tuple, v26[["gems_id", "gsim_id", "parameter"]].values))
    sub = out[[tuple(r) in cte_keys
               for r in out[["gems_id", "gsim_id", "parameter"]].values]]

    print(f"\n=== STEP 4: naive s6 TE/CTE on ALL naive combos (n={len(out)}, errors={errors}) ===")
    for tag, df in [("all-491", out), ("272-subsample", sub)]:
        k = int((df.cte_f2w > df.cte_w2f).sum())
        n = len(df)
        kc = int((df.cte_f2w == 0).sum()) + int((df.cte_w2f == 0).sum())
        print(f"{tag:15s} CTE flow-dom {100*k/n:.1f}% ({k}/{n}, binom 2p={binom_p(k,n):.3g}); "
              f"cte_f2w mean {df.cte_f2w.mean():+.4f} median {df.cte_f2w.median():+.4f}; "
              f"cte_w2f mean {df.cte_w2f.mean():+.4f} median {df.cte_w2f.median():+.4f}; "
              f"exact zeros f2w/w2f {int((df.cte_f2w==0).sum())}/{int((df.cte_w2f==0).sum())}")
    kt = int((out.te_f2w > out.te_w2f).sum())
    print(f"TE (uncond s6 proper) flow-dom {100*kt/len(out):.1f}% ({kt}/{len(out)})")
    print(f"[reference: station_level_sensitivity records naive Conditional TE 65.4% "
          f"combo-level on the 272 subsample; transfer_entropy.csv n={len(v26)}]")
    print(f"written: {T}/transfer_entropy_491.csv")


# =================================================================== step 6

def step6():
    rows = []

    # ---------- (a) z-scored ksg_mi recovers truth on mixed-scale data --------
    n = 5000
    r = np.random.default_rng(SEED)
    u = r.standard_normal(n)
    w = r.standard_normal(n)
    xg = 100.0 * u                       # flow-like, ~1e2
    yg = 0.01 * (0.6 * u + 0.8 * w)      # nutrient-like, ~1e-2, r = 0.6
    true_mi = -0.5 * np.log(1 - 0.36)
    mi_raw = ksg_mi(xg, yg)              # unstandardised (the old pipeline's path)
    mi_z = ksg_mi(zscore_valid(xg, np.ones(n, bool)), zscore_valid(yg, np.ones(n, bool)))
    ok_a = abs(mi_z - true_mi) < 0.03
    print(f"\n=== STEP 6: scale-invariance synthetic validation ===")
    print(f"(a) correlated Gaussian r=0.6, mixed scales 1e2 vs 1e-2, n={n}, seed {SEED}:")
    print(f"    true MI {true_mi:.4f} | z-scored estimate {mi_z:.4f} "
          f"(diff {mi_z-true_mi:+.4f}, <0.03: {ok_a})")
    print(f"    unstandardised estimate {mi_raw:.4f} (diff {mi_raw-true_mi:+.4f}) -- "
          f"the old pipeline's failure mode, same data")
    rows.append({"check": "a_zscored_mixed_scale_bivariate", "value": round(mi_z, 4),
                 "reference": round(true_mi, 4),
                 "pass": bool(ok_a),
                 "note": f"n={n}, seed={SEED}; |diff|<0.03 required"})
    rows.append({"check": "a_unstandardised_mixed_scale_bivariate", "value": round(mi_raw, 4),
                 "reference": round(true_mi, 4), "pass": bool(abs(mi_raw - true_mi) < 0.03),
                 "note": "same data, no z-score: max-norm eps dominated by the 1e2-scale "
                         "marginal; the raw-scale failure mode"})

    # ---------- (b) minimal eps-cancellation example (old cte_w2f) ------------
    def mini(seed, nn=240):
        rr = np.random.default_rng(seed)
        flow = np.zeros(nn)
        e = rr.standard_normal(nn)
        for i in range(1, nn):
            flow[i] = (0.6 * flow[i-1] + 2.0 * np.sin(2 * np.pi * i / 12)
                       + 0.5 * np.cos(2 * np.pi * i / 12) + e[i])
        flow = flow * 300.0                                   # ~1e3 (m3/s-like)
        wq = np.zeros(nn)
        h = rr.standard_normal(nn)
        for i in range(1, nn):
            wq[i] = 0.5 * wq[i-1] + 0.35 * (flow[i-1] / 300.0) \
                + 0.4 * np.sin(2 * np.pi * i / 12) + 0.15 * h[i]
        wq = wq * 0.01                                        # ~1e-2 (mg/L-like)
        dx, dy = detrend(flow), detrend(wq)                   # old pipeline: detrend only
        t = np.arange(nn)
        seas = np.column_stack([np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)])
        fp, wp2, sp = dx[:-1], dy[:-1], seas[:-1]
        yt_f, yt_w = dx[1:], dy[1:]
        cte_w2f_old = (ksg_mi(np.column_stack([wp2, fp, sp]), yt_f)
                       - ksg_mi(np.column_stack([fp, sp]), yt_f))
        cte_f2w_old = (ksg_mi(np.column_stack([fp, wp2, sp]), yt_w)
                       - ksg_mi(np.column_stack([wp2, sp]), yt_w))
        # z-scored (new convention)
        on = np.ones(nn, bool)
        zx, zy = zscore_valid(dx, on), zscore_valid(dy, on)
        zfp, zwp2, zsp = zx[:-1], zy[:-1], seas[:-1]
        cte_w2f_z = (ksg_mi(np.column_stack([zwp2, zfp, zsp]), zx[1:])
                     - ksg_mi(np.column_stack([zfp, zsp]), zx[1:]))
        cte_f2w_z = (ksg_mi(np.column_stack([zfp, zwp2, zsp]), zy[1:])
                     - ksg_mi(np.column_stack([zwp2, zsp]), zy[1:]))
        return cte_w2f_old, cte_f2w_old, cte_w2f_z, cte_f2w_z

    mc = np.array([mini(s) for s in range(1, 21)])
    n_zero_w2f = int((mc[:, 0] == 0.0).sum())
    print(f"(b) minimal cte_w2f-type example (flow 1e3 vs nutrient 1e-2, true coupling "
          f"flow->nutrient POSITIVE, 20 Monte-Carlo seeds):")
    print(f"    OLD unstandardised cte_w2f: exact 0.0 in {n_zero_w2f}/20 draws "
          f"(eps-cancellation: both joint spaces contain flow -> identical eps -> "
          f"identical marginal counts -> bitwise cancellation)")
    print(f"    OLD unstandardised cte_f2w: mean {mc[:,1].mean():+.4f} (range "
          f"[{mc[:,1].min():+.4f}, {mc[:,1].max():+.4f}]) -- strongly negative despite "
          f"true POSITIVE flow->nutrient coupling")
    print(f"    z-scored: cte_f2w mean {mc[:,3].mean():+.4f} (recovers the coupling, "
          f"0/20 exact zeros), cte_w2f mean {mc[:,2].mean():+.4f} (true value 0; "
          f"{int((mc[:,2]==0.0).sum())}/20 exact zeros)")
    rows.append({"check": "b_old_cte_w2f_exact_zeros_20seeds", "value": n_zero_w2f,
                 "reference": 20, "pass": n_zero_w2f == 20,
                 "note": "eps-cancellation mechanism reproduced: both KSG terms' joint "
                         "spaces contain the 1e3-scale flow -> identical eps -> "
                         "bitwise-identical digamma sums -> TE = 0.0 exactly"})
    rows.append({"check": "b_old_cte_f2w_mean", "value": round(float(mc[:, 1].mean()), 4),
                 "reference": 0.2, "pass": False,
                 "note": "old unstandardised f2w CTE is strongly negative although the true "
                         "flow->nutrient coupling is positive (spurious sign)"})
    rows.append({"check": "b_zscored_cte_f2w_mean", "value": round(float(mc[:, 3].mean()), 4),
                 "reference": None, "pass": bool(mc[:, 3].mean() > 0),
                 "note": "z-scored CTE recovers the true positive flow->nutrient coupling; "
                         "exact zeros: f2w %d/20, w2f %d/20"
                         % (int((mc[:, 3] == 0).sum()), int((mc[:, 2] == 0).sum()))})

    # ---------- (c) independent Gaussian, corrected estimator -----------------
    np.random.seed(SEED)  # fig3a convention: one stream, consecutive draws
    a = np.random.standard_normal(n)
    b = np.random.standard_normal(n)
    mi_c = ksg_mi(zscore(a), zscore(b))
    mi_c_naive = naive_kraskov_mi(zscore(a), zscore(b))
    print(f"(c) independent Gaussian, n={n}, seed {SEED}: corrected KSG MI = {mi_c:+.4f} nats "
          f"(true 0; manuscript's 0.007 nats to be replaced by this number); "
          f"naive Euclidean-joint MI = {mi_c_naive:+.4f} nats (the -0.25-nats failure)")
    rows.append({"check": "c_independent_gaussian_corrected", "value": round(float(mi_c), 4),
                 "reference": 0.0, "pass": bool(abs(mi_c) < 0.01),
                 "note": "replaces the manuscript's 0.007 nats"})
    rows.append({"check": "c_independent_gaussian_naive", "value": round(float(mi_c_naive), 4),
                 "reference": 0.0, "pass": False,
                 "note": "naive Euclidean-joint Kraskov for contrast"})

    out = pd.DataFrame(rows)
    out.to_csv(f"{T}/scale_invariance_checks.csv", index=False)
    print(f"written: {T}/scale_invariance_checks.csv")


# =================================================================== main

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"step1": step1, "step3": step3, "step4": step4, "step6": step6}
    if stage == "all":
        for name in ["step1", "step3", "step4", "step6"]:
            print(f"\n########## {name} ##########", flush=True)
            fns[name]()
    elif stage in fns:
        fns[stage]()
    else:
        raise SystemExit(f"unknown stage {stage}; use one of {list(fns)} or all")
