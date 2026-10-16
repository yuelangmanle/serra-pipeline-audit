#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrected global pipeline: std CCM / deseasonalized / M-CCM / TE tables.

Same-time CCM on calendar-true lags, KSG-1 mutual information with max-norm
joint distance, seeds fixed. Reads the GEMStat/GSIM extracts, writes the
core tables (combos, std_ccm, deseason, mccm, te, seasonal_r2).
"""
import os
import sys
import zipfile
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"
os.makedirs(T, exist_ok=True)

E, TAU, K, SEED, LIBF = 3, 1, 4, 42, 0.7


# ------------------------------------------------------------------ data IO

def read_gsim(gsim_id):
    z = zipfile.ZipFile(f"{ROOT}/data/raw/global/gsim/indices/GSIM_indices.zip")
    try:
        with z.open(f"GSIM_indices/TIMESERIES/monthly/{gsim_id}.mon") as f:
            lines = f.read().decode("utf-8", "ignore").split("\n")
    except KeyError:
        return None
    dates, vals = [], []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        p = line.replace(",", "").split()
        if len(p) >= 2:
            dates.append(p[0])
            try:
                v = float(p[1])
                vals.append(np.nan if v <= -900 or abs(v) > 1e6 else v)
            except Exception:
                vals.append(np.nan)
    df = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "flow": vals}).dropna()
    if df.empty:
        return None
    df["ym"] = df.date.dt.to_period("M")
    return df.groupby("ym")["flow"].mean()


def build_wq_total(qids):
    wq = {}
    n_frac_only = 0
    for param, fn, codes, hi in [
        ("TN", "Other_Nitrogen.csv", ["TN", "TOTN"], 100.0),
        ("TP", "Phosphorus.csv", ["TP", "TOTP"], 10.0),
    ]:
        df = pd.read_csv(f"{ROOT}/data/raw/global/gemstat/extracted/{fn}",
                         encoding="latin-1", low_memory=False)
        all_stations = set(df["GEMS Station Number"])
        df = df[df["GEMS Station Number"].isin(qids) & df["Parameter Code"].isin(codes)]
        drop_stations = (all_stations & qids) - set(df["GEMS Station Number"])
        n_frac_only += len(drop_stations)
        df["date"] = pd.to_datetime(df["Sample Date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["ym"] = df.date.dt.to_period("M")
        df["value"] = pd.to_numeric(df["Value"], errors="coerce")
        df.loc[df["Value Flags"] == "<", "value"] = df.loc[df["Value Flags"] == "<", "value"] / 2
        df = df[(df.value >= 0) & (df.value <= hi)]

        def iqr_ok(g):
            med = g.median()
            q1, q3 = g.quantile(0.25), g.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                return pd.Series(True, index=g.index)  # degenerate IQR: no filtering
            return (g >= med - 3 * iqr) & (g <= med + 3 * iqr)
        df = df[df.groupby("GEMS Station Number")["value"].transform(iqr_ok).fillna(False)]
        m = df.groupby(["GEMS Station Number", "ym"])["value"].mean().reset_index()
        for _, r in m.iterrows():
            wq.setdefault(r["GEMS Station Number"], {}).setdefault(param, {})[str(r["ym"])] = r["value"]
    return wq, n_frac_only


def extend_short_gaps(v, max_gap=2):
    """Extend a boolean validity pattern across interior gaps of length
    <= max_gap. Pure mask rule: the same rule fill_short_gaps_pair applies
    to the observation's support, reused for rotated supports."""
    v = v.copy()
    n = len(v)
    i = 0
    while i < n:
        if not v[i]:
            j = i
            while j < n and not v[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                v[i:j] = True
            i = j
        else:
            i += 1
    return v


def rotated_support(obs_base, obs_target, k, fill=True):
    """Analysis support of the k-year whole-block rotation of the base record.

    The base series' own observation pattern rotates with its values; the
    target record is untouched, so the joint pattern is the rotated base
    pattern intersected with the target's own. With fill=True the pipeline's
    short-gap rule is applied to that joint pattern -- the same rule the
    observation's support gets -- and the result is intersected with the
    rotated pattern of the base record's own usable months (observed, or a
    fill across at most two months). The intersection keeps every analysed
    month's base value of the provenance class the observation draws on:
    a real observation or a fill across at most two months. It matters only
    at the rotation seam and the trailing partial-block junction, where a
    long base gap can be truncated into a positionally short joint gap;
    without it those few months would carry long-gap interpolants that the
    observation's support never contains.
    """
    v = year_shift(obs_base, k) & obs_target
    if not fill:
        return v
    usable_base = extend_short_gaps(obs_base)
    return extend_short_gaps(v) & year_shift(usable_base, k)


def fill_short_gaps_pair(x, y, valid_obs, max_gap=2):
    """Fill interior common-validity gaps of length <= max_gap; each series is
    linearly interpolated from its own observed values. Returns (x, y, mask)."""
    v = extend_short_gaps(valid_obs, max_gap)
    idx = np.arange(len(x))
    x_ok = ~np.isnan(x); y_ok = ~np.isnan(y)
    xf = np.interp(idx, idx[x_ok], x[x_ok])
    yf = np.interp(idx, idx[y_ok], y[y_ok])
    return xf, yf, v


def merged_calendar(flow, wq_series):
    start = max(flow.index.min(), min(pd.Period(k, freq="M") for k in wq_series))
    end = min(flow.index.max(), max(pd.Period(k, freq="M") for k in wq_series))
    cal = pd.period_range(start, end, freq="M")
    f = flow.reindex(cal)
    w = pd.Series(wq_series)
    w.index = pd.PeriodIndex(w.index, freq="M")
    w = w.reindex(cal)
    valid = (f.notna() & w.notna()).values
    return f.values.astype(float), w.values.astype(float), valid, cal


# ------------------------------------------------------------------ CCM core

def valid_targets(valid, E, tau, ahead):
    n = len(valid)
    out = []
    for t in range((E - 1) * tau, n):
        tt = t + ahead
        if tt >= n or not valid[tt]:
            continue
        if all(valid[t - j * tau] for j in range(E)):
            out.append(t)
    return np.array(out, dtype=int)


def ccm_run(base, target, valid, ahead=0, controls=None,
            lib_fraction=LIBF, seed=SEED, E=E, tau=TAU, k=K):
    """Simplex cross-map on calendar-valid embeddings (reindexed grid)."""
    idx = valid_targets(valid, E, tau, ahead)
    if len(idx) < 40:
        return np.nan, np.nan, len(idx)
    Mv = np.array([[base[t - j * tau] for j in range(E)] +
                   (list(controls[t]) if controls is not None else []) for t in idx])
    tgt = np.asarray([target[t + ahead] for t in idx])
    rng = np.random.RandomState(seed)
    perm = idx.copy(); rng.shuffle(perm)
    L = int(len(perm) * lib_fraction)
    libset = set(perm[:L].tolist())
    lib_m = np.array([Mv[i] for i, t in enumerate(idx) if t in libset])
    lib_t = np.array([tgt[i] for i, t in enumerate(idx) if t in libset])
    pred_m = np.array([Mv[i] for i, t in enumerate(idx) if t not in libset])
    pred_t = np.array([tgt[i] for i, t in enumerate(idx) if t not in libset])
    if len(lib_m) < k + 5 or len(pred_m) < 10:
        return np.nan, np.nan, len(idx)
    tree = cKDTree(lib_m)
    d, ix = tree.query(pred_m, k=k)
    w = np.exp(-d / np.maximum(d[:, [0]], 1e-10))
    w /= w.sum(axis=1, keepdims=True)
    P = np.sum(w * lib_t[ix], axis=1)
    return stats.pearsonr(P, pred_t)[0], len(pred_t), len(idx)


def ccm_bidir(x, y, valid, ahead=0, controls=None, lib_fraction=LIBF, seed=SEED):
    f2w, nf, ne = ccm_run(y, x, valid, ahead, controls, lib_fraction, seed)  # WQ manifold -> flow
    w2f, nw, _ = ccm_run(x, y, valid, ahead, controls, lib_fraction, seed)   # flow manifold -> WQ
    return f2w, w2f, ne


# ------------------------------------------------- corrected KSG-1 MI / TE

def ksg_mi(X, Y, k=3, strict=True):
    """Kraskov Algorithm 1 for I(X;Y), max-norm geometry.

    strict=True (default) counts marginal points strictly inside the joint
    k-distance, which is the rule of the original paper and the one the
    manuscript's primary numbers use. strict=False counts points within the
    closed ball up to eps (the earlier behavior; kept so the two conventions
    can be compared, e.g. 52.2% (inclusive) vs 45.6% (strict) for the
    corrected unconditional TE).
    X, Y: (n, d) arrays; 1-D inputs are reshaped.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    n = len(X)
    joint = np.hstack([X, Y])
    d, _ = cKDTree(joint).query(joint, k=k + 1, p=np.inf)
    eps = d[:, k]
    if strict:
        # marginal distances by max-norm; count strictly below eps
        def counts(A):
            out = np.empty(n, dtype=int)
            for i in range(n):
                dmax = np.max(np.abs(A - A[i]), axis=1)
                out[i] = int(np.sum(dmax < eps[i])) - 1
            return np.maximum(out, 0)
        nx, ny = counts(X), counts(Y)
    else:
        if X.shape[1] == 1:
            sx = np.sort(X[:, 0])
            nx = np.maximum(np.searchsorted(sx, X[:, 0] + eps) - np.searchsorted(sx, X[:, 0] - eps) - 1, 0)
        else:
            tx = cKDTree(X)
            nx = np.array([max(len(tx.query_ball_point(X[i], e, p=np.inf)) - 1, 0)
                           for i, e in enumerate(eps)])
        if Y.shape[1] == 1:
            sy = np.sort(Y[:, 0])
            ny = np.maximum(np.searchsorted(sy, Y[:, 0] + eps) - np.searchsorted(sy, Y[:, 0] - eps) - 1, 0)
        else:
            ty = cKDTree(Y)
            ny = np.array([max(len(ty.query_ball_point(Y[i], e, p=np.inf)) - 1, 0)
                           for i, e in enumerate(eps)])
    return digamma(k) + digamma(n) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))


def seasonal_controls(periods):
    months = np.array([p.month for p in periods])
    return np.column_stack([np.sin(2 * np.pi * months / 12),
                            np.cos(2 * np.pi * months / 12)])


def seasonal_r2(x, months):
    sin_m = np.sin(2 * np.pi * months / 12)
    cos_m = np.cos(2 * np.pi * months / 12)
    X = np.column_stack([np.ones(len(x)), sin_m, cos_m])
    beta, _, _, _ = np.linalg.lstsq(X, x, rcond=None)
    res = np.sum((x - X @ beta) ** 2)
    tot = np.sum((x - np.mean(x)) ** 2)
    return 1 - res / tot if tot > 0 else np.nan


def detrend(a):
    t = np.arange(len(a))
    s, _, _, _, _ = stats.linregress(t, a)
    return a - s * t


def year_shift(x, k):
    """Rotate a series by whole calendar years, exactly k years.

    Only the complete-year blocks are rotated (cyclically, by k of them); a
    trailing partial year, when the record's length is not a multiple of
    twelve, stays in place. Every position therefore keeps the calendar
    month it had in the observation -- a short tail entering the rotation
    would shift every later block off its month phase -- so the seasonal
    phase is preserved exactly at any record length, with no wrap-around at
    the array boundary.
    """
    n = len(x)
    y = n // 12
    head = x[:12 * y].reshape(y, 12)
    head = np.roll(head, k, axis=0).reshape(-1)
    return np.concatenate([head, x[12 * y:]])


# ------------------------------------------------------------------ main

def main():
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv")
    res = res[res.n_months >= 60].reset_index(drop=True)
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv")
    wq, n_frac_only = build_wq_total(set(qual.gems_id))
    print(f"candidate combos={len(res)}; total-code-only stations retained, "
          f"fractional-only dropped: {n_frac_only}", flush=True)

    combos, std_rows, des_rows, mccm_rows, r2_rows = [], [], [], [], []
    cache, done = {}, 0
    for _, r in res.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        key = (gid, sid, param)
        if key not in cache:
            flow = read_gsim(sid)
            series = wq.get(gid, {}).get(param, {})
            if flow is None or len(series) < 60:
                cache[key] = None
                continue
            x, y, valid, cal = merged_calendar(flow, series)
            cache[key] = (x, y, valid, cal) if valid.sum() >= 60 else None
        c = cache[key]
        if c is None:
            continue
        x_raw, y_raw, valid_obs, cal = c
        # interior gaps <= 2 months linearly interpolated (calendar-true months
        # preserved; longer gaps remain excluded). Documented in Methods.
        x, y, valid = fill_short_gaps_pair(x_raw, y_raw, valid_obs, max_gap=2)
        # minimum-information gate: drop degenerate series (IQR=0 filtering can
        # leave near-constant DL-dominated records that break k-NN estimation)
        if np.nanstd(y) < 1e-6 or len(np.unique(np.round(y, 6))) < 10:
            cache[key] = None
            continue
        months = np.array([p.month for p in cal])
        dx, dy = detrend(x), detrend(y)

        y_years = len(cal) // 12
        combos.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                       "n_calendar": len(cal), "n_valid": int(valid.sum()),
                       "n_full_years": y_years,
                       "n_distinct_rotations": min(y_years, 11),
                       "identity_rotation": bool(y_years and
                                                 any(k % y_years == 0
                                                     for k in range(1, 12)))})

        for lf in (0.3, 0.5, 0.7):
            f2w, w2f, ne = ccm_bidir(x, y, valid, lib_fraction=lf)
            std_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                             "lib": lf, "f2w": f2w, "w2f": w2f, "n_emb": ne})

        def sub_clim(a):
            a2 = a.copy()
            for m in np.unique(months[valid]):
                sel = (months == m) & valid
                a2[sel] -= a2[sel].mean()
            return a2
        f2w_d, w2f_d, _ = ccm_bidir(detrend(sub_clim(x)), detrend(sub_clim(y)), valid)
        des_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                         "f2w": f2w_d, "w2f": w2f_d})

        z = seasonal_controls(cal)
        f2w_m, w2f_m, ne = ccm_bidir(dy, dx, valid, ahead=1, controls=z)
        f2w_s, w2f_s, _ = ccm_bidir(dx, dy, valid, ahead=1)
        mccm_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                          "mccm_f2w": f2w_m, "mccm_w2f": w2f_m,
                          "std_f2w": f2w_s, "std_w2f": w2f_s, "n_emb": ne})

        r2_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                        "r2_flow_nodetrend": seasonal_r2(x[valid], months[valid]),
                        "r2_flow": seasonal_r2(dx[valid], months[valid]),
                        "r2_wq": seasonal_r2(dy[valid], months[valid])})

        done += 1
        if done % 50 == 0:
            print(f"done {done}", flush=True)

    pd.DataFrame(combos).to_csv(f"{T}/combos.csv", index=False)
    pd.DataFrame(std_rows).to_csv(f"{T}/std_ccm.csv", index=False)
    pd.DataFrame(des_rows).to_csv(f"{T}/deseason.csv", index=False)
    pd.DataFrame(mccm_rows).to_csv(f"{T}/mccm_naive_labels_swapped.csv", index=False)
    pd.DataFrame(r2_rows).to_csv(f"{T}/seasonal_r2.csv", index=False)
    print(f"core layer done: {len(combos)} combos", flush=True)


if __name__ == "__main__":
    main()
