#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-factor ablation ladder: naive -> corrected, one change per rung.

Tells us which pipeline choice moved the directional share. Each rung keeps
n, series and library settings identical to the previous one.
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
sys.path.insert(0, HERE)
import pipeline as gp  # noqa: E402  (import-safe: __main__ guarded)

E, TAU, SEED, LIBF = 3, 1, 42, 0.7
OUT = f"{T}/ablation_ladder.csv"

# ------------------------------------------------------------------ naive layer
# Verbatim legacy implementations: the true-xmap check and the full same-time comparison.


def naive_read_gsim(gsim_id):
    """Legacy naive GSIM reader (skips header line, returns per-month rows)."""
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
    """Legacy naive composite WQ builder: ALL species codes averaged per station-month."""
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
    """Row-index pairing: inner join on observed months, sort -> lag arithmetic
    on packed rows crosses calendar gaps (the naive 'row-index lag')."""
    rec = [(ym, v) for ym, v in series_dict.items()]
    wqd = pd.DataFrame(rec, columns=["ym", "wq"])
    wqd["ym"] = pd.PeriodIndex(wqd.ym, freq="M")
    mg = pd.merge(flow_df, wqd, on="ym", how="inner").sort_values("ym")
    return mg


def naive_simplex(manifold, target, lib_idx, pred_idx, E=3, tau=1):
    """Naive same-time simplex cross-map (verbatim semantics of
    the legacy same-time comparison: k=E+1, exp(-d/d0) weights, lag-0 target)."""
    n = len(target)
    mi = (E - 1) * tau

    def emb(idx):
        V, o = [], []
        for i in idx:
            if i >= mi and i < n:
                V.append([manifold[i - j * tau] for j in range(E)])
                o.append(i)
        return np.array(V), np.array(o)

    lv, lok = emb(lib_idx)
    pv, pok = emb(pred_idx)
    k = E + 1
    if len(lv) < k + 1 or len(pv) == 0:
        return np.nan
    tree = cKDTree(lv)
    P, A = [], []
    for i, q in enumerate(pv):
        d, ix = tree.query(q, k=k + 1)
        v = d > 1e-10
        if v.sum() < k:
            continue
        ix, d = ix[v][:k], d[v][:k]
        w = np.exp(-d / d[0]) if d[0] > 1e-10 else None
        if w is None:
            w = np.zeros(k)
            w[0] = 1.0
        w /= w.sum()
        P.append(np.sum(w * target[lok[ix]]))
        A.append(target[pok[i]])  # lag=0 same-time target
    if len(P) < 5:
        return np.nan
    return stats.pearsonr(P, A)[0]


def naive_xmap_bidir(x, y, lib_fraction=LIBF, seed=SEED):
    """Naive bidirectional same-time CCM (legacy same-time comparison, lf=0.7):
    z-score after detrend, shuffle row indices, lib = first int(n*lf) rows."""
    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)
    n = len(x)
    mi = (E - 1) * TAU
    rng = np.random.RandomState(seed)
    idx = np.arange(mi, n - TAU)
    rng.shuffle(idx)
    lib = idx[:int(n * lib_fraction)]
    pred = idx[int(n * lib_fraction):]
    f2w = naive_simplex(y, x, lib, pred)  # WQ manifold -> flow target
    w2f = naive_simplex(x, y, lib, pred)  # flow manifold -> WQ target
    return f2w, w2f


def naive_kraskov_mi(x, y, k=3):
    """Naive (biased) Kraskov estimator, preserved verbatim from the
    original transfer-entropy code: Euclidean joint-space k-NN distance +
    KSG marginal counting formulas (the paper's
    'mixed implementation'; returns -0.25 nats on independent Gaussian data)."""
    n = len(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    dists, _ = tree_xy.query(xy, k=k + 1)          # Euclidean joint distance
    eps = dists[:, k]
    nx = np.array([len(tree_x.query_ball_point(xi, eps_i)) - 1 for xi, eps_i in zip(x, eps)])
    ny = np.array([len(tree_y.query_ball_point(yi, eps_i)) - 1 for yi, eps_i in zip(y, eps)])
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))
    return mi


# ------------------------------------------------------------------ shared helpers


def uncond_te(x, y, mi_fn, lag=1):
    """Unconditional TE(X->Y) = I(Y_t; X_{t-lag}) - I(Y_t; Y_{t-lag}) over ALL
    packed row pairs (row-index semantics; no calendar mask)."""
    y_t = y[lag:]
    x_past = x[:-lag]
    y_past = y[:-lag]
    return mi_fn(x_past, y_t) - mi_fn(y_past, y_t)


def zscore(a):
    return (a - a.mean()) / (a.std() + 1e-10)


def share_stats(df, a, b, method):
    """flow-dominant share + n + two-sided p on non-NaN pairs."""
    d = df.dropna(subset=[a, b])
    n = len(d)
    k = int((d[a] > d[b]).sum())
    share = k / n if n else np.nan
    if method == "TE":
        p = stats.binomtest(k, n, 0.5).pvalue if n else np.nan
        ptype = "binom_two_sided"
    else:
        try:
            p = stats.wilcoxon(d[a], d[b]).pvalue if n else np.nan
        except ValueError:
            p = np.nan
        ptype = "wilcoxon_two_sided"
    return n, k, share, p, ptype


# ------------------------------------------------------------------ stage runners


def build_series_rowindex(flow_df, series_dict, min_len=60):
    """Naive row-index series: detrended flow/WQ on packed rows."""
    if flow_df is None or len(series_dict) < 30:
        return None
    mg = naive_merge_rows(flow_df, series_dict)
    if len(mg) < min_len:
        return None
    x = gp.detrend(mg.flow.values.astype(float))
    y = gp.detrend(mg.wq.values.astype(float))
    return x, y


def build_series_calendar(sid, series_dict, min_valid=60):
    """Corrected series: calendar grid + gap-fill + validity mask."""
    flow = gp.read_gsim(sid)
    if flow is None or len(series_dict) < 60:
        return None
    x_raw, y_raw, valid_obs, cal = gp.merged_calendar(flow, series_dict)
    if valid_obs.sum() < min_valid:
        return None
    x, y, valid = gp.fill_short_gaps_pair(x_raw, y_raw, valid_obs, max_gap=2)
    if np.nanstd(y) < 1e-6 or len(np.unique(np.round(y, 6))) < 10:
        return None
    return x, y, valid, cal


def main():
    print(f"ablation ladder | seed={SEED} lib={LIBF} E={E} tau={TAU}", flush=True)

    # ---------------- combo sets ----------------
    res = pd.read_csv(f"{T}/ccm_analysis_results.csv")
    res = res[res.n_months >= 60].reset_index(drop=True)          # naive candidate set
    qual = pd.read_csv(f"{ROOT}/data/interim/ccm_qualified_stations.csv")
    qids = set(qual.gems_id)
    combos316 = pd.read_csv(f"{T}/combos.csv")                 # corrected combo set
    keys316 = list(map(tuple, combos316[["gems_id", "gsim_id", "parameter"]].values))

    wq_all = naive_build_wq(qids)      # naive composite species (legacy implementation)
    wq_tot, n_frac_only = gp.build_wq_total(qids)  # corrected total-species codes
    print(f"naive candidate combos={len(res)}; corrected combos={len(keys316)}; "
          f"fractional-only stations dropped by species correction: {n_frac_only}", flush=True)

    rows = []
    cache_naive_all, cache_naive_tot = {}, {}   # row-index series caches (detrended)
    c0_rows, c1_rows, c2_rows = [], [], []
    t0_rows, t1_rows, t2_rows, t3_rows = [], [], [], []

    # ================= CCM ladder =================

    # ---- C0: all-species + row-index + naive CCM (n=491, anchor 60.1%)
    print("\n[C0] all-species + row-index + naive CCM ...", flush=True)
    done = 0
    for _, r in res.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        key = (gid, sid, param)
        if key not in cache_naive_all:
            series = {ym: v[param] for ym, v in wq_all.get(gid, {}).items() if param in v}
            cache_naive_all[key] = build_series_rowindex(naive_read_gsim(sid), series)
        sc = cache_naive_all[key]
        if sc is None:
            continue
        x, y = sc
        f2w, w2f = naive_xmap_bidir(x, y)
        c0_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                        "f2w": f2w, "w2f": w2f})
        done += 1
        if done % 100 == 0:
            print(f"  C0 done {done}", flush=True)
    c0 = pd.DataFrame(c0_rows)
    n0, k0, s0, p0, _ = share_stats(c0, "f2w", "w2f", "CCM")
    p0_one = stats.wilcoxon(c0.dropna()["f2w"], c0.dropna()["w2f"], alternative="greater").pvalue
    anchor = pd.read_csv(f"{T}/true_xmap_sametime_full.csv")
    akeys = set(map(tuple, anchor[["gems_id", "gsim_id", "parameter"]].values))
    c0keys = set(map(tuple, c0[["gems_id", "gsim_id", "parameter"]].dropna().values))
    print(f"  C0: n={n0} share={s0:.4f} wilcoxon 2p={p0:.3e} 1p={p0_one:.3e} | "
          f"key-set matches anchor: {c0keys == akeys} "
          f"({len(c0keys)} vs {len(akeys)})", flush=True)
    rows.append({"stage": "C0", "method": "CCM_std",
                 "factor_changed": "baseline (naive pipeline: all-species codes + row-index lags + naive CCM)",
                 "n": n0, "flow_dom_share": round(s0, 4), "p_two_sided": p0,
                 "note": f"anchor check: paper naive std CCM 60.1% one-sided p=2.6e-6 -> "
                         f"replicated {s0*100:.1f}% one-sided p={p0_one:.2e}; "
                         f"wilcoxon_two_sided; key-set identical to true_xmap_sametime_full.csv={c0keys == akeys}"})

    # ---- C1: total-species + row-index + naive CCM (n=316, species factor)
    print("\n[C1] total-species + row-index + naive CCM ...", flush=True)
    done = 0
    for gid, sid, param in keys316:
        key = (gid, sid, param)
        if key not in cache_naive_tot:
            series = wq_tot.get(gid, {}).get(param, {})
            cache_naive_tot[key] = build_series_rowindex(naive_read_gsim(sid), series)
        sc = cache_naive_tot[key]
        if sc is None:
            continue
        x, y = sc
        f2w, w2f = naive_xmap_bidir(x, y)
        c1_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                        "f2w": f2w, "w2f": w2f})
        done += 1
        if done % 100 == 0:
            print(f"  C1 done {done}", flush=True)
    c1 = pd.DataFrame(c1_rows)
    n1, k1, s1, p1, _ = share_stats(c1, "f2w", "w2f", "CCM")
    print(f"  C1: n={n1} share={s1:.4f} p={p1:.3e}", flush=True)
    rows.append({"stage": "C1", "method": "CCM_std",
                 "factor_changed": "species codes: all-species -> total-only "
                                   "(sample contraction 491 -> 316 is part of the correction)",
                 "n": n1, "flow_dom_share": round(s1, 4), "p_two_sided": p1,
                 "note": f"isolation of species factor vs C0: {100*(s1-s0):+.1f} pp; "
                         f"indexing & CCM machinery held at naive; wilcoxon_two_sided"})

    # ---- C2: total-species + calendar + corrected CCM (n=316, anchor 51.9%)
    # NOTE: as in the corrected pipeline, the "standard CCM" rows use the gap-filled series
    # WITHOUT detrending (only deseason/M-CCM/TE rows use detrended dx, dy);
    # the detrended variant is computed as a robustness note below.
    print("\n[C2] total-species + calendar + corrected CCM ...", flush=True)
    done = 0
    for gid, sid, param in keys316:
        sc = build_series_calendar(sid, wq_tot.get(gid, {}).get(param, {}))
        if sc is None:
            continue
        x, y, valid, cal = sc
        f2w, w2f, _ = gp.ccm_bidir(x, y, valid, lib_fraction=LIBF, seed=SEED)
        f2w_dt, w2f_dt, _ = gp.ccm_bidir(gp.detrend(x), gp.detrend(y), valid,
                                          lib_fraction=LIBF, seed=SEED)
        c2_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                        "f2w": f2w, "w2f": w2f, "f2w_dt": f2w_dt, "w2f_dt": w2f_dt})
        done += 1
        if done % 100 == 0:
            print(f"  C2 done {done}", flush=True)
    c2 = pd.DataFrame(c2_rows)
    n2, k2, s2, p2, _ = share_stats(c2, "f2w", "w2f", "CCM")
    # exact-match check vs the published corrected table
    ref = pd.read_csv(f"{T}/std_ccm.csv")
    ref = ref[ref.lib == LIBF][["gems_id", "gsim_id", "parameter", "f2w", "w2f"]]
    mg = c2.merge(ref, on=["gems_id", "gsim_id", "parameter"], suffixes=("", "_ref"))
    dmax = (max(float(np.nanmax(np.abs(mg["f2w"] - mg["f2w_ref"]))),
                float(np.nanmax(np.abs(mg["w2f"] - mg["w2f_ref"]))))
            if len(mg) else np.nan)
    print(f"  C2: n={n2} share={s2:.4f} p={p2:.3e} | max|diff| vs std_ccm lib{LIBF}: {dmax:.2e} "
          f"({len(mg)}/{n2} matched)", flush=True)
    _, _, s2_dt, _, _ = share_stats(c2, "f2w_dt", "w2f_dt", "CCM")
    print(f"  C2 detrended variant: share={s2_dt:.4f}", flush=True)
    rows.append({"stage": "C2", "method": "CCM_std",
                 "factor_changed": "indexing: row-arithmetic -> calendar-true lags "
                                   "(reindexed monthly grid + validity mask + gap<=2 fill)",
                 "n": n2, "flow_dom_share": round(s2, 4), "p_two_sided": p2,
                 "note": f"anchor check: std CCM 51.9% (two-sided p=0.81) -> replicated "
                         f"{s2*100:.1f}% p={p2:.2f}; max|diff| vs the reference std_ccm table={dmax:.1e}; "
                         f"std-CCM rows follow the corrected-pipeline convention (gap-filled, undetrended; "
                         f"detrended variant share={100*s2_dt:.1f}%); "
                         f"isolation of calendar factor vs C1: {100*(s2-s1):+.1f} pp; "
                         f"wilcoxon_two_sided"})

    # ================= TE ladder =================

    # ---- T0: all-species + row-index + naive MI estimator (n=491)
    print("\n[T0] all-species + row-index + naive Kraskov MI ...", flush=True)
    done = 0
    for _, r in c0.iterrows():
        key = (r.gems_id, r.gsim_id, r.parameter)
        x, y = cache_naive_all[key]
        x, y = zscore(x), zscore(y)   # naive s6 pipeline standardizes before TE
        te_f2w = uncond_te(x, y, naive_kraskov_mi, lag=1)
        te_w2f = uncond_te(y, x, naive_kraskov_mi, lag=1)
        t0_rows.append({"gems_id": r.gems_id, "gsim_id": r.gsim_id,
                        "parameter": r.parameter, "te_f2w": te_f2w, "te_w2f": te_w2f})
        done += 1
        if done % 100 == 0:
            print(f"  T0 done {done}", flush=True)
    t0 = pd.DataFrame(t0_rows)
    n_t0, k_t0, s_t0, p_t0, _ = share_stats(t0, "te_f2w", "te_w2f", "TE")
    # sanity: paper's naive uncond TE was 40.4% on the 272-combo TE subsample
    v26 = pd.read_csv(f"{T}/transfer_entropy.csv")
    cte_keys = set(map(tuple, v26[["gems_id", "gsim_id", "parameter"]].values))
    sub = t0[[tuple(r) in cte_keys
              for r in t0[["gems_id", "gsim_id", "parameter"]].values]]
    ns, _, ss, _, _ = share_stats(sub, "te_f2w", "te_w2f", "TE")
    print(f"  T0: n={n_t0} share={s_t0:.4f} binom 2p={p_t0:.3e} | "
          f"272-combo subsample share={ss:.4f} (paper naive uncond TE 40.4%)", flush=True)
    rows.append({"stage": "T0", "method": "TE_uncond",
                 "factor_changed": "baseline (naive pipeline: all-species codes + row-index lags + naive MI)",
                 "n": n_t0, "flow_dom_share": round(s_t0, 4), "p_two_sided": p_t0,
                 "note": f"naive estimator = Euclidean-joint Kraskov (the original transfer-entropy code); "
                         f"paper naive uncond TE on the 272-combo subsample 40.4% -> replicated {ss*100:.1f}% "
                         f"(n={ns}); binom_two_sided"})

    # ---- T1: all-species + row-index + corrected KSG-1 MI (n=491; estimator factor)
    print("\n[T1] all-species + row-index + corrected KSG-1 MI ...", flush=True)
    done = 0
    for _, r in c0.iterrows():
        key = (r.gems_id, r.gsim_id, r.parameter)
        x, y = cache_naive_all[key]
        x, y = zscore(x), zscore(y)
        te_f2w = uncond_te(x, y, gp.ksg_mi, lag=1)
        te_w2f = uncond_te(y, x, gp.ksg_mi, lag=1)
        t1_rows.append({"gems_id": r.gems_id, "gsim_id": r.gsim_id,
                        "parameter": r.parameter, "te_f2w": te_f2w, "te_w2f": te_w2f})
        done += 1
        if done % 100 == 0:
            print(f"  T1 done {done}", flush=True)
    t1 = pd.DataFrame(t1_rows)
    n_t1, k_t1, s_t1, p_t1, _ = share_stats(t1, "te_f2w", "te_w2f", "TE")
    print(f"  T1: n={n_t1} share={s_t1:.4f} binom 2p={p_t1:.3e}", flush=True)
    rows.append({"stage": "T1", "method": "TE_uncond",
                 "factor_changed": "MI estimator: naive (Euclidean joint, unclamped counts) -> KSG-1 max-norm",
                 "n": n_t1, "flow_dom_share": round(s_t1, 4), "p_two_sided": p_t1,
                 "note": f"isolation of estimator factor vs T0 (identical sample & series): "
                         f"{100*(s_t1-s_t0):+.1f} pp; binom_two_sided"})

    # ---- T2: total-species + row-index + corrected KSG-1 MI (n=316; species factor)
    print("\n[T2] total-species + row-index + corrected KSG-1 MI ...", flush=True)
    done = 0
    for gid, sid, param in keys316:
        key = (gid, sid, param)
        sc = cache_naive_tot.get(key)
        if sc is None:
            continue
        x, y = sc
        x, y = zscore(x), zscore(y)
        te_f2w = uncond_te(x, y, gp.ksg_mi, lag=1)
        te_w2f = uncond_te(y, x, gp.ksg_mi, lag=1)
        t2_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                        "te_f2w": te_f2w, "te_w2f": te_w2f})
        done += 1
        if done % 100 == 0:
            print(f"  T2 done {done}", flush=True)
    t2 = pd.DataFrame(t2_rows)
    n_t2, k_t2, s_t2, p_t2, _ = share_stats(t2, "te_f2w", "te_w2f", "TE")
    print(f"  T2: n={n_t2} share={s_t2:.4f} binom 2p={p_t2:.3e}", flush=True)
    rows.append({"stage": "T2", "method": "TE_uncond",
                 "factor_changed": "species codes: all-species -> total-only "
                                   "(sample contraction 491 -> 316 is part of the correction)",
                 "n": n_t2, "flow_dom_share": round(s_t2, 4), "p_two_sided": p_t2,
                 "note": f"isolation of species factor vs T1: {100*(s_t2-s_t1):+.1f} pp; "
                         f"binom_two_sided"})

    # ---- T3: total-species + calendar + corrected KSG-1 MI (n=316, anchor 41.5%)
    print("\n[T3] total-species + calendar + corrected KSG-1 MI ...", flush=True)
    done = 0
    for gid, sid, param in keys316:
        sc = build_series_calendar(sid, wq_tot.get(gid, {}).get(param, {}))
        if sc is None:
            continue
        x_raw, y_raw, valid, cal = sc
        dx, dy = gp.detrend(x_raw), gp.detrend(y_raw)
        z = gp.seasonal_controls(cal)
        idx = np.array([t for t in range(1, len(dx)) if valid[t] and valid[t - 1]], dtype=int)
        if len(idx) < 40:
            continue
        yt, yp = dy[idx], dy[idx - 1]
        xp = dx[idx - 1]
        te_f2w = gp.ksg_mi(xp, yt) - gp.ksg_mi(yp, yt)
        xt = dx[idx]
        te_w2f = gp.ksg_mi(yp, xt) - gp.ksg_mi(xp, xt)
        t3_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                        "te_f2w": te_f2w, "te_w2f": te_w2f})
        done += 1
        if done % 100 == 0:
            print(f"  T3 done {done}", flush=True)
    t3 = pd.DataFrame(t3_rows)
    n_t3, k_t3, s_t3, p_t3, _ = share_stats(t3, "te_f2w", "te_w2f", "TE")
    ref_te = pd.read_csv(f"{T}/te.csv")[["gems_id", "gsim_id", "parameter", "te_f2w", "te_w2f"]]
    mgt = t3.merge(ref_te, on=["gems_id", "gsim_id", "parameter"], suffixes=("", "_ref"))
    dtmax = max(float(np.nanmax(np.abs(mgt["te_f2w"] - mgt["te_f2w_ref"]))),
                float(np.nanmax(np.abs(mgt["te_w2f"] - mgt["te_w2f_ref"])))) if len(mgt) else np.nan
    print(f"  T3: n={n_t3} share={s_t3:.4f} binom 2p={p_t3:.3e} | max|diff| vs te: {dtmax:.2e} "
          f"({len(mgt)}/{n_t3} matched)", flush=True)
    rows.append({"stage": "T3", "method": "TE_uncond",
                 "factor_changed": "indexing: row-arithmetic -> calendar-true lags "
                                   "(reindexed monthly grid + validity mask + gap<=2 fill)",
                 "n": n_t3, "flow_dom_share": round(s_t3, 4), "p_two_sided": p_t3,
                 "note": f"anchor check: unconditional TE 41.5% -> replicated {s_t3*100:.1f}%; "
                         f"max|diff| vs the reference te table={dtmax:.1e}; isolation of calendar factor vs T2: "
                         f"{100*(s_t3-s_t2):+.1f} pp; binom_two_sided"})

    # ================= output =================
    out = pd.DataFrame(rows)[["stage", "method", "factor_changed", "n",
                              "flow_dom_share", "p_two_sided", "note"]]
    out.to_csv(OUT, index=False)

    print("\n=== ABLATION LADDER (flow-dominant share, two-sided p) ===", flush=True)
    print(f"{'stage':6s} {'method':10s} {'n':>5s} {'share':>8s} {'p':>10s}  factor", flush=True)
    for _, r in out.iterrows():
        print(f"{r.stage:6s} {r.method:10s} {r.n:5d} {100*r.flow_dom_share:7.1f}% "
              f"{r.p_two_sided:10.3g}  {r.factor_changed}", flush=True)
    print("\nPer-factor effects (percentage points):", flush=True)
    print(f"  CCM species  (C0->C1): {100*(s1-s0):+.1f} pp   ({s0*100:.1f} -> {s1*100:.1f})", flush=True)
    print(f"  CCM calendar (C1->C2): {100*(s2-s1):+.1f} pp   ({s1*100:.1f} -> {s2*100:.1f})", flush=True)
    print(f"  TE  estimator(T0->T1): {100*(s_t1-s_t0):+.1f} pp   ({s_t0*100:.1f} -> {s_t1*100:.1f})", flush=True)
    print(f"  TE  species  (T1->T2): {100*(s_t2-s_t1):+.1f} pp   ({s_t1*100:.1f} -> {s_t2*100:.1f})", flush=True)
    print(f"  TE  calendar (T2->T3): {100*(s_t3-s_t2):+.1f} pp   ({s_t2*100:.1f} -> {s_t3*100:.1f})", flush=True)
    print(f"\nwritten: {OUT}", flush=True)


if __name__ == "__main__":
    main()
