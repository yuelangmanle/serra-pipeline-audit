#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assorted robustness numbers, added as they were asked for.

Library-fraction shares and the same-pipeline self-flip baseline, the
geographic funnel, the leave-one-country-out scan, and the concordance
permutation band. All from existing tables.
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"
rng = np.random.RandomState(42)


def votes(df, fcol, wcol):
    v = pd.DataFrame({"key": df.gems_id + "|" + df.gsim_id + "|" + df.parameter})
    v["vote"] = np.where(df[fcol] > df[wcol], 1, np.where(df[wcol] > df[fcol], 0, np.nan))
    return v.set_index("key")["vote"]


def kappa(a, b):
    po = (a == b).mean()
    pe = (a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main():
    # ---------- 1. lib shares + self-flip ----------
    std = pd.read_csv(f"{T}/std_ccm.csv")
    shares, flip_rows = {}, []
    piv = {}
    for lf in (0.3, 0.5, 0.7):
        s = std[std.lib == lf]
        v = votes(s, "f2w", "w2f")
        piv[lf] = v
        shares[lf] = v.mean()
    for a, b in ((0.3, 0.5), (0.3, 0.7), (0.5, 0.7)):
        va, vb = piv[a], piv[b]
        ok = va.notna() & vb.notna()
        flip_rows.append({"lib_a": a, "lib_b": b, "n": int(ok.sum()),
                          "flip_rate": round(float((va[ok] != vb[ok]).mean()), 3)})
    pd.DataFrame({"lib": list(shares), "flow_dom_share": [round(shares[k], 3) for k in shares]}
                 ).to_csv(f"{T}/batch1_libshare.csv", index=False)
    pd.DataFrame(flip_rows).to_csv(f"{T}/batch1_selfflip.csv", index=False)
    print("lib shares:", {k: round(v, 3) for k, v in shares.items()})
    print("self-flips:", flip_rows)

    # ---------- 2. naive->corrected flips, detrended same-time naive ----------
    naive = pd.read_csv(f"{T}/true_xmap_sametime_full.csv", low_memory=False)
    vn = votes(naive, "f2w_0.7", "w2f_0.7")
    vc = votes(std[std.lib == 0.7], "f2w", "w2f")
    keys = vn.index.intersection(vc.index)
    ok = vn[keys].notna() & vc[keys].notna()
    flips = int((vn[keys][ok] != vc[keys][ok]).sum())
    n_naive_share = float(vn[keys].mean())
    print(f"naive(detrended same-time, lib .7) share on common keys: {n_naive_share:.3f}")
    print(f"naive->corrected flips: {flips}/{int(ok.sum())} = {flips/ok.sum():.3f}")
    pd.DataFrame([{"naive_source": "true_xmap_sametime_full (detrended same-time, lib 0.7)",
                   "n": int(ok.sum()), "naive_share": round(n_naive_share, 3),
                   "corrected_share": 0.519, "flips": flips,
                   "flip_rate": round(flips / ok.sum(), 3)}]
                 ).to_csv(f"{T}/batch1_flips.csv", index=False)
    pd.DataFrame([{"comparison": "std CCM naive vs corrected",
                   "n": int(ok.sum()),
                   "agree": round(1.0 - flips / ok.sum(), 3),
                   "flips": flips,
                   "flip_rate": round(flips / ok.sum(), 3)}]
                 ).to_csv(f"{T}/batch1_concordance.csv", index=False)

    # ---------- 3. geographic funnel ----------
    m = pd.read_csv(f"{ROOT}/data/interim/matched_river_stations_50km_10yr.csv", low_memory=False)
    res491 = pd.read_csv(f"{T}/ccm_analysis_results.csv", low_memory=False)
    res491 = res491[res491.n_months >= 60]
    combos = pd.read_csv(f"{T}/combos.csv")
    cc = m[["gems_id", "gems_country"]].drop_duplicates("gems_id")

    def by_country(df, key="gems_id"):
        j = df.merge(cc, on=key, how="left") if key in df.columns else df
        return j.gems_country.value_counts()
    f9377 = m.drop_duplicates(["gems_id", "nearest_gsim_id"]).gems_country.value_counts()
    f491 = by_country(res491)
    f316 = by_country(combos)
    funnel = pd.DataFrame({"matched_pairs": f9377, "combos_491": f491, "combos_316": f316}).fillna(0).astype(int)
    funnel.loc["TOTAL"] = funnel.sum()
    funnel.to_csv(f"{T}/batch1_funnel.csv")
    print("funnel total:", funnel.loc["TOTAL"].to_dict())

    # ---------- 4. TE-z leave-one-country-out ----------
    te = pd.read_csv(f"{T}/te_z.csv")
    te = te.merge(cc, on="gems_id", how="left")
    te["vote"] = np.where(te.te_f2w > te.te_w2f, 1, 0)
    rows = []
    countries = sorted(te.gems_country.unique())
    for c in [None] + countries:
        sub = te if c is None else te[te.gems_country != c]
        k = int(sub.vote.sum()); n = len(sub)
        p = stats.binomtest(k, n, 0.5).pvalue
        rows.append({"scope": "full" if c is None else f"leave-out {c}",
                     "n": n, "share": round(k / n, 3), "p_two_sided_binom": round(p, 4)})
    pd.DataFrame(rows).to_csv(f"{T}/batch1_te_loco.csv", index=False)
    print("TE-z LOCO:", rows)

    # ---------- 5. concordance permutation ----------
    des = pd.read_csv(f"{T}/deseason.csv")
    mc = pd.read_csv(f"{T}/mccm_fixed.csv")
    tz = pd.read_csv(f"{T}/te_z.csv")
    V = {"std": votes(std[std.lib == 0.7], "f2w", "w2f"),
         "des": votes(des, "f2w", "w2f"),
         "mccm": votes(mc, "mccm_f2w", "mccm_w2f"),
         "te": votes(tz, "te_f2w", "te_w2f"),
         "cte": votes(tz, "cte_f2w", "cte_w2f")}
    keys = set.intersection(*[set(v.dropna().index) for v in V.values()])
    keys = sorted(keys)
    M = np.stack([V[n][keys].values for n in V], axis=1).astype(float)  # n x 5, 1/0
    n_sub, m_n = M.shape

    # station blocks for permutation (cluster on gsim_id)
    key_df = pd.DataFrame({"key": keys})
    key_df["gsim"] = [k.split("|")[1] for k in keys]
    blocks = key_df.groupby("gsim").indices.values()

    def fleiss(Mx):
        nf = Mx.sum(axis=1); nw = m_n - nf
        Pi = (nf * (nf - 1) + nw * (nw - 1)) / (m_n * (m_n - 1))
        pF = Mx.mean(); pE = pF ** 2 + (1 - pF) ** 2
        return (Pi.mean() - pE) / (1 - pE)

    obs_fleiss = fleiss(M)
    perm = np.empty(5000)
    pair_perm = np.empty((5000, 10))
    pairs = list(itertools.combinations(range(m_n), 2))
    blocks = [np.array(b) for b in blocks]
    sizes = [len(b) for b in blocks]
    by_size = {}
    for bi, sz in enumerate(sizes):
        by_size.setdefault(sz, []).append(bi)
    for i in range(5000):
        Mp = M.copy()
        for col in range(m_n):                      # each pipeline independently
            v = M[:, col].copy()
            vp = v.copy()
            for sz, members in by_size.items():     # whole stations swap within
                order = rng.permutation(len(members))   # equal-size strata: no
                for tgt_i, src_i in enumerate(order):   # block splits, margins kept
                    vp[blocks[members[tgt_i]]] = v[blocks[members[src_i]]]
            Mp[:, col] = vp
        perm[i] = fleiss(Mp)
        for j, (a, b) in enumerate(pairs):
            pair_perm[i, j] = kappa(Mp[:, a], Mp[:, b])
    lo, hi = np.percentile(perm, [2.5, 97.5])
    p_lower = (np.sum(perm <= obs_fleiss) + 1) / 5001
    p_upper = (np.sum(perm >= obs_fleiss) + 1) / 5001
    p_two = min(1.0, 2 * min(p_lower, p_upper))
    print(f"Fleiss obs={obs_fleiss:.3f}; independence-null 95% [{lo:.3f},{hi:.3f}]; "
          f"P(null<=obs)={p_lower:.3f}")
    pd.DataFrame([{"statistic": "fleiss_kappa", "observed": round(obs_fleiss, 3),
                   "null_mean": round(float(perm.mean()), 4), "null_lo": round(float(lo), 4),
                   "null_hi": round(float(hi), 4), "p_one_sided_lower": round(float(p_lower), 4), "p_two_sided": round(float(p_two), 4),
                   "n": n_sub, "n_perm": 5000}]
                 ).to_csv(f"{T}/batch1_concperm.csv", index=False)
    # pairwise observed kappas for the corrected text
    pr = []
    for j, (a, b) in enumerate(pairs):
        klo, khi = np.percentile(pair_perm[:, j], [2.5, 97.5])
        pr.append({"pair": f"{list(V)[a]}-{list(V)[b]}",
                   "kappa": round(kappa(M[:, a], M[:, b]), 3),
                   "null_lo": round(float(klo), 3), "null_hi": round(float(khi), 3)})
    pd.DataFrame(pr).to_csv(f"{T}/batch1_pairwise_kappa.csv", index=False)
    print(pd.DataFrame(pr).to_string(index=False))


if __name__ == "__main__":
    main()
