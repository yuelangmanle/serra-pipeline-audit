"""The strict-counting layer: corrected TE / conditional TE, one definition.

Estimator: KSG-1 with the max-norm joint distance and Kraskov's counting rule
(strictly less than the joint k-distance), the rule the manuscript's methods
section states. Everything corrected-layer in the paper traces to this
script: the standardized shares, the unstandardized companion table, the
eleven-shift nulls, the strict-observed-months subsample and its own null,
the leave-one-country-out scan, the station-cluster bootstrap, and the
per-combination table carrying both counting variants side by side.

The eleven-shift null rotates the processed flow series by whole calendar
years: the complete-year blocks (a trailing partial year stays in place) are
rotated cyclically by k. The flow record's own observation pattern rotates
with its values, and each draw's analysis support is rebuilt from the rotated
record -- the rotated flow pattern intersected with the water-quality record's
own pattern, with the pipeline's two-month short-gap rule bridging gaps, the
same rule the observation's support gets, and the result intersected with the
rotated pattern of the flow record's own usable months (observed or filled
across at most two months) -- so every analyzed month of a draw carries a
flow value of the same provenance class the observation draws on. Rotating the detrended, standardized
series (rather than the raw record) avoids the saw-tooth that a cyclic
rotation of a trended series would leave behind for the detrending step.
Every position keeps its calendar month at every record length (seasonal
phase is preserved exactly, with no wrap discontinuity), and when the
complete-year count divides k the rotation is an exact identity that
reproduces the observation.

Run after pipeline.py and te.py; writes into data/results/tables/ under
PAPER_BASE.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"
sys.path.insert(0, HERE)

import pipeline as gp  # noqa: E402
from te import process_pair, lag_pairs, load_wq_total  # noqa: E402
from pipeline import (read_gsim, merged_calendar, year_shift,  # noqa: E402
                      rotated_support, fill_short_gaps_pair, detrend,
                      seasonal_controls)

KS = list(range(1, 12))


def te_and_cte(xp, yp, zp, yt, strict=True):
    """TE and conditional TE from lag-aligned rows, one counting rule."""
    te = (gp.ksg_mi(np.column_stack([xp, yp]), yt, strict=strict)
          - gp.ksg_mi(yp, yt, strict=strict))
    cte = (gp.ksg_mi(np.column_stack([xp, yp, zp]), yt, strict=strict)
           - gp.ksg_mi(np.column_stack([yp, zp]), yt, strict=strict))
    return te, cte


def main():
    t0 = time.time()
    combos = pd.read_csv(f"{T}/combos.csv")
    wq, _ = load_wq_total()

    z_rows, u_rows, b_rows = [], [], []
    null_rows, m_rows, mnull_rows = [], [], []
    identity_shifts = 0

    for i, c in combos.iterrows():
        gid, sid, param = c.gems_id, c.gsim_id, c.parameter
        series = wq.get(gid, {}).get(param, {})
        flow = read_gsim(sid)
        if flow is None or len(series) < 60:
            continue
        x_raw, y_raw, vobs, cal = merged_calendar(flow, series)
        if vobs.sum() < 60:
            continue
        built = process_pair(x_raw, y_raw, vobs, cal)
        if built is None:
            continue
        dzx, dzy, valid, _ = built
        zp_all = seasonal_controls(cal)
        obs_x = ~np.isnan(x_raw)
        obs_y = ~np.isnan(y_raw)

        y_years = len(x_raw) // 12
        if any(k % y_years == 0 for k in KS):
            identity_shifts += 1

        # ---- standardized strict: observations + inclusive comparison ----
        idx, xp, yp, yt = lag_pairs(dzx, dzy, valid)
        if len(idx) < 40:
            continue
        zpi = zp_all[idx - 1]
        xt = dzx[idx]
        te_f, cte_f = te_and_cte(xp, yp, zpi, yt)
        te_w, cte_w = te_and_cte(yp, xp, zpi, xt)
        z_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                       "te_f2w": te_f, "te_w2f": te_w,
                       "cte_f2w": cte_f, "cte_w2f": cte_w, "n_pairs": len(idx)})
        te_f_i, cte_f_i = te_and_cte(xp, yp, zpi, yt, strict=False)
        te_w_i, cte_w_i = te_and_cte(yp, xp, zpi, xt, strict=False)
        b_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                       "te_f2w_strict": te_f, "te_w2f_strict": te_w,
                       "cte_f2w_strict": cte_f, "cte_w2f_strict": cte_w,
                       "te_f2w_inclusive": te_f_i, "te_w2f_inclusive": te_w_i,
                       "cte_f2w_inclusive": cte_f_i, "cte_w2f_inclusive": cte_w_i})

        # ---- unstandardized strict ----
        x2, y2, v2 = fill_short_gaps_pair(x_raw, y_raw, vobs, max_gap=2)
        dx, dy = detrend(x2), detrend(y2)
        zd = seasonal_controls(cal)
        idxu, xpu, ypu, ytu = lag_pairs(dx, dy, v2)
        if len(idxu) >= 40:
            zpu = zd[idxu - 1]
            xtu = dx[idxu]
            tu_f, cu_f = te_and_cte(xpu, ypu, zpu, ytu)
            tu_w, cu_w = te_and_cte(ypu, xpu, zpu, xtu)
            u_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                           "te_f2w": tu_f, "te_w2f": tu_w,
                           "cte_f2w": cu_f, "cte_w2f": cu_w, "n_pairs": len(idxu)})

        # ---- eleven-shift nulls, standardized strict: whole-year rotation
        #      of the processed series. The flow record's own observation
        #      pattern rotates with its values; the support of each draw is
        #      rebuilt from the rotated record (rotated flow pattern and the
        #      WQ record's own pattern intersected, short gaps bridged by the
        #      same rule the observation's support gets), so every analyzed
        #      month carries usable values from both records ----
        for k in KS:
            xs = year_shift(dzx, k)
            vk = rotated_support(obs_x, obs_y, k)
            idxk, xpk, ypk, ytk = lag_pairs(xs, dzy, vk)
            if len(idxk) < 40:
                continue
            zpk = zp_all[idxk - 1]
            xtk = xs[idxk]
            tek_f, ctek_f = te_and_cte(xpk, ypk, zpk, ytk)
            tek_w, ctek_w = te_and_cte(ypk, xpk, zpk, xtk)
            null_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                              "k": k,
                              "te_f": tek_f, "te_w": tek_w,
                              "cte_f": ctek_f, "cte_w": ctek_w,
                              "n_pairs": len(idxk)})

        # ---- strict-observed-months TE: windows restricted to observed
        #      months; trend and scaling stay estimated on the full series ----
        mask_obs = vobs & valid
        idxm, xpm, ypm, ytm = lag_pairs(dzx, dzy, mask_obs)
        if len(idxm) >= 40:
            zpm = zp_all[idxm - 1]
            xtm = dzx[idxm]
            tm_f = te_and_cte(xpm, ypm, zpm, ytm)[0]
            tm_w = te_and_cte(ypm, xpm, zpm, xtm)[0]
            m_rows.append({"gems_id": gid, "gsim_id": sid, "parameter": param,
                           "te_f": tm_f, "te_w": tm_w,
                           "n": int(mask_obs.sum())})
            mnull_rows.append({"gems_id": gid, "gsim_id": sid,
                               "parameter": param, "k": 0,
                               "vote": 1 if tm_f > tm_w else 0,
                               "n": len(idxm)})
            for k in KS:
                xs = year_shift(dzx, k)
                vsk = rotated_support(obs_x, obs_y, k, fill=False)
                idxs, xps, yps, yts = lag_pairs(xs, dzy, vsk)
                if len(idxs) < 40:
                    continue
                zps = zp_all[idxs - 1]
                xts = xs[idxs]
                ts_f = te_and_cte(xps, yps, zps, yts)[0]
                ts_w = te_and_cte(yps, xps, zps, xts)[0]
                mnull_rows.append({"gems_id": gid, "gsim_id": sid,
                                   "parameter": param, "k": k,
                                   "vote": 1 if ts_f > ts_w else 0,
                                   "n": len(idxs)})

        if (i + 1) % 40 == 0:
            print(f"{i+1} ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- write tables ----
    from scipy import stats
    dz = pd.DataFrame(z_rows)
    du = pd.DataFrame(u_rows)
    db = pd.DataFrame(b_rows)
    dz.to_csv(f"{T}/te_z.csv", index=False)
    du.to_csv(f"{T}/te.csv", index=False)
    db.to_csv(f"{T}/ksg_boundary_check.csv", index=False)
    dn = pd.DataFrame(null_rows)
    dn.to_csv(f"{T}/strict_nulls.csv", index=False)
    dm = pd.DataFrame(m_rows)
    dm.to_csv(f"{T}/strict_months.csv", index=False)
    dnm = pd.DataFrame(mnull_rows)
    dnm.to_csv(f"{T}/strict_months_null.csv", index=False)

    n = len(dz)
    kte = int((dz.te_f2w > dz.te_w2f).sum())
    kcte = int((dz.cte_f2w > dz.cte_w2f).sum())
    print(f"te_z  strict(max-norm): n={n} TE {100*kte/n:.1f}% ({kte}/{n}) "
          f"CTE {100*kcte/n:.1f}% ({kcte}/{n})")
    print(f"   TE binom two-sided {stats.binomtest(kte, n, 0.5).pvalue:.4f} | "
          f"CTE {stats.binomtest(kcte, n, 0.5).pvalue:.3g}")
    ith = int((db.te_f2w_inclusive > db.te_w2f_inclusive).sum())
    ich = int((db.cte_f2w_inclusive > db.cte_w2f_inclusive).sum())
    ni = len(db)
    print(f"   inclusive variant: TE {100*ith/ni:.1f}% | CTE {100*ich/ni:.1f}%")
    nu = len(du)
    ktu = int((du.te_f2w > du.te_w2f).sum())
    print(f"te unstd strict: n={nu} share {100*ktu/nu:.1f}%; "
          f"exact zeros f2w {int((du.te_f2w == 0).sum())} w2f {int((du.te_w2f == 0).sum())}")
    print(f"combos with an identity shift (block count <= 11): {identity_shifts} of {n}")
    sh = 100 * dn.assign(_hit=dn.te_f > dn.te_w).groupby("k")._hit.mean()
    sc = 100 * dn.assign(_hit=dn.cte_f > dn.cte_w).groupby("k")._hit.mean()
    print(f"TE  null(strict): mean {sh.mean():.1f} range [{sh.min():.1f},{sh.max():.1f}] "
          f"| draws >= observed {int((sh >= 100*kte/n).sum())}/11")
    print(f"CTE null(strict): mean {sc.mean():.1f} range [{sc.min():.1f},{sc.max():.1f}] "
          f"| draws >= observed {int((sc >= 100*kcte/n).sum())}/11")
    if len(dm):
        km = int((dm.te_f > dm.te_w).sum())
        print(f"strict months TE: n={len(dm)} share {100*km/len(dm):.1f}%")
        obs = dnm[dnm.k == 0].vote.mean()
        shm = dnm[dnm.k > 0].groupby("k").vote.mean()
        print(f"  its null: mean {100*shm.mean():.1f} "
              f"range [{100*shm.min():.1f},{100*shm.max():.1f}] "
              f"| draws >= observed {int((shm >= obs).sum())}/11 rank p "
              f"{int((shm >= obs).sum() + 1)}/12")

    # leave-one-country-out on the standardized votes
    dz2 = dz.copy()
    dz2["country"] = dz2.gems_id.str[:3]
    dz2["vote"] = (dz2.te_f2w > dz2.te_w2f).astype(int)
    loco = []
    for scope, g in [("full", dz2)] + [(f"leave-out {c}", dz2[dz2.country != c])
                                       for c in sorted(dz2.country.unique())]:
        k_, n_ = int(g.vote.sum()), len(g)
        loco.append({"scope": scope, "n": n_, "share": round(k_ / n_, 4),
                     "p_two_sided": round(stats.binomtest(k_, n_, 0.5).pvalue, 4)})
    pd.DataFrame(loco).to_csv(f"{T}/te_loco_strict.csv", index=False)
    print(pd.DataFrame(loco).to_string())

    # station-cluster bootstrap for both corrected TE shares
    du2 = du.copy()
    du2["vote"] = (du2.te_f2w > du2.te_w2f).astype(int)
    boot_rows = []
    for tag, frame in (("te_share_strict", dz2), ("te_share_unstd", du2)):
        rng = np.random.RandomState(42)
        votes = frame.vote.values.astype(float)
        clus = frame.gsim_id.values
        cl = pd.unique(clus)
        boots = []
        for _ in range(2000):
            pick = rng.choice(cl, size=len(cl), replace=True)
            idxs = np.concatenate([np.where(clus == c_)[0] for c_ in pick])
            boots.append(votes[idxs].mean())
        lo95, hi95 = np.percentile(boots, [2.5, 97.5])
        boot_rows.append({"statistic": tag, "observed": round(votes.mean(), 4),
                          "ci_lo": round(lo95, 4), "ci_hi": round(hi95, 4),
                          "n": int(len(votes)), "n_clusters": int(len(cl)),
                          "n_boot": 2000, "seed": 42})
        print(f"cluster CI {tag}: {100*votes.mean():.1f} "
              f"[{100*lo95:.1f}, {100*hi95:.1f}]")
    pd.DataFrame(boot_rows).to_csv(f"{T}/cluster_robust_strict.csv", index=False)
    print(f"done ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
