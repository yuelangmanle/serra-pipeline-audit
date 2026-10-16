"""Year-block permutation nulls for the lakes, argmax-matched over E.

Observed statistic and every draw are scored on the same complete-year
window, so an identity permutation reproduces the observed statistic.

Poyang has seven complete years, so its permutation null is enumerated
exactly over all 7! = 5,040 orders (p = (b + 1) / 5,041, no Monte-Carlo
error). Taihu keeps 1,000 Monte-Carlo draws and the report carries the
draw-count resolution and a three-seed sweep of the argmax-matched p.
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


def year_blocks(years):
    counts = pd.Series(years).value_counts()
    full = sorted(counts[counts == 12].index)
    mask = np.isin(years, full)
    idx = np.where(mask)[0]
    blocks = [idx[i * 12:(i + 1) * 12] for i in range(len(full))]
    return blocks, len(full)


def rho_fixed(base, target, valid, E):
    r, _ = simplex_ccm(base, target, valid, E)
    return r


ENUM_MAX = 7          # 7! = 5,040 orders -- enumerate exactly at or below
SEED_SWEEP = (42, 43, 44)


def _perm_p(base, target, valid, blocks, rho_obs, E=None,
            n_surr=N_SURR, seed=SEED):
    """Upper-tail permutation p for the observed statistic.

    Exact enumeration when the block count allows it (Poyang: 7 blocks);
    1,000 random draws otherwise. Returns (p, n_used)."""
    import math
    nb = len(blocks)
    if nb <= ENUM_MAX:
        from itertools import permutations
        orders = permutations(range(nb))
        n_used = math.factorial(nb)
    else:
        rng = np.random.RandomState(seed)
        orders = (rng.permutation(nb) for _ in range(n_surr))
        n_used = n_surr
    b = 0
    for order in orders:
        bs = np.concatenate([base[blocks[o]] for o in order])
        vs = np.concatenate([valid[blocks[o]] for o in order])
        if E is None:
            best = -np.inf
            for e in range(2, 7):
                r = rho_fixed(bs, target, vs, e)
                if not np.isnan(r) and r > best:
                    best = r
            hit = best >= rho_obs
        else:
            r = rho_fixed(bs, target, vs, E)
            hit = (not np.isnan(r)) and r >= rho_obs
        if hit:
            b += 1
    return (b + 1) / (n_used + 1), n_used


def perm_p_argmax(base, target, valid, blocks, rho_obs, n_surr=N_SURR, seed=SEED):
    return _perm_p(base, target, valid, blocks, rho_obs, None, n_surr, seed)


def perm_p_fixedE(base, target, valid, blocks, rho_obs, E, n_surr=N_SURR, seed=SEED):
    return _perm_p(base, target, valid, blocks, rho_obs, E, n_surr, seed)


def detrend_valid(a, valid):
    t = np.arange(len(a))
    sl, ic, _, _, _ = stats.linregress(t[valid], a[valid])
    return a - (sl * t + ic)


def load_pairs():
    pairs = {}
    p = pd.read_csv(f"{ROOT}/data/processed/poyang_merged_hydrology_wq.csv")
    p["date"] = pd.to_datetime(dict(year=p.year, month=p.month, day=1))
    p = p.set_index("date").sort_index()
    # TN's calendar is the Poyang reference grid (reference convention)
    tn_v, cal_tn, _ = prep(p.TN)
    idx = cal_tn
    series = {}
    for name, col in [("WL", "water_level"), ("TN", "TN"), ("TP", "TP"), ("DIS", "discharge")]:
        col_data = p[col].dropna() if col == "discharge" else p[col]
        v, c, _ = prep(col_data)
        series[name] = pd.Series(v, index=c).reindex(idx)
    years = np.array([p_.year for p_ in idx])
    for a, b in [("WL", "TN"), ("WL", "TP"), ("DIS", "TN"), ("DIS", "TP")]:
        pairs[("Poyang", a, b)] = (series[a].values, series[b].values, idx, years)
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
    pairs = load_pairs()
    rows = []
    for (lake, a, b), (sa, sb, cal, years) in pairs.items():
        for x, y, dname in [(sa, sb, f"{a}->{b}"), (sb, sa, f"{b}->{a}")]:
            # observed: per-E rho on deseason+z series, argmax selection
            blocks_pre, _ = year_blocks(years)
            fy_idx = np.concatenate(blocks_pre)
            xs_fy, ys_fy = x[fy_idx], y[fy_idx]
            years_fy = years[fy_idx]
            va = deseason_z(xs_fy, cal[fy_idx], ~np.isnan(xs_fy))
            vb = deseason_z(ys_fy, cal[fy_idx], ~np.isnan(ys_fy))
            valid = ~np.isnan(va) & ~np.isnan(vb)
            rho_es = {E: rho_fixed(vb, va, valid, E) for E in range(2, 7)}
            rho_es = {E: r for E, r in rho_es.items() if not np.isnan(r)}
            E_best = max(rho_es, key=rho_es.get)
            rho_best = rho_es[E_best]
            # detrended variant, same selection (same full-year window)
            vad = detrend_valid(xs_fy, ~np.isnan(xs_fy))
            vbd = detrend_valid(ys_fy, ~np.isnan(ys_fy))
            vadz = deseason_z(vad, cal[fy_idx], ~np.isnan(vad))
            vbdz = deseason_z(vbd, cal[fy_idx], ~np.isnan(vbd))
            validd = ~np.isnan(vadz) & ~np.isnan(vbdz)
            rho_es_d = {E: rho_fixed(vbdz, vadz, validd, E) for E in range(2, 7)}
            rho_es_d = {E: r for E, r in rho_es_d.items() if not np.isnan(r)}
            E_best_d = max(rho_es_d, key=rho_es_d.get)
            rho_best_d = rho_es_d[E_best_d]
            # permutation nulls (argmax-matched and fixed-E)
            blocks, n_full = year_blocks(years_fy)
            p_arg, n_arg = perm_p_argmax(vb, va, valid, blocks, rho_best)
            p_fix, n_fix = perm_p_fixedE(vb, va, valid, blocks, rho_best, E_best)
            p_arg_d, n_argd = perm_p_argmax(vbdz, vadz, validd, blocks,
                                            rho_best_d, seed=SEED + 1)
            if len(blocks) > ENUM_MAX:
                sweep = [perm_p_argmax(vb, va, valid, blocks, rho_best,
                                       seed=s)[0] for s in SEED_SWEEP]
                sweep_d = [perm_p_argmax(vbdz, vadz, validd, blocks,
                                         rho_best_d, seed=s)[0]
                           for s in SEED_SWEEP]
                p_lo, p_hi = min(sweep), max(sweep)
                pd_lo, pd_hi = min(sweep_d), max(sweep_d)
            else:
                p_lo = p_hi = p_arg
                pd_lo = pd_hi = p_arg_d
            rows.append({
                "lake": lake, "direction": dname, "E_best": E_best,
                "rho_best": round(rho_best, 3), "p_yearperm_fixedE": round(p_fix, 4),
                "p_yearperm_argmax": round(p_arg, 4),
                "rho_best_detrended": round(rho_best_d, 3), "E_best_detrended": E_best_d,
                "p_yearperm_argmax_detrended": round(p_arg_d, 4),
                "p_argmax_seed_lo": round(p_lo, 4), "p_argmax_seed_hi": round(p_hi, 4),
                "p_argmax_detrended_seed_lo": round(pd_lo, 4),
                "p_argmax_detrended_seed_hi": round(pd_hi, 4),
                "n_perm_used": int(n_arg),
                "n": int(valid.sum()), "n_full_years": n_full,
                "n_distinct_permutations": "5040" if n_full == 7 else f"{n_full}!",
                "per_E_rho": ";".join(f"E{E}:{r:.3f}" for E, r in sorted(rho_es.items()))})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{T}/lakes_null.csv", index=False)
    print("saved", f"{T}/lakes_null.csv")


if __name__ == "__main__":
    main()
