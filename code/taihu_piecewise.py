#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Taihu positive control under piecewise-linear detrending (2009 break).

A shared slow driver that moved both series after 2009 survives plain linear
detrending only through its residual slope; a hinge term at 2009-01 absorbs
it. Applied per raw series before deseason+z, then the same year-block
permutation family as lakes_null.py (1,000 draws, argmax-matched with the
fixed-E p alongside). If all six directions clear the matched null here, the
positive-control claim is robust to the break.

Writes the unified data/results/tables/taihu_piecewise.csv (Taihu piecewise columns joined onto lakes_null.csv; same schema as the submission supplement).
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
sys.path.insert(0, HERE)
from lakes import deseason_z  # noqa: E402
from lakes_null import (load_pairs, year_blocks, rho_fixed,  # noqa: E402
                        perm_p_argmax, perm_p_fixedE)

T = f"{ROOT}/data/results/tables"
BREAK_YEAR = 2009


def piecewise_detrend(a, years, valid):
    """a - [1, t, (t - t_break)+] least-squares fit over valid months."""
    t = np.arange(len(a), dtype=float)
    tb = float(np.where(years == BREAK_YEAR)[0][0])
    X = np.column_stack([np.ones(len(a)), t, np.maximum(0.0, t - tb)])
    coef, *_ = np.linalg.lstsq(X[valid], a[valid], rcond=None)
    return a - X @ coef


def main():
    t0 = time.time()
    pairs = load_pairs()
    rows = []
    for (lake, a, b), (sa, sb, cal, years) in pairs.items():
        if lake != "Taihu":
            continue
        for x, y, dname in [(sa, sb, f"{a}->{b}"), (sb, sa, f"{b}->{a}")]:
            vx = ~np.isnan(x)
            vy = ~np.isnan(y)
            xpw = piecewise_detrend(x, years, vx)
            ypw = piecewise_detrend(y, years, vy)
            xz = deseason_z(xpw, cal, vx)
            yz = deseason_z(ypw, cal, vy)
            valid = ~np.isnan(xz) & ~np.isnan(yz)
            rho_es = {E: rho_fixed(yz, xz, valid, E) for E in range(2, 7)}
            rho_es = {E: r for E, r in rho_es.items() if not np.isnan(r)}
            E_best = max(rho_es, key=rho_es.get)
            rho_best = rho_es[E_best]
            blocks, n_full = year_blocks(years)
            p_arg, _ = perm_p_argmax(yz, xz, valid, blocks, rho_best)
            p_fix, _ = perm_p_fixedE(yz, xz, valid, blocks, rho_best, E_best)
            rows.append({
                "lake": "Taihu", "direction": dname,
                "detrend": "piecewise_2009", "E_best": E_best,
                "rho_best": round(rho_best, 3),
                "p_yearperm_fixedE": round(p_fix, 4),
                "p_yearperm_argmax": round(p_arg, 4),
                "n": int(valid.sum()), "n_full_years": n_full,
                "per_E_rho": ";".join(f"E{E}:{r:.3f}" for E, r in sorted(rho_es.items()))})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    # ship the unified table: the Taihu piecewise columns joined onto the
    # lake-direction table (same schema as the submission supplement)
    pw = df.rename(columns={"E_best": "E_piecewise2009",
                            "rho_best": "rho_piecewise2009",
                            "p_yearperm_argmax": "p_yearperm_argmax_piecewise2009",
                            "n": "n_piecewise"})[
        ["direction", "E_piecewise2009", "rho_piecewise2009",
         "p_yearperm_argmax_piecewise2009", "n_piecewise"]]
    pw["lake"] = "Taihu"
    base = pd.read_csv(f"{T}/lakes_null.csv")
    out = base.merge(pw, on=["lake", "direction"], how="left")
    cols = ["lake", "direction", "E_best", "rho_best", "p_yearperm_fixedE",
            "p_yearperm_argmax", "rho_best_detrended", "E_best_detrended",
            "p_yearperm_argmax_detrended", "n", "n_full_years",
            "n_distinct_permutations", "per_E_rho", "rho_piecewise2009",
            "E_piecewise2009", "p_yearperm_argmax_piecewise2009", "n_piecewise"]
    out = out[cols]
    assert out[out.lake == "Taihu"].rho_piecewise2009.notna().all()
    assert len(out) == len(base)
    out.to_csv(f"{T}/taihu_piecewise.csv", index=False)
    print(f"written: {T}/taihu_piecewise.csv ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
