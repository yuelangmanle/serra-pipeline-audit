#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended M-CCM synthetic grid + mixture expectation.

Zero-coupling pairs, seasonal R2 on both axes, 150 realizations per cell.
The mixture expectation scores every synthetic realization with the
pipeline's own seasonal R2 and averages over the real combinations.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
from pipeline import ccm_run, seasonal_controls, detrend, seasonal_r2  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "archive"))
import mccm_synth as base  # noqa: E402

T = f"{ROOT}/data/results/tables"
N_REAL = 150
SEED = 42


def make_pair_ext(r2_x, r2_y, rng, n_months=240, phase_shift=0):
    s = base.seasonal_shape(n_months)
    if phase_shift:
        s = np.roll(s, phase_shift)          # nutrient seasonal shape lagged
    ax = np.sqrt(r2_x / (1 - r2_x))
    ay = np.sqrt(r2_y / (1 - r2_y)) if r2_y > 0 else 0.0
    x = ax * s + base.ar1(n_months, base.PHI, rng)
    y = ay * s + base.ar1(n_months, base.PHI, rng)
    cal = pd.period_range("2020-01", periods=n_months, freq="M")
    return x, y, np.ones(n_months, bool), cal


def run_point(r2_x, r2_y, n_real=N_REAL, knn=4, phase_shift=0, n_months=240, seed0=SEED):
    votes, pipe_r2x, pipe_r2y = [], [], []
    for i in range(n_real):
        rng = np.random.RandomState(seed0 + 7919 * i)
        x, y, valid, cal = make_pair_ext(r2_x, r2_y, rng, n_months, phase_shift)
        dx, dy = detrend(x), detrend(y)
        z = seasonal_controls(cal)
        f2w, _, _ = ccm_run(dy, dx, valid, ahead=1, controls=z, k=knn)
        w2f, _, _ = ccm_run(dx, dy, valid, ahead=1, controls=z, k=knn)
        if not (np.isnan(f2w) or np.isnan(w2f)):
            votes.append(1 if f2w > w2f else 0)
        months = np.array([p.month for p in cal])
        pipe_r2x.append(seasonal_r2(dx[valid], months[valid]))
        pipe_r2y.append(seasonal_r2(dy[valid], months[valid]))
    return {"r2_flow_target": r2_x, "r2_wq_target": r2_y, "knn": knn,
            "phase_shift": phase_shift, "n_months": n_months,
            "share": float(np.mean(votes)), "n_real": len(votes),
            "pipe_r2_flow": float(np.nanmean(pipe_r2x)),
            "pipe_r2_wq": float(np.nanmean(pipe_r2y))}


def main():
    rows = []
    for r2f in (0.15, 0.30, 0.45):
        for r2w in (0.0, 0.06, 0.12, 0.18, 0.30, 0.45):
            pt = run_point(r2f, r2w)
            rows.append(pt)
            print(pt, flush=True)
    # fairness variants
    for kw, tag in [({"knn": 6}, "knn6"),
                    ({"phase_shift": 3}, "phase3"),
                    ({"n_months": 168}, "len168")]:
        pt = run_point(0.30, 0.12, **kw)
        rows.append(pt)
        print(pt, flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(f"{T}/synth_grid.csv", index=False)

    # ---- mixture expectation on the real combinations ----
    r2 = pd.read_csv(f"{T}/seasonal_r2.csv")
    combos = pd.read_csv(f"{T}/combos.csv")
    r2 = r2.merge(combos[["gems_id", "gsim_id", "parameter"]], on=["gems_id", "gsim_id", "parameter"])
    grid = g[(g.knn == 4) & (g.phase_shift == 0) & (g.n_months == 240)]
    P = grid[["pipe_r2_flow", "pipe_r2_wq"]].values
    S = grid["share"].values
    def predict(rf, rw):
        d = np.sqrt((P[:, 0] - rf) ** 2 + (P[:, 1] - rw) ** 2)
        w = 1 / np.maximum(d, 1e-6) ** 2
        return float((w * S).sum() / w.sum())
    r2["pred_null_share"] = [predict(rf, rw) for rf, rw in
                             zip(r2.r2_flow.clip(0.05, 0.55), r2.r2_wq.clip(0, 0.5))]
    mix_mean = r2.pred_null_share.mean()
    # station-level cluster bootstrap
    rng = np.random.RandomState(42)
    stations = r2.gsim_id.unique()
    boots = []
    for _ in range(2000):
        pick = rng.choice(stations, len(stations), replace=True)
        parts = [r2[r2.gsim_id == s] for s in pick]
        boots.append(pd.concat(parts).pred_null_share.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = pd.DataFrame([{"mixture_expected_null_share": round(mix_mean, 3),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                         "observed_share": 0.859, "observed_ci": "81.8-89.7",
                         "n": len(r2)}])
    out.to_csv(f"{T}/synth_mixture.csv", index=False)
    print("\nmixture expected null share:", out.to_dict("records"))


if __name__ == "__main__":
    main()
