#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v36 synthetic check A: M-CCM proxy-geometry hazard under ZERO coupling.

Question: on bivariate series with no causal coupling of any kind -- both
driven only by a shared deterministic seasonal cycle plus independent AR(1)
noise -- does M-CCM with seasonal coordinates still return a directional
vote, and does that vote track the seasonality asymmetry between the series?

Design (mirrors the real-data regime: flow seasonal R^2 ~ 0.29, nutrients
~ 0.14):
  x_t = a_x * s_t + e_x_t   (flow proxy; seasonal R^2 = 0.30)
  y_t = a_y * s_t + e_y_t   (nutrient proxy; seasonal R^2 swept)
  s_t: deterministic 2-harmonic seasonal shape (unit variance)
  e:   independent AR(1) phi=0.6, normalized to unit variance
  a_x, a_y set so Var(seasonal)/(Var(seasonal)+Var(noise)) = target R^2.

Machinery is the corrected pipeline's own simplex cross-map
(src/pipeline.py: ccm_run), used unmodified: E=3, tau=1, k=4,
lib 0.7, seed 42, ahead=1 for M-CCM (base manifold + sin/cos controls),
ahead=0 same-time for standard CCM. Vote F (flow->WQ) when the skill of
predicting flow from the WQ manifold exceeds the reverse.

Realizations: 200 per configuration, seed 42.
Output: data/results/tables/mccm_synthetic.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
from pipeline import ccm_run, seasonal_controls, detrend  # noqa: E402

T = f"{ROOT}/data/results/tables"

E, TAU, K, SEED, LIBF = 3, 1, 4, 42, 0.7
N_MONTHS = 240          # 20 years, comparable to the real median (168 valid)
N_REAL = 200
PHI = 0.6               # AR(1) coefficient of the noise


def ar1(n, phi, rng):
    e = np.empty(n)
    e[0] = rng.normal()
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal()
    return e / np.sqrt(1.0 / (1.0 - phi ** 2))   # unit stationary variance


def seasonal_shape(n):
    t = np.arange(n)
    s = np.sin(2 * np.pi * t / 12) + 0.4 * np.sin(4 * np.pi * t / 12 + 1.0)
    return s / np.std(s)


def make_pair(r2_x, r2_y, rng):
    s = seasonal_shape(N_MONTHS)
    ax = np.sqrt(r2_x / (1 - r2_x))
    ay = np.sqrt(r2_y / (1 - r2_y)) if r2_y > 0 else 0.0
    x = ax * s + ar1(N_MONTHS, PHI, rng)
    y = ay * s + ar1(N_MONTHS, PHI, rng)
    months = np.tile(np.arange(1, 13), N_MONTHS // 12)
    years = np.repeat(2020 + np.arange(N_MONTHS // 12), 12)
    cal = pd.period_range("2020-01", periods=N_MONTHS, freq="M")
    return x, y, np.ones(N_MONTHS, bool), cal


def run_config(r2_x, r2_y, n_real=N_REAL, seed=SEED):
    votes = {"mccm": [], "std": []}
    for i in range(n_real):
        rng = np.random.RandomState(seed + i)
        x, y, valid, cal = make_pair(r2_x, r2_y, rng)
        dx, dy = detrend(x), detrend(y)
        z = seasonal_controls(cal)
        # M-CCM: base manifold + seasonal controls, ahead=1 (pipeline convention)
        f2w_m, _, _ = ccm_run(dy, dx, valid, ahead=1, controls=z, E=E, tau=TAU, k=K,
                              lib_fraction=LIBF, seed=SEED)
        w2f_m, _, _ = ccm_run(dx, dy, valid, ahead=1, controls=z, E=E, tau=TAU, k=K,
                              lib_fraction=LIBF, seed=SEED)
        # standard CCM: same-time, no controls
        f2w_s, _, _ = ccm_run(dy, dx, valid, ahead=0, controls=None, E=E, tau=TAU, k=K,
                              lib_fraction=LIBF, seed=SEED)
        w2f_s, _, _ = ccm_run(dx, dy, valid, ahead=0, controls=None, E=E, tau=TAU, k=K,
                              lib_fraction=LIBF, seed=SEED)
        if not (np.isnan(f2w_m) or np.isnan(w2f_m)):
            votes["mccm"].append(1 if f2w_m > w2f_m else 0)
        if not (np.isnan(f2w_s) or np.isnan(w2f_s)):
            votes["std"].append(1 if f2w_s > w2f_s else 0)
    out = []
    for meth, v in votes.items():
        v = np.array(v)
        share = v.mean()
        lo, hi = stats.beta.ppf([0.025, 0.975], v.sum() + 0.5, len(v) - v.sum() + 0.5)
        p = stats.binomtest(int(v.sum()), len(v), 0.5).pvalue
        out.append({"r2_flow": r2_x, "r2_wq": r2_y, "method": meth,
                    "share_flow_dom": round(float(share), 3), "n": len(v),
                    "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                    "binom_p_two_sided": float(p)})
    return out


def main():
    rows = []
    # dose-response: flow seasonality fixed at the observed mean (0.30),
    # nutrient seasonality swept across the observed range (median 0.08,
    # mean 0.14, upper range 0.30)
    for r2_y in (0.00, 0.06, 0.12, 0.18, 0.30):
        rows += run_config(0.30, r2_y)
        for r in rows[-2:]:
            print(r, flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(f"{T}/mccm_synthetic.csv", index=False)
    print("saved", f"{T}/mccm_synthetic.csv", flush=True)


if __name__ == "__main__":
    main()
