#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eleven-shift alignment nulls for the corrected standard CCM and the
deseasonalized CCM.

Table 1's corrected CCM rows are read against their own design, not only
against the 50% line. Same year-block rotation as the TE nulls: the flow
series is rotated by whole years, the flow record's own observation pattern
rotating with its values, and each draw is scored on the support rebuilt
from the rotated record (rotated flow pattern intersected with the WQ
pattern, short gaps bridged by the pipeline's rule, and the result
intersected with the rotated pattern of the flow record's own usable
months so every analysed flow value has observation-or-fill provenance). The deseasonalized
variant re-subtracts the climatology and re-detrends before the shift, in
the order the corrected pipeline uses.

Writes data/results/tables/ccm_shifts_corrected.csv.
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

from te import load_wq_total  # noqa: E402
from pipeline import (ccm_run, detrend, read_gsim, merged_calendar,  # noqa: E402
                      fill_short_gaps_pair, year_shift, rotated_support)

KS = list(range(1, 12))


def sub_clim(a, months, valid):
    a2 = a.copy()
    for m in np.unique(months[valid]):
        sel = (months == m) & valid
        a2[sel] -= a2[sel].mean()
    return a2


def main():
    t0 = time.time()
    combos = pd.read_csv(f"{T}/combos.csv")
    wq, _ = load_wq_total()

    rows = []
    n_rows = []
    for i, c in combos.iterrows():
        flow = read_gsim(c.gsim_id)
        series = wq.get(c.gems_id, {}).get(c.parameter, {})
        if flow is None or len(series) < 60:
            continue
        x_raw, y_raw, valid_obs, cal = merged_calendar(flow, series)
        if valid_obs.sum() < 60:
            continue
        x, y, valid = fill_short_gaps_pair(x_raw, y_raw, valid_obs, max_gap=2)
        if np.nanstd(y) < 1e-6 or len(np.unique(np.round(y, 6))) < 10:
            continue
        months = np.array([p.month for p in cal])
        obs_x = ~np.isnan(x_raw)
        obs_y = ~np.isnan(y_raw)
        # observed shares (the reference convention the corrected tables use)
        f_std, _, _ = ccm_run(y, x, valid, 0, None)     # WQ manifold -> flow
        w_std, _, _ = ccm_run(x, y, valid, 0, None)
        row = {"gems_id": c.gems_id, "gsim_id": c.gsim_id,
               "parameter": c.parameter,
               "std_f": f_std, "std_w": w_std}
        da = detrend(sub_clim(x, months, valid))
        db = detrend(sub_clim(y, months, valid))
        f_des, _, _ = ccm_run(db, da, valid, 0, None)
        w_des, _, _ = ccm_run(da, db, valid, 0, None)
        row.update({"des_f": f_des, "des_w": w_des})
        # whole-year rotations on the values; each draw's support is rebuilt
        # from the rotated flow record (its own observation pattern rotates
        # with the values, intersected with the WQ record's pattern, short
        # gaps bridged by the pipeline's rule). The deseasonalized path
        # rotates the already deseasoned+detrended series (rotating the raw
        # record first would leave a saw-tooth for the detrending step)
        for k in KS:
            xs = year_shift(x, k)
            xsa = year_shift(da, k)
            vk = rotated_support(obs_x, obs_y, k)
            fs, _, ns = ccm_run(y, xs, vk, 0, None)
            ws, _, _ = ccm_run(xs, y, vk, 0, None)
            gs, _, nd = ccm_run(db, xsa, vk, 0, None)
            hs, _, _ = ccm_run(xsa, db, vk, 0, None)
            row.update({f"std_f_{k}": fs, f"std_w_{k}": ws,
                        f"des_f_{k}": gs, f"des_w_{k}": hs})
            # per-draw sample size: the number of usable target months under
            # the rotated support (identical for every skill in the draw);
            # a draw below the 40-pair minimum is recorded as NaN above and
            # carries its (small) n here, never a fabricated value
            n_rows.append({"gems_id": c.gems_id, "gsim_id": c.gsim_id,
                           "parameter": c.parameter, "k": k,
                           "n_std": ns, "n_des": nd})
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"{i+1} ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{T}/ccm_shifts_corrected.csv", index=False)
    pd.DataFrame(n_rows).to_csv(f"{T}/ccm_shifts_n.csv", index=False)
    for tag, fcol, wcol in (("std", "std_f", "std_w"), ("des", "des_f", "des_w")):
        obs = 100 * (df[fcol] > df[wcol]).mean()
        # per-draw shares over the valid draws; a draw recorded as NaN
        # (rotated record below the 40-pair minimum) is excluded from its
        # draw's denominator rather than silently counted as not-flow-dominant
        def _draw_share(k):
            pair = df[[f"{fcol}_{k}", f"{wcol}_{k}"]].dropna()
            return 100 * (pair.iloc[:, 0] > pair.iloc[:, 1]).mean()
        sh = np.array([_draw_share(k) for k in KS])
        exceed = int((sh >= obs).sum())
        print(f"{tag}: observed {obs:.1f}% | null mean {sh.mean():.1f} "
              f"range [{sh.min():.1f}, {sh.max():.1f}] | "
              f"shift exceedances {exceed}/11 | rank p <= {(exceed+1)/12:.3f}")
    print(f"done ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
