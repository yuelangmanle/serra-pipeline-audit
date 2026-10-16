#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same enumeration for the corrected TE / conditional TE layer.

Must run after te.py; the series convention (detrend, z-score on valid
months) is copied from there so the observed shares match the standardized
tables.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
import te as tz
from pipeline import read_gsim, merged_calendar, fill_short_gaps_pair, detrend

T = f"{ROOT}/data/results/tables"
LAG = tz.LAG
KS = list(range(1, 12))


def prep_combo(sid, series):
    built = tz.build_corrected(sid, series)
    if built is None:
        return None
    dzx, dzy, valid, cal = built
    months = np.array([p.month for p in cal])
    z = np.column_stack([np.sin(2 * np.pi * months / 12),
                         np.cos(2 * np.pi * months / 12)])
    return dzx, dzy, valid, z


def vote_pair(dzx, dzy, valid, z):
    idx = np.array([t for t in range(LAG, len(dzx))
                    if valid[t] and valid[t - LAG]], dtype=int)
    if len(idx) < 40:
        return None
    yt = dzy[idx]; xp = dzx[idx - LAG]; yp = dzy[idx - LAG]; zp = z[idx - LAG]
    te_f, cte_f = tz.te_cte_proper(xp, yp, zp, yt)
    xt = dzx[idx]; xq = dzx[idx - LAG]; yq = dzy[idx - LAG]
    te_w, cte_w = tz.te_cte_proper(yq, xq, zp, xt)
    if any(np.isnan(v) for v in (te_f, te_w, cte_f, cte_w)):
        return None
    return (1 if te_f > te_w else 0), (1 if cte_f > cte_w else 0), te_f, te_w, cte_f, cte_w


def main():
    wq, _ = tz.load_wq_total()
    combos = pd.read_csv(f"{T}/combos.csv")
    obs, raw = {}, {}
    for _, r in combos.iterrows():
        gid, sid, param = r.gems_id, r.gsim_id, r.parameter
        series = wq.get(gid, {}).get(param, {})
        p = prep_combo(sid, series)
        if p is None:
            continue
        dzx, dzy, valid, z = p
        v = vote_pair(dzx, dzy, valid, z)
        if v is None:
            continue
        obs[(gid, sid, param)] = (v[0], v[1])
        raw[(gid, sid, param)] = v[2:]
    print(f"obs combos: {len(obs)}", flush=True)
    f_share = np.mean([v[0] for v in obs.values()])
    c_share = np.mean([v[1] for v in obs.values()])
    print(f"TE-z observed F-share: {f_share:.4f}  CTE-z: {c_share:.4f}", flush=True)

    csv = pd.read_csv(f"{T}/te_z.csv")
    n_match = n_tot = 0
    for (gid, sid, param), v in obs.items():
        row = csv[(csv.gems_id == gid) & (csv.gsim_id == sid) & (csv.parameter == param)]
        if row.empty:
            continue
        n_tot += 1
        if (abs(row.te_f2w.iloc[0] - raw[(gid, sid, param)][0]) < 1e-9
                and abs(row.cte_f2w.iloc[0] - raw[(gid, sid, param)][2]) < 1e-9):
            n_match += 1
    print(f"exact match vs te_z.csv: {n_match}/{n_tot}", flush=True)

    rows = []
    for i, ((gid, sid, param), _) in enumerate(obs.items()):
        series = wq.get(gid, {}).get(param, {})
        flow = read_gsim(sid)
        x_raw, y_raw, valid_obs, cal = merged_calendar(flow, series)
        x_f, y_f, valid2 = fill_short_gaps_pair(x_raw, y_raw, valid_obs, max_gap=2)
        if np.nanstd(y_f) < 1e-6:
            continue
        months = np.array([p.month for p in cal])
        z = np.column_stack([np.sin(2 * np.pi * months / 12),
                             np.cos(2 * np.pi * months / 12)])
        dyk = detrend(y_f)
        zv2 = dyk[valid2]
        zy = (dyk - zv2.mean()) / zv2.std()
        for k in KS:
            dxk = detrend(np.roll(x_f, 12 * k))
            zv = dxk[valid2]
            zx = (dxk - zv.mean()) / zv.std()
            v = vote_pair(zx, zy, valid2, z)
            if v is None:
                continue
            rows.append({"pipeline": "te_z", "k": k, "vote": v[0], "gems_id": gid})
            rows.append({"pipeline": "cte_z", "k": k, "vote": v[1], "gems_id": gid})
        if (i + 1) % 50 == 0:
            print(f"align done {i+1}", flush=True)
    al = pd.DataFrame(rows)
    al.to_csv(f"{T}/null_alignments_corrected.csv", index=False)
    for p in ("te_z", "cte_z"):
        g = al[al.pipeline == p]
        shares = g.groupby("k").vote.mean()
        print(f"\n{p}: 11 alignment shares mean {shares.mean():.3f} "
              f"range [{shares.min():.3f}, {shares.max():.3f}] sd {shares.std():.3f}")


if __name__ == "__main__":
    main()
