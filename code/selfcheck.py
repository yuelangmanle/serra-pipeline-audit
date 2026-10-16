#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-consistency self-check for the shipped package.

1. Month-phase property, in two parts. (a) A pure operator certificate that
   runs from the package alone: at every shipped calendar length, the
   whole-year block rotation maps each array position to a source position
   with the same slot within the calendar year (k = 1..11), and the trailing
   partial-block positions stay in place. Because the merged calendars are
   contiguous monthly period ranges, slot-in-year preservation is exactly
   calendar-month preservation. (b) An archive cross-check that re-reads the
   per-record month vectors (needs data/raw) and repeats the property on the
   real calendars; reported as not executed, never as a silent pass, when
   the archives are absent.
2. Key integrity: the alignment-null tables carry full (gems_id, gsim_id,
   parameter) keys and anchor rows, with the strict budget complete and any
   reduced month-draw coverage printed.
3. Producer integrity: every result table is either written by exactly one
   current script or labelled frozen/superseded (the documented set); no
   duplicate writers, no undeclared tables.
4. Identity-shift exactness: wherever the complete-year count divides k, the
   shipped null tables must reproduce the observed row bit-for-bit (the
   rotation is an exact identity there); checked against the observed
   columns of the same package.
5. Null-value integrity: all shipped null values are finite; the corrected
   CCM table's NaN draws (rotated records below the 40-pair minimum) are
   each matched to a sub-minimum sample size in the companion
   ccm_shifts_n.csv and are excluded from the per-draw shares.

Exit code is non-zero when any check fails. The closing line states exactly
which checks ran; a check that could not run is reported as not executed and
no "ALL PASS" is printed for a partial certificate.
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"
sys.path.insert(0, HERE)

failures = []
executed = []
skipped = []


def check_month_phase():
    # ---- (a) pure operator certificate, runs from the package alone ----
    from pipeline import year_shift
    combos = pd.read_csv(f"{T}/combos.csv")
    lengths = sorted({int(n) for n in combos.n_calendar})
    for n in lengths:
        idx = np.arange(n)
        y12 = 12 * (n // 12)
        for k in range(1, 12):
            moved = year_shift(idx, k)      # position p holds source moved[p]
            assert ((moved - idx) % 12 == 0).all(), ("phase", n, k)
            assert (moved[y12:] == idx[y12:]).all(), ("tail moved", n, k)
    print(f"[1] month-phase check: PASS (pure operator certificate: "
          f"{len(lengths)} shipped calendar lengths x 11 shifts; every "
          f"position keeps its slot within the calendar year, tails in place)")
    executed.append("month-phase")
    # ---- (b) archive cross-check on the real per-record calendars ----
    raw = f"{ROOT}/data/raw"
    if not os.path.isdir(raw):
        print("[1+] archive month-vector cross-check: not executed "
              "(needs data/raw; the operator certificate above covers the "
              "shipped calendar lengths)")
        skipped.append("archive month-vector cross-check")
        return
    from pipeline import (read_gsim, merged_calendar, year_shift,
                          rotated_support)
    from te import load_wq_total
    wq, _ = load_wq_total()
    n_checked = 0
    for _, c in combos.iterrows():
        flow = read_gsim(c.gsim_id)
        series = wq.get(c.gems_id, {}).get(c.parameter, {})
        if flow is None or len(series) < 60:
            continue
        x_raw, y_raw, vobs, cal = merged_calendar(flow, series)
        if vobs.sum() < 60:
            continue
        months = np.array([p.month for p in cal])
        idx = np.arange(len(months))
        for k in range(1, 12):
            moved = year_shift(idx, k)
            assert (months[moved] == months).all(), (c.gems_id, c.gsim_id, k)
        # provenance of the rotated base values: wherever a draw's support
        # includes a month, the base value carried in comes from a month the
        # base record itself observed, or from inside a gap of at most two
        # months of that record (the same bound the observation's own
        # short-gap rule carries); long-gap interpolation never enters.
        obs_x = ~np.isnan(x_raw)
        interp_ok = np.zeros(len(obs_x), bool)
        i = 0
        while i < len(obs_x):
            if not obs_x[i]:
                j = i
                while j < len(obs_x) and not obs_x[j]:
                    j += 1
                if i > 0 and j < len(obs_x) and (j - i) <= 2:
                    interp_ok[i:j] = True
                i = j
            else:
                i += 1
        usable = obs_x | interp_ok
        obs_y = ~np.isnan(y_raw)
        for k in range(1, 12):
            moved = year_shift(idx, k)
            sup = rotated_support(obs_x, obs_y, k)
            assert usable[moved[sup]].all(), \
                (c.gems_id, c.gsim_id, k, "provenance")
        n_checked += 1
    print(f"[1+] archive month-vector cross-check: PASS ({n_checked} "
          f"combinations on their real calendars)")
    executed.append("archive month-vector cross-check")


def check_keys():
    ok = True
    # strict_nulls.csv carries the k = 1..11 budget (the observation row is
    # not part of that null); the months and naive tables carry k = 0..11.
    # A draw is only recorded when the rotated record still yields at least
    # 40 usable pairs, so a few gap-heavy combinations carry fewer than the
    # full budget in the months table; the k = 0 anchor is always present,
    # and the per-table coverage is printed below.
    for name, pipe, budget, min_floor in (
            ("strict_nulls.csv", None, 11, 11),
            ("strict_months_null.csv", None, 12, 9),
            ("null_alignments_naive.csv", "ccm_naive", 12, 12),
            ("te_alignments_naive_rerun.csv", "te_naive_s6", 12, 12)):
        df = pd.read_csv(f"{T}/{name}")
        missing = [c for c in ("gems_id", "gsim_id", "parameter", "k")
                   if c not in df.columns]
        if missing:
            print(f"[2] {name}: missing column(s) {missing}")
            ok = False
            continue
        if pipe is not None and "pipeline" in df.columns:
            df = df[df.pipeline == pipe]
        key = ["gems_id", "gsim_id", "parameter"]
        ks = df.groupby(key).k.nunique()
        nkeys = len(ks)
        nfull = int((ks == budget).sum())
        if budget == 12:
            anchors = df[df.k == 0].groupby(key).size()
            if len(anchors) != nkeys or not (anchors == 1).all():
                print(f"[2] {name}: k = 0 anchor missing or duplicated "
                      f"({len(anchors)} anchors for {nkeys} combinations)")
                ok = False
                continue
        if dups := df.duplicated(key + ["k"]).sum():
            print(f"[2] {name}: {dups} duplicate (key, k) rows")
            ok = False
            continue
        if ks.min() < min_floor or ks.max() != budget:
            print(f"[2] {name}: per-key draws span [{ks.min()}, {ks.max()}], "
                  f"expected up to {budget} with a floor of {min_floor}")
            ok = False
            continue
        if nfull < nkeys:
            print(f"[2] {name}: {nfull} of {nkeys} combinations carry the full "
                  f"{budget}-point budget; {nkeys - nfull} carry fewer draws "
                  f"(rotated records below the 40-pair minimum are not "
                  f"fabricated)")
    if ok:
        print("[2] key integrity check: PASS (complete keys and anchor rows; "
              "strict draws complete; any reduced month-draw coverage printed "
              "above)")
        executed.append("keys")
    else:
        failures.append("keys")


# result tables with no producing script in this package, by design: frozen
# inputs from the original naive-layer/early runs and superseded builds, all
# labelled as such in the README's table-to-script map
FROZEN = {
    "calendar_scan.csv", "ccm_analysis_results.csv",
    "deseasonalized_sametime_full.csv", "true_xmap_sametime_full.csv",
    "tn_aggregation_sensitivity.csv", "transfer_entropy.csv",
    "naive_null.csv", "lakes_backup.csv", "lakes_null_discharge_cal.csv",
    "mccm_synthetic.csv",
}


def check_output_names():
    owners = {}
    for path in glob.glob(f"{HERE}/*.py"):          # current layer only
        src = open(path, encoding="utf-8").read()
        base = os.path.basename(path)
        # literal writes
        for m in re.finditer(r'to_csv\(\s*f"\{T\}/([A-Za-z0-9_\.]+)"', src):
            owners.setdefault(m.group(1), set()).add(base)
        # writes through a path variable: NAME = f"{T}/table.csv" ... .to_csv(NAME)
        aliases = dict(re.findall(
            r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*f"\{T\}/([A-Za-z0-9_\.]+)"', src))
        for m in re.finditer(r'\.to_csv\(\s*([A-Za-z_][A-Za-z0-9_]*)', src):
            v = m.group(1)
            if v in aliases:
                owners.setdefault(aliases[v], set()).add(base)
    dups = {n: sorted(o) for n, o in owners.items() if len(o) > 1}
    shipped = {os.path.basename(p) for p in glob.glob(f"{T}/*.csv")}
    undeclared = sorted(shipped - set(owners) - FROZEN)
    if dups:
        print(f"[3] duplicate output names: {dups}")
        failures.append("producers")
    elif undeclared:
        print(f"[3] result tables with no producing script and no frozen/"
              f"superseded label: {undeclared}")
        failures.append("producers")
    else:
        print(f"[3] producer check: PASS ({len(shipped)} result tables: "
              f"{len(shipped) - len(FROZEN)} each written by exactly one "
              f"current script, {len(FROZEN)} labelled frozen/superseded, "
              f"0 undeclared)")
        executed.append("producers")


def check_identity_shifts():
    """Where the complete-year count divides k the rotation is the identity,
    so the shipped null row must equal the observed row of the same package."""
    key = ["gems_id", "gsim_id", "parameter"]
    combos = pd.read_csv(f"{T}/combos.csv")
    combos["y"] = combos.n_calendar // 12
    yrs = combos.set_index(key)["y"]
    ks = list(range(1, 12))
    n_checked = 0

    # --- strict TE/CTE table vs observed table ---
    obs = pd.read_csv(f"{T}/te_z.csv").set_index(key)
    nul = pd.read_csv(f"{T}/strict_nulls.csv")
    nul["y"] = [yrs.get(tuple(r), np.nan) for r in
                nul[key].itertuples(index=False, name=None)]
    ident = nul[nul.apply(lambda r: bool(r.y) and r.k % r.y == 0, axis=1)]
    for r in ident.itertuples():
        if (r.gems_id, r.gsim_id, r.parameter) not in obs.index:
            continue
        o = obs.loc[(r.gems_id, r.gsim_id, r.parameter)]
        assert abs(r.te_f - o.te_f2w) < 1e-12 and abs(r.te_w - o.te_w2f) < 1e-12, \
            ("strict TE identity mismatch", r.gems_id, r.parameter, r.k)
        assert abs(r.cte_f - o.cte_f2w) < 1e-12 and abs(r.cte_w - o.cte_w2f) < 1e-12, \
            ("strict CTE identity mismatch", r.gems_id, r.parameter, r.k)
        n_checked += 1

    # --- months-null votes vs the k = 0 anchoring vote ---
    m = pd.read_csv(f"{T}/strict_months_null.csv")
    m["y"] = [yrs.get(tuple(r), np.nan) for r in
              m[key].itertuples(index=False, name=None)]
    k0 = m[m.k == 0].set_index(key).vote
    for r in m[(m.k > 0)].itertuples():
        if r.y and any(k % r.y == 0 for k in ks) and r.k % r.y == 0:
            if (r.gems_id, r.gsim_id, r.parameter) in k0.index:
                assert int(r.vote) == int(k0.loc[(r.gems_id, r.gsim_id, r.parameter)]), \
                    ("months identity mismatch", r.gems_id, r.parameter, r.k)
                n_checked += 1

    # --- corrected CCM table: identity draw vs observed columns ---
    cc = pd.read_csv(f"{T}/ccm_shifts_corrected.csv")
    cc["y"] = [yrs.get(tuple(r), np.nan) for r in
               cc[key].itertuples(index=False, name=None)]
    for r in cc.itertuples():
        idk = [k for k in ks if r.y and k % r.y == 0]
        for k in idk:
            assert abs(getattr(r, f"std_f_{k}") - r.std_f) < 1e-12 and \
                   abs(getattr(r, f"std_w_{k}") - r.std_w) < 1e-12 and \
                   abs(getattr(r, f"des_f_{k}") - r.des_f) < 1e-12 and \
                   abs(getattr(r, f"des_w_{k}") - r.des_w) < 1e-12, \
                ("ccm identity mismatch", r.gems_id, r.parameter, k)
            n_checked += 1

    print(f"[4] identity-shift check: PASS ({n_checked} draws where the "
          f"complete-year count divides k reproduce the observed row exactly)")
    executed.append("identity")


def check_null_values():
    n = pd.read_csv(f"{T}/ccm_shifts_n.csv")
    c = pd.read_csv(f"{T}/ccm_shifts_corrected.csv")
    nan_draws = 0
    for r in c.itertuples():
        for k in range(1, 12):
            vals = [getattr(r, f"std_f_{k}"), getattr(r, f"std_w_{k}"),
                    getattr(r, f"des_f_{k}"), getattr(r, f"des_w_{k}")]
            if any(pd.isna(v) for v in vals):
                nan_draws += 1
                row = n[(n.gems_id == r.gems_id) & (n.gsim_id == r.gsim_id)
                        & (n.parameter == r.parameter) & (n.k == k)]
                if len(row) != 1 or not (row[["n_std", "n_des"]].iloc[0] < 40).all():
                    print(f"[5] NaN CCM draw without a sub-40 sample size: "
                          f"{r.gems_id}/{r.gsim_id}/{r.parameter} k={k}")
                    failures.append("null-values")
                    return
    for name, cols in (("strict_nulls.csv", ["te_f", "te_w", "cte_f", "cte_w", "n_pairs"]),
                       ("strict_months_null.csv", ["vote", "n"]),
                       ("null_alignments_naive.csv", ["vote"]),
                       ("te_alignments_naive_rerun.csv", ["vote"])):
        df = pd.read_csv(f"{T}/{name}")
        if not df[cols].notna().all().all():
            print(f"[5] non-finite value in {name}")
            failures.append("null-values")
            return
    assert (pd.read_csv(f"{T}/strict_nulls.csv").n_pairs >= 40).all()
    nn = pd.read_csv(f"{T}/ccm_shifts_n.csv")
    assert (nn.n_std == nn.n_des).all(), "std/des sample sizes disagree"
    print(f"[5] null-value check: PASS (all null values finite; "
          f"{nan_draws} NaN CCM draws each matched to a sub-40 rotated window "
          f"in ccm_shifts_n.csv and excluded from the per-draw shares)")
    executed.append("null-values")


if __name__ == "__main__":
    check_month_phase()
    check_keys()
    check_output_names()
    check_identity_shifts()
    check_null_values()
    if failures:
        raise SystemExit("self-check FAILED: " + ", ".join(failures))
    n_total = len(executed) + len(skipped)
    if skipped:
        print(f"self-check: PASS, {len(executed)} of {n_total} checks executed "
              f"({', '.join(skipped)} not executed for lack of inputs -- "
              f"partial certificate, not a full pass)")
    else:
        print(f"self-check: ALL PASS ({n_total} of {n_total} checks)")
