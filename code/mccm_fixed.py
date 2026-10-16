#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The corrected M-CCM table under the documented label convention.

The estimates are one set; the label convention is the repair (manuscript
Section 2.5): the bidirectional routine assigns labels by argument order, so
the swapped-label table carries the two directions exchanged. This script
applies the documented convention to the swapped-label table and writes the
repaired table the manuscript's 85.9% headline reads from.

Writes data/results/tables/mccm_fixed.csv.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PAPER_BASE", os.path.dirname(HERE))
T = f"{ROOT}/data/results/tables"


def main():
    sw = pd.read_csv(f"{T}/mccm_naive_labels_swapped.csv")
    fixed = sw.copy()
    fixed["mccm_f2w"] = sw["mccm_w2f"]
    fixed["mccm_w2f"] = sw["mccm_f2w"]
    fixed.to_csv(f"{T}/mccm_fixed.csv", index=False)
    valid = fixed.dropna(subset=["mccm_f2w", "mccm_w2f"])
    dom = (valid.mccm_f2w > valid.mccm_w2f).mean()
    print(f"mccm_fixed: n = {len(valid)} valid of {len(fixed)} rows  "
          f"flow-dominant share = {100*dom:.1f}%")


if __name__ == "__main__":
    main()
