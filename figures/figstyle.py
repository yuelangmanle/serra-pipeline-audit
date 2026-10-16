#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified figure style for the v39 figure suite (SERRA / Springer compliant).

Standard (fixed 2026-10-15, applies to every figure in the manuscript):
  * Canvas width: 174 mm = 6.85 in (Springer single-column text area; the old
    7.2 in canvas exceeded the 174 mm limit).
  * Fonts: Arial everywhere, nothing below 8 pt at final size (SERRA asks for
    8-12 pt lettering; the old suite mixed 7.2-9.5 pt).
  * No titles inside figures (SERRA: "no titles or captions inside
    illustrations"); panel titles move to the manuscript captions.
  * Panel labels: bold lowercase a, b, c at a fixed 6 pt offset from the axes
    corner (axes-fraction offsets drift between tall and short panels).
  * Export: vector PDF + EPS with editable TrueType text, 600 dpi RGB TIFF
    (combination art >= 600 dpi) and a 600 dpi PNG preview.
  * One restrained palette: the five pipeline methods keep one Okabe-Ito hue
    each, identical across all figures; neutral gray is reserved for naive /
    baseline states and gray bands for null ranges.
  * Every multi-panel figure passes the 1.5 pt panel-alignment gate before
    export; every exported PDF passes the collision audit afterwards.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

# panel-alignment gate: the helper ships next to this file
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

# Springer single-column text area
FULL_W = 174 / 25.4          # 6.850 in
HALF_W = 84 / 25.4           # 3.307 in
BASE_PT = 8.0                # SERRA lettering floor

# Okabe-Ito, one hue per pipeline method, shared by every figure
METHOD_COLORS = {
    "Standard CCM": "#0072B2",
    "Deseason. CCM": "#56B4E9",
    "M-CCM": "#D55E00",
    "TE (unconditioned)": "#CC79A7",
    "Conditional TE": "#009E73",
}
METHOD_ORDER = list(METHOD_COLORS)
# neutral roles
NAIVE_GRAY = "#9A9A9A"
NULL_BAND = "#D9D9D9"
CHANCE_GRAY = "#6E6E6E"
LAKE_POYANG = "#0072B2"
LAKE_TAIHU = "#009E73"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",       # editable text in SVG
    "pdf.fonttype": 42,           # editable TrueType text in PDF
    "ps.fonttype": 42,            # editable TrueType text in EPS
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT,
    "xtick.labelsize": BASE_PT,
    "ytick.labelsize": BASE_PT,
    "legend.fontsize": BASE_PT,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    # no savefig.bbox='tight': the exported canvas is exactly figsize
    "savefig.bbox": None,
    "mathtext.fontset": "dejavusans",
})


def panel_label(ax, label, x=0.0, y=1.0, x_off=-14, y_off=4, fs=10):
    """Bold lowercase panel label at a fixed physical offset from the corner."""
    offset = ScaledTranslation(x_off / 72.0, y_off / 72.0, ax.figure.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + offset,
            fontsize=fs, fontweight="bold", ha="left", va="bottom")


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def alignment_gate(fig, filename, row_groups=None, column_groups=None,
                   exemptions=None, exclude_axes=None):
    """Run the 1.5 pt panel-alignment gate; returns True when passed/NA."""
    base = os.path.splitext(filename)[0]
    opts = {"require_panel_labels": True}
    if row_groups is not None:
        opts["row_groups"] = row_groups
    if column_groups is not None:
        opts["column_groups"] = column_groups
    if exemptions is not None:
        opts["exemptions"] = exemptions
    if exclude_axes is not None:
        opts["exclude_axes"] = exclude_axes
    try:
        require_matplotlib_panel_alignment(
            fig,
            json_out=f"{base}.alignment.json",
            overlay_svg=f"{base}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
            **opts,
        )
        return True
    except Exception as exc:  # PanelAlignmentError or NOT AUDITABLE
        print(f"ALIGNMENT GATE ISSUE {base}: {exc}")
        return False


def save_pub(fig, path_no_ext, dpi=600, formats=("pdf", "eps", "tif", "png")):
    """Export the submission bundle; TIFF at 600 dpi flattened to RGB."""
    for fmt in formats:
        p = f"{path_no_ext}.{fmt}"
        if fmt == "tif":
            fig.savefig(p, dpi=dpi)
            from PIL import Image
            with Image.open(p) as im:
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(p, compression="tiff_lzw", dpi=(600, 600))
        elif fmt == "png":
            fig.savefig(p, dpi=dpi)
        else:
            fig.savefig(p, dpi=dpi)
        print(f"  wrote {p}")


def wilson_ci(share_pct, n, z=1.959963985):
    """Wilson score interval for a percentage share, returned in percent."""
    import numpy as np
    p = share_pct / 100.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return 100.0 * (center - half), 100.0 * (center + half)
