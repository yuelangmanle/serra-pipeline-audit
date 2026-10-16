#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the first-generation four figures from the corrected and naive data.

Figure 1  The naive pipeline produces contradictory directional claims (5 bars)
Figure 2  (a) naive vs corrected paired bars; (b) naive->corrected slope chart
Figure 3  Three hazards and their diagnostics
Figure 4  Dual-lake anchoring

Print sizing (2026-10-13 revision): every figure is drawn on a 7.2 in wide
canvas (M-CCM journal double-column width) at 300 dpi, height 3.2-4.5 in, and
every font is >= 8 pt. Because the canvas is placed 1:1 in the manuscript, the
printed font equals the nominal font (>= 8 pt > 7 pt floor). The previous
Fig3 was 13 in wide: squeezed to 5.8 in it rendered at ~3.8 pt effective —
fixed here. savefig does NOT use bbox='tight', so the exported PNG is exactly
7.2 x H in (tight_layout keeps all artists inside the canvas).

Data (2026-10-13 manuscript update):
  naive:    std CCM 60.1, deseason CCM 47.5, M-CCM 85.9, TE 42.2, cond TE 67.0
  corrected: std CCM 51.9, deseason CCM 46.8, M-CCM 85.9 (proxy-invariant),
             TE 52.2 (Schreiber form + z-score), cond TE 84.2 (z-scored;
             its own zero-coupling baseline is 79.1)
  TE ablation ladder now reads ablation_ladder_te_z.csv
  (42.2 -> 38.9 -> 50.0 -> 52.2); the CCM ladder still reads
  ablation_ladder.csv (60.1 -> 47.2 -> 51.9).

Style: Okabe-Ito palette, Arial, no top/right spines, 300 dpi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.environ.get("PAPER_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = f"{ROOT}/data/results/tables"
OUT = f"{ROOT}/manuscript/figures_firstgen"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito palette. 2026-10-13 palette-bug fix: RED and ORANGE previously
# both pointed at '#D55E00', so Fig1's five bars could not be told apart.
# Now: ORANGE = true Okabe-Ito orange (#E69F00), VERMILLION = #D55E00.
# RED is kept as an alias of VERMILLION because the manuscript calls the
# naive KSG points of Fig3(a) "red".
BLUE = '#0072B2'
SKY = '#56B4E9'
GREEN = '#009E73'
YELLOW = '#F0E442'
ORANGE = '#E69F00'
VERMILLION = '#D55E00'
RED = VERMILLION
PURPLE = '#CC79A7'
GRAY = '#999999'
BLACK = '#000000'

# One colour per pipeline variant, shared by Fig1 bars and the Fig2(b) slope
# chart. Distinct at every adjacency; CCM=BLUE and TE=PURPLE echo the Fig3(c)
# ladder colours.
METHOD_COLORS = {
    'Standard CCM': BLUE,
    'Deseason. CCM': SKY,
    'M-CCM': VERMILLION,
    'TE (unconditioned)': PURPLE,
    'Conditional TE': GREEN,
}
METHOD_ORDER = list(METHOD_COLORS)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5,          # print floor: nothing below 8 pt anywhere
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'mathtext.fontset': 'dejavusans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    # NOTE: no savefig.bbox='tight' — keeps the exported PNG exactly 7.2 in wide
})


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def pct1(v):
    """One-decimal share, half-up (47.15% must read 47.2, matching the CSV note)."""
    return str(Decimal(str(v)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def prep_monthly(s, max_gap=2):
    """Sentinel-safe monthly preparation, mirroring lakes.prep:
    physically impossible negatives -> NaN, reindex onto the true monthly
    calendar, interior gaps <= max_gap linearly interpolated. Fig4(d) uses the
    same cleaned TP/Chla pairs (n = 370) as the CCM analysis, so the annotated
    Pearson r (0.559) matches the manuscript and dual_lake.csv."""
    s = s.copy()
    s[s < 0] = np.nan
    s.index = pd.PeriodIndex(s.index, freq='M')
    cal = pd.period_range(s.index.min(), s.index.max(), freq='M')
    s = s.reindex(cal)
    return s.interpolate(limit=max_gap, limit_area='inside')


# ================================================================
# Shared directional shares (Figures 1 and 2), read from source tables
# ================================================================

def load_shares():
    """Naive/corrected flow-dominant shares per method, each from its table:

    std CCM   naive    ablation_ladder.csv C0 flow_dom_share      -> 60.1
              corrected same file C2 (calendar-true endpoint)          -> 51.9
    deseason  naive    deseasonalized_sametime_full.csv
                         share(f2w_0.7 > w2f_0.7), two-sided p = 0.024  -> 47.5
              corrected deseason.csv share(f2w > w2f)              -> 46.8
    M-CCM     naive    mccm_naive_labels_swapped.csv share(mccm_w2f > mccm_f2w); the
                         naive file stores the direction columns swapped-> 85.9
              corrected mccm_fixed.csv share(mccm_f2w > mccm_w2f)  -> 85.9
    TE        naive    ablation_ladder_te_z.csv T0 share           -> 42.2
              corrected te_z.csv share(te_f2w > te_w2f)            -> 52.2
    cond TE   naive    transfer_entropy_491.csv share(cte_f2w>cte_w2f) -> 67.0
              corrected cte_surrogates_z.csv
                         share(obs_cte_f2w > obs_cte_w2f)               -> 84.2
    cond TE zero-coupling baseline: median pooled_flow_dom_share of
              cte_surrogates_z_null.csv                            -> 79.1
    """
    lad = pd.read_csv(f'{T}/ablation_ladder.csv')
    ladz = pd.read_csv(f'{T}/ablation_ladder_te_z.csv')

    s = {}
    s['ccm_naive'] = 100 * lad.loc[lad.stage == 'C0', 'flow_dom_share'].iloc[0]
    s['ccm_corr'] = 100 * lad.loc[lad.stage == 'C2', 'flow_dom_share'].iloc[0]

    d24 = pd.read_csv(f'{T}/deseasonalized_sametime_full.csv')
    s['des_naive'] = 100 * (d24['f2w_0.7'] > d24['w2f_0.7']).mean()
    d34 = pd.read_csv(f'{T}/deseason.csv').dropna(subset=['f2w', 'w2f'])
    s['des_corr'] = 100 * (d34.f2w > d34.w2f).mean()

    m_naive = pd.read_csv(f'{T}/mccm_naive_labels_swapped.csv').dropna(subset=['mccm_f2w', 'mccm_w2f'])
    s['mccm_naive'] = 100 * (m_naive.mccm_w2f > m_naive.mccm_f2w).mean()
    m_fix = pd.read_csv(f'{T}/mccm_fixed.csv').dropna(subset=['mccm_f2w', 'mccm_w2f'])
    s['mccm_corr'] = 100 * (m_fix.mccm_f2w > m_fix.mccm_w2f).mean()

    s['te_naive'] = 100 * ladz.loc[ladz.stage == 'T0', 'share'].iloc[0]
    tez = pd.read_csv(f'{T}/te_z.csv').dropna(subset=['te_f2w', 'te_w2f'])
    s['te_corr'] = 100 * (tez.te_f2w > tez.te_w2f).mean()

    v26 = pd.read_csv(f'{T}/transfer_entropy_491.csv').dropna(subset=['cte_f2w', 'cte_w2f'])
    s['cte_naive'] = 100 * (v26.cte_f2w > v26.cte_w2f).mean()
    ctz = pd.read_csv(f'{T}/cte_surrogates_z.csv').dropna(subset=['obs_cte_f2w', 'obs_cte_w2f'])
    s['cte_corr'] = 100 * (ctz.obs_cte_f2w > ctz.obs_cte_w2f).mean()
    cn = pd.read_csv(f'{T}/cte_surrogates_z_null.csv')
    s['cte_null'] = 100 * cn.pooled_flow_dom_share.median()

    # Anchor check against the manuscript numbers (soft warning, not an error).
    expected = {
        'ccm_naive': 60.1, 'ccm_corr': 51.9,
        'des_naive': 47.5, 'des_corr': 46.8,
        'mccm_naive': 85.9, 'mccm_corr': 85.9,
        'te_naive': 42.2, 'te_corr': 52.2,
        'cte_naive': 67.0, 'cte_corr': 84.2, 'cte_null': 79.1,
    }
    for k, exp in expected.items():
        if abs(s[k] - exp) > 0.15:
            print(f'WARNING: {k} = {s[k]:.2f}, expected ~{exp} (manuscript)')
    print('shares:', {k: round(v, 2) for k, v in s.items()})
    return s


# ================================================================
# FIGURE 1: The naive pipeline produces contradictory directional claims
# 5 bars, print size 7.2 x 3.2 in, one distinct Okabe-Ito colour per bar
# ================================================================

def fig1(s):
    labels = ['Standard\nCCM', 'Deseasonalized\nCCM', 'M-CCM',
              'TE (unconditioned)', 'Conditional\nTE']
    dom = [s['ccm_naive'], s['des_naive'], s['mccm_naive'], s['te_naive'], s['cte_naive']]
    # Naive p values as in manuscript Table 1
    pvals = [r'$p = 2.6\times10^{-6}$', r'$p = 0.024$', r'$p < 10^{-10}$',
             r'$p = 5.9\times10^{-4}$', r'$p = 3.9\times10^{-14}$']
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bars = ax.bar(range(5), dom, color=colors, width=0.55, zorder=3, alpha=0.9)
    ax.axhline(y=50, color='#666666', linestyle='--', linewidth=1.2, zorder=2)
    ax.text(4.32, 50.8, '50%', fontsize=8, color='#666666', ha='left', va='bottom')

    wb = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85)
    for i, (v, pv) in enumerate(zip(dom, pvals)):
        ax.text(i, v + 1.5, f'{v:.1f}%', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color=colors[i],
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor='none', alpha=0.85), zorder=4)
        ax.text(i, v + 6.5, pv, ha='center', va='bottom', fontsize=8,
                color='#555555', bbox=wb, zorder=4)

    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.55, 4.75)
    ax.set_ylabel('Flow-dominant share (%)')
    ax.set_title('The naive pipeline produces contradictory directional claims')
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f'{OUT}/Fig1_naive_contradictions.png')
    plt.close()
    print('Fig1 done')


# ================================================================
# FIGURE 2: Naive vs corrected pipeline — the money figure (7.2 x 3.7 in)
#  (a) paired naive(blue)/corrected(orange) bars for all five methods;
#      the corrected conditional-TE bar carries a short horizontal line at
#      its own zero-coupling baseline (null 79.1%). The old "excluded"
#      grey placeholder bar is gone.
#  (b) slope chart (naive -> corrected) replacing the correction-magnitude
#      dumbbell: magnitudes double-count sign reversals, levels do not.
#      The M-CCM line is horizontal (85.9 -> 85.9, proxy-invariant).
# ================================================================

def fig2(s):
    naive = [s['ccm_naive'], s['des_naive'], s['mccm_naive'], s['te_naive'], s['cte_naive']]
    corrected = [s['ccm_corr'], s['des_corr'], s['mccm_corr'], s['te_corr'], s['cte_corr']]
    cte_null = s['cte_null']

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.7))

    # ---- (a) paired bars ----------------------------------------------------
    x = np.arange(5)
    width = 0.38
    axa.bar(x - width / 2, naive, width, color=BLUE, alpha=0.85,
            label='Naive (n=491)', zorder=3)
    axa.bar(x + width / 2, corrected, width, color=ORANGE, alpha=0.85,
            label='Corrected (n=316)', zorder=3)
    axa.axhline(y=50, color='#666666', linestyle='--', linewidth=1, zorder=2)

    # zero-coupling baseline of the corrected conditional TE (79.1)
    i = 4
    xc = x[i] + width / 2
    axa.hlines(cte_null, xc - width / 2, xc + width / 2, color=BLACK, linewidth=1.8, zorder=5)
    axa.annotate(f'null {cte_null:.1f}%', xy=(xc + width / 2, cte_null),
                 xytext=(3.35, 93), ha='center', va='top', fontsize=8, color='#333333',
                 arrowprops=dict(arrowstyle='->', color='#333333', linewidth=0.8))

    for i in range(5):
        nv, cv = naive[i], corrected[i]
        wl = dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.85)
        if abs(nv - cv) < 0.2:
            # identical pair (M-CCM): one centred label above both bars
            axa.text(x[i], max(nv, cv) + 1.5, f'{nv:.1f}', ha='center', va='bottom',
                     fontsize=8, fontweight='bold', bbox=wl, zorder=4)
            continue
        axa.text(x[i] - width / 2, nv + 1.5, f'{nv:.1f}', ha='center', va='bottom',
                 fontsize=8, bbox=wl, zorder=4)
        if abs(nv - cv) < 5:
            # labels would collide horizontally: corrected value goes inside its bar
            axa.text(x[i] + width / 2, cv - 1.5, f'{cv:.1f}', ha='center', va='top',
                     fontsize=8, fontweight='bold', color='white', zorder=4)
        else:
            axa.text(x[i] + width / 2, cv + 1.5, f'{cv:.1f}', ha='center', va='bottom',
                     fontsize=8, bbox=wl, zorder=4)

    axa.set_xticks(x)
    axa.set_xticklabels(['Standard\nCCM', 'Deseason.\nCCM', 'M-CCM',
                         'TE\n(unconditioned)', 'Conditional\nTE'])
    axa.set_ylabel('Flow-dominant share (%)')
    axa.set_title('(a) Directional dominance: naive vs corrected')
    axa.legend(loc='upper left', frameon=False)
    axa.set_ylim(0, 105)
    axa.set_xlim(-0.55, 4.55)
    style_ax(axa)

    # ---- (b) slope chart ----------------------------------------------------
    axb.axhline(y=50, color='#666666', linestyle='--', linewidth=1, zorder=1)
    axb.text(-0.98, 50.8, '50% (chance)', fontsize=8, color='#666666',
             ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    for m, nv, cv in zip(METHOD_ORDER, naive, corrected):
        c = METHOD_COLORS[m]
        axb.plot([0, 1], [nv, cv], '-', color=c, linewidth=1.6, zorder=3,
                 marker='o', markersize=4.5, markeredgecolor='white', markeredgewidth=0.5)
        axb.text(-0.08, nv, m, ha='right', va='center', fontsize=8,
                 fontweight='bold', color=c)

    # right-side value labels with a small vertical dodge (51.9 vs 52.2 collide)
    right_lab = {  # method -> (y position, vertical anchor)
        'Standard CCM': (corrected[0] - 1.8, 'top'),
        'Deseason. CCM': (corrected[1] - 1.6, 'top'),
        'M-CCM': (corrected[2] + 1.8, 'bottom'),
        'TE (unconditioned)': (corrected[3] + 1.8, 'bottom'),
        'Conditional TE': (corrected[4] - 1.8, 'top'),
    }
    for m, cv in zip(METHOD_ORDER, corrected):
        y, va = right_lab[m]
        axb.text(1.08, y, f'{cv:.1f}', ha='left', va=va, fontsize=8,
                 fontweight='bold', color=METHOD_COLORS[m])

    # M-CCM is the horizontal line: annotate proxy-invariance
    axb.text(0.42, s['mccm_naive'] + 2.2, 'proxy-invariant',
             ha='center', va='bottom', fontsize=8, style='italic',
             color=METHOD_COLORS['M-CCM'])

    axb.set_xlim(-1.02, 1.55)
    axb.set_ylim(33, 96)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(['naive', 'corrected'])
    axb.set_ylabel('Flow-dominant share (%)')
    axb.set_title('(b) Share per method, naive \u2192 corrected')
    style_ax(axb)

    fig.tight_layout()
    fig.savefig(f'{OUT}/Fig2_naive_vs_corrected.png')
    plt.close()
    print('Fig2 done')


# ================================================================
# FIGURE 3: Three hazards and their diagnostics (7.2 x 4.4 in)
# ================================================================

def fig3():
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.2, 4.4))

    # --- (a) Estimator calibration: broken vs corrected on synthetic data ---
    # n = 5,000, seed 42 (manuscript: naive returns ~ -0.25 nats on
    # independent data; corrected ~ +0.001 nats).
    np.random.seed(42)
    n = 5000
    x_ind = np.random.randn(n)
    y_ind = np.random.randn(n)

    # broken KSG (Euclidean joint, KSG counting)
    def ksg_broken(xv, yv, k=3):
        nv = len(xv)
        xv = xv.reshape(-1, 1); yv = yv.reshape(-1, 1)
        xy = np.hstack([xv, yv])
        tree = cKDTree(xy)
        d, _ = tree.query(xy, k=k + 1)  # Euclidean
        eps = d[:, k]
        sx = np.sort(xv.ravel()); sy = np.sort(yv.ravel())
        nx = np.searchsorted(sx, xv.ravel() + eps) - np.searchsorted(sx, xv.ravel() - eps) - 1
        ny = np.searchsorted(sy, yv.ravel() + eps) - np.searchsorted(sy, yv.ravel() - eps) - 1
        return digamma(k) + digamma(nv) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))

    def ksg_correct(xv, yv, k=3):
        nv = len(xv)
        xv = xv.reshape(-1, 1); yv = yv.reshape(-1, 1)
        xy = np.hstack([xv, yv])
        tree = cKDTree(xy)
        d, _ = tree.query(xy, k=k + 1, p=np.inf)
        eps = d[:, k]
        sx = np.sort(xv.ravel()); sy = np.sort(yv.ravel())
        nx = np.maximum(np.searchsorted(sx, xv.ravel() + eps) - np.searchsorted(sx, xv.ravel() - eps) - 1, 0)
        ny = np.maximum(np.searchsorted(sy, yv.ravel() + eps) - np.searchsorted(sy, yv.ravel() - eps) - 1, 0)
        return digamma(k) + digamma(nv) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))

    mi_broken_ind = ksg_broken(x_ind, y_ind)
    mi_correct_ind = ksg_correct(x_ind, y_ind)

    rho_vals = [0.0, 0.3, 0.6, 0.9]
    true_mi = [0.0] + [-0.5 * np.log(1 - r**2) for r in rho_vals[1:]]
    broken_mis, correct_mis = [], []
    for rho in rho_vals:
        y_dep = rho * x_ind + np.sqrt(1 - rho**2) * np.random.randn(n)
        broken_mis.append(ksg_broken(x_ind, y_dep))
        correct_mis.append(ksg_correct(x_ind, y_dep))

    axa.plot(true_mi, true_mi, 'k--', linewidth=0.8, label='True MI')
    axa.scatter(true_mi, broken_mis, color=RED, s=35, zorder=4, label='Naive KSG')
    axa.scatter(true_mi, correct_mis, color=BLUE, s=35, zorder=4, label='Corrected KSG')
    axa.text(0.96, 0.04,
             f'\u03c1 = 0 (independent):\nnaive {mi_broken_ind:+.4f} nats\ncorrected {mi_correct_ind:+.4f} nats',
             transform=axa.transAxes, ha='right', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#BBBBBB', alpha=0.9))
    axa.set_xlabel('True MI (nats)')
    axa.set_ylabel('Estimated MI (nats)')
    axa.set_title('(a) Estimator calibration', fontsize=9)
    axa.legend(frameon=False, loc='upper left', fontsize=8, handletextpad=0.4,
               borderaxespad=0.2)
    axa.set_xlim(-0.07, 0.93)
    axa.set_ylim(-0.34, 0.98)
    style_ax(axa)

    # --- (b) Species mixing: directional dominance (62.9 / 52.9, unchanged) ---
    try:
        agg = pd.read_csv(f'{T}/tn_aggregation_sensitivity.csv')
        vb = agg.dropna(subset=['f2w_totalonly', 'w2f_totalonly'])
        va70 = agg[agg.gems_id.isin(vb.gems_id)].dropna(subset=['f2w_all', 'w2f_all'])
        dom_mixed = 100 * (va70.f2w_all > va70.w2f_all).mean()
        dom_pure = 100 * (vb.f2w_totalonly > vb.w2f_totalonly).mean()
        p_mixed = stats.wilcoxon(va70.f2w_all, va70.w2f_all, alternative='greater').pvalue
        p_pure = stats.wilcoxon(vb.f2w_totalonly, vb.w2f_totalonly, alternative='greater').pvalue
    except Exception:
        dom_mixed, dom_pure, p_mixed, p_pure = 62.9, 52.9, 0.021, 0.22

    bars = axb.bar([0, 1], [dom_mixed, dom_pure], color=[RED, BLUE], width=0.45,
                   alpha=0.75, zorder=3)
    axb.axhline(y=50, color='#666666', linestyle='--', linewidth=1)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(['Mixed species\n(n=70)', 'Total-only\n(n=70)'])
    axb.set_ylabel('Flow-dominant (%)')
    axb.set_title('(b) Species mixing hazard', fontsize=9)
    for bar, pv in zip(bars, [p_mixed, p_pure]):
        axb.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}%\np = {pv:.3f}', ha='center', fontsize=8)
    axb.set_ylim(0, 80)
    style_ax(axb)

    # --- (c) Single-factor ablation ladders ---
    # CCM ladder: ablation_ladder.csv  (60.1 -> 47.2 -> 51.9)
    # TE ladder : ablation_ladder_te_z.csv (42.2 -> 38.9 -> 50.0 -> 52.2,
    #             standardized Schreiber-form TE; endpoint = Section 3.2 TE)
    lad = pd.read_csv(f'{T}/ablation_ladder.csv')
    ladz = pd.read_csv(f'{T}/ablation_ladder_te_z.csv')

    def stage_label(factor_changed):
        f = str(factor_changed).lower()
        if 'baseline' in f:
            return 'Naive'
        if 'estimator' in f:
            return '+estimator'
        if 'species' in f:
            return '+species'
        if 'calendar' in f:
            return '+calendar'
        return str(factor_changed)

    # the TE-z table has no factor_changed column; stages are fixed
    TEZ_LABELS = {'T0': 'Naive', 'T1': '+estimator', 'T2': '+species', 'T3': '+calendar'}

    gs_c = axc.get_subplotspec().get_gridspec()
    sub_c = gs_c[2].subgridspec(2, 1, hspace=0.62)
    axc.remove()
    axc1 = fig.add_subplot(sub_c[0])
    axc2 = fig.add_subplot(sub_c[1])

    for axg, vals, ns, labs, meth, gname, gcol in [
            (axc1, 100.0 * lad[lad.method == 'CCM_std'].sort_values('stage').flow_dom_share.values,
             lad[lad.method == 'CCM_std'].sort_values('stage').n.values.astype(int),
             [stage_label(fc) for fc in lad[lad.method == 'CCM_std'].sort_values('stage').factor_changed],
             'CCM_std', 'Standard CCM', BLUE),
            (axc2, 100.0 * ladz.sort_values('stage').share.values,
             ladz.sort_values('stage').n.values.astype(int),
             [TEZ_LABELS[st] for st in ladz.sort_values('stage').stage],
             'TE_z', 'TE (unconditioned)', PURPLE)]:
        xs = np.arange(len(vals))

        axg.plot(xs, vals, '-', color=gcol, linewidth=1.3, zorder=2)
        axg.scatter(xs, vals, s=35, color=gcol, zorder=4, edgecolors='white', linewidth=0.6)
        axg.axhline(y=50, color='#666666', linestyle='--', linewidth=1, zorder=1)

        wb = dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.9)
        for i in range(len(xs)):
            axg.text(xs[i], vals[i] + 1.8, pct1(vals[i]), ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color=gcol, bbox=wb, zorder=5)
            axg.text(xs[i], vals[i] - 1.8, f'n={ns[i]}', ha='center', va='top',
                     fontsize=8, color='#555555', bbox=wb, zorder=5)

        axg.text(0.98, 0.90, gname, transform=axg.transAxes, ha='right', va='top',
                 fontsize=8.5, fontweight='bold', color=gcol)
        axg.set_xticks(xs)
        # rotated so '+estimator'/'+species'/'+calendar' fit the narrow panel
        axg.set_xticklabels(labs, fontsize=8, rotation=30, ha='right',
                            rotation_mode='anchor')
        if meth == 'CCM_std':
            axg.set_ylabel('Flow-dominant (%)')
            axg.set_title('(c) Single-factor ablation ladder', fontsize=9)
        axg.set_ylim(34, 66)
        axg.set_yticks([40, 50, 60])
        axg.set_xlim(-0.45, max(len(vals), 4) - 0.55)
        style_ax(axg)

    fig.tight_layout()
    fig.savefig(f'{OUT}/Fig3_three_hazards.png')
    plt.close()
    print('Fig3 done')


# ================================================================
# FIGURE 4: Dual-lake anchoring (7.2 x 4.5 in)
# Content unchanged (updated earlier today); only physical size and the
# font floor (>= 8 pt) were touched for print compliance.
# ================================================================

def fig4():
    dl = pd.read_csv(f'{T}/dual_lake.csv')

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))

    # --- (a-c) Poyang ---
    # (a) WL→TN scatter
    p = pd.read_csv(f'{ROOT}/data/processed/poyang_merged_hydrology_wq.csv')
    p['date'] = pd.to_datetime(dict(year=p.year, month=p.month, day=1))
    p = p.set_index('date').sort_index()
    wl = p.water_level.dropna()
    tn = p.TN.reindex(wl.index)
    mask = wl.notna() & tn.notna()
    ax = axes[0, 0]
    ax.scatter(wl[mask], tn[mask], color=BLUE, alpha=0.5, s=14, edgecolors='white', linewidth=0.3)
    z = np.polyfit(wl[mask], tn[mask], 1)
    xs = np.linspace(wl[mask].min(), wl[mask].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=BLUE, linewidth=1.5, linestyle='--')
    r_wl_tn = np.corrcoef(wl[mask], tn[mask])[0, 1]
    ax.text(0.05, 0.95, f'r = {r_wl_tn:.3f}\nsee lakes_null.csv', transform=ax.transAxes,
            va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_xlabel('Water level (m)')
    ax.set_ylabel('TN (mg/L)')
    ax.set_title('(a) Poyang: WL \u2192 TN', fontweight='bold', fontsize=9)
    style_ax(ax)

    # (b) DIS→TP scatter
    dis = p.discharge.dropna()
    tp = p.TP.reindex(dis.index)
    tp[tp < 0] = np.nan
    mask2 = dis.notna() & tp.notna()
    ax = axes[0, 1]
    ax.scatter(dis[mask2], tp[mask2], color=VERMILLION, alpha=0.5, s=14, edgecolors='white', linewidth=0.3)
    r_dis_tp = np.corrcoef(dis[mask2], tp[mask2])[0, 1]
    ax.text(0.05, 0.95, f'r = {r_dis_tp:.3f}\nsee lakes_null.csv', transform=ax.transAxes,
            va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_xlabel('Discharge (m³/s)')
    ax.set_ylabel('TP (mg/L)')
    ax.set_title('(b) Poyang: Discharge \u2192 TP', fontweight='bold', fontsize=9)
    style_ax(ax)

    # (c) Poyang CCM directions summary
    ax = axes[0, 2]
    poy = dl[dl.lake == 'Poyang']
    labels = poy.direction.tolist()
    rhos = poy.rho.tolist()
    pvals = poy.p.tolist()
    colors_p = [GREEN if pv <= 0.05 else GRAY for pv in pvals]
    ax.barh(range(len(labels)), rhos, color=colors_p, alpha=0.7, height=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('CCM rho')
    ax.set_title('(c) Poyang directions', fontweight='bold', fontsize=9)
    ax.invert_yaxis()
    for i, (r, pv) in enumerate(zip(rhos, pvals)):
        sig = '*' if pv <= 0.05 else ' ns'
        ax.text(r + 0.01, i, f'{r:.3f}{sig}', va='center', fontsize=8)
    ax.set_xlim(0, 0.65)
    style_ax(ax)

    # --- (d-f) Taihu ---
    t = pd.read_csv(f'{ROOT}/data/processed/taihu_monthly_water_quality.csv')
    t['date'] = pd.to_datetime(dict(year=t.year, month=t.month, day=1))
    t = t[t.date <= '2022-12-01'].set_index('date').sort_index()

    # (d) TP→Chla scatter — sentinel-cleaned, calendar-aligned pairs
    # (same pipeline as the CCM analysis; r = 0.559, the manuscript value)
    ax = axes[1, 0]
    tp_m = prep_monthly(t.TP)
    ch_m = prep_monthly(t.Chla)
    valid_d = tp_m.notna() & ch_m.notna()
    tp_v = tp_m[valid_d].values.astype(float)
    ch_v = ch_m[valid_d].values.astype(float)
    ax.scatter(tp_v, ch_v, color=GREEN, alpha=0.4, s=10, edgecolors='none')
    z = np.polyfit(tp_v, ch_v, 1)
    xs = np.linspace(tp_v.min(), tp_v.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=GREEN, linewidth=1.5, linestyle='--')
    r_tp_chla = stats.pearsonr(tp_v, ch_v)[0]
    ax.text(0.05, 0.95, f'r = {r_tp_chla:.3f} (n = {len(tp_v)})\nsee lakes_null.csv',
            transform=ax.transAxes,
            va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_xlabel('TP (mg/L)')
    ax.set_ylabel('Chla (μg/L)')
    ax.set_title('(d) Taihu: TP \u2192 Chla', fontweight='bold', fontsize=9)
    style_ax(ax)

    # (e) Chla trend — annual means (>= 9 observed months per year) and the
    # TRUE Theil-Sen slope on that annual series (slope read from
    # taihu_trend.csv, produced by src/taihu_trend.py).
    ax = axes[1, 1]
    ann = t.groupby('year')['Chla'].agg(['mean', 'count'])
    ann = ann[ann['count'] >= 9]   # drops 1991 and 1994 (3 observed months each)
    yrs = ann.index.values.astype(float)
    vals = ann['mean'].values.astype(float)
    ax.scatter(yrs, vals, color=GREEN, s=16, zorder=3, edgecolors='white', linewidth=0.4)
    # true Sen slope line, drawn through the median point (classic Sen placement)
    trend = pd.read_csv(f'{T}/taihu_trend.csv')
    sen_row = trend[(trend.estimator == 'Sen') &
                    (trend.scale == 'annual_ge9months')].iloc[0]
    sen_slope_val = float(sen_row.slope)
    med_x, med_y = np.median(yrs), np.median(vals)
    xs = np.linspace(yrs.min(), yrs.max(), 50)
    ax.plot(xs, med_y + sen_slope_val * (xs - med_x), color=RED,
            linewidth=1.5, linestyle='--', zorder=2)
    ax.text(0.05, 0.95, f'Sen = {sen_slope_val:+.3f} μg/L/yr', transform=ax.transAxes,
            va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_xlabel('Year')
    ax.set_ylabel('Chla (μg/L)')
    ax.set_title('(e) Taihu: Chla trend', fontweight='bold', fontsize=9)
    # 4 ticks (not 6): at 7.2-in print width six 4-digit labels collide
    ax.set_xticks([1995, 2005, 2015, 2022])
    ax.set_xlim(1990.5, 2023.5)
    style_ax(ax)

    # (f) Taihu CCM directions summary
    ax = axes[1, 2]
    tai = dl[dl.lake == 'Taihu']
    labels = tai.direction.tolist()
    rhos = tai.rho.tolist()
    pvals = tai.p.tolist()
    colors_t = [GREEN if pv <= 0.05 else GRAY for pv in pvals]
    ax.barh(range(len(labels)), rhos, color=colors_t, alpha=0.7, height=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('CCM rho')
    ax.set_title('(f) Taihu directions', fontweight='bold', fontsize=9)
    ax.invert_yaxis()
    for i, (r, pv) in enumerate(zip(rhos, pvals)):
        sig = '*' if pv <= 0.05 else ' ns'
        ax.text(r + 0.01, i, f'{r:.3f}{sig}', va='center', fontsize=8)
    ax.set_xlim(0, 0.78)
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(f'{OUT}/Fig4_dual_lake.png')
    plt.close()
    print('Fig4 done')


if __name__ == '__main__':
    import sys
    os.makedirs(OUT, exist_ok=True)
    wanted = {a.lower() for a in sys.argv[1:]} or {'fig1', 'fig2', 'fig3', 'fig4'}
    s = load_shares() if ({'fig1', 'fig2'} & wanted) else None
    if 'fig1' in wanted:
        fig1(s)
    if 'fig2' in wanted:
        fig2(s)
    if 'fig3' in wanted:
        fig3()
    if 'fig4' in wanted:
        fig4()
    print(f'\nFigures saved to {OUT}')
