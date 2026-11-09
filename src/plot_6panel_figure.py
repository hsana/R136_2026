#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import orbital_configuration_posteriors_R136 as func


def get_output_config(output):
    """Return (xmin, xmax, scale_choice, xlabel, Ndigit) for a given output type."""
    cfg = {
        'RV':              (-15,    15,    'linear', 'RV (km/s)',                      3),
        'Vtan_mas':        (0.001,  0.1,   'log',    'Tangential velocity (mas/yr)',    3),
        'Vtan_micas':      (0,      125,   'linear', 'Astrometric motion (1e-3mas/yr)',2),
        'Ratio':           (0,      2,     'linear', 'Sep/a',                           3),
        'Proj_Separation': (200,    2000,  'log',    'Proj. Separation (au)',           0),
        'Semi-major_axis': (200,    10000, 'log',    'a (au)',                          0),
        'Period':          (200,    19999, 'log',    'Period (year)',                   0),
    }
    return cfg[output]


def plot_marginalised_panel(ax, output, object, nbins=200, N=900_000, rng=None):
    """
    Plot the fully marginalised posterior for a given output and object onto ax.
    Reproduces panel 4 of the original 4-panel script exactly.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    xmin, xmax, scale_choice, xlabel, Ndigit = get_output_config(output)

    bins = np.linspace(xmin, xmax, nbins + 1)
    cent = 0.5 * (bins[:-1] + bins[1:])

    x    = func.sample_sep_over_a(N, rng=rng, object=object, output=output)
    h, _ = np.histogram(x, bins=bins, density=False)
    pdf  = h / np.trapz(h, x=cent)

    ax.plot(cent, pdf, color='k', lw=2.8)
    ax.set_xscale(scale_choice)

    if scale_choice == 'log':
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=True))
        # Force specific ticks depending on range
        if xmax <= 10000:   # Semi-major_axis: 200–10000
            ax.set_xticks([300, 1000, 3000, 10000])
        else:               # Period: 200–20000
            ax.set_xticks([300, 1000, 3000, 10000, 20000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    x_sorted = np.sort(x)
    median   = np.median(x_sorted)
    hdi_68   = func.hdi_bounds(x_sorted, 0.68)
    hdi_95   = func.hdi_bounds(x_sorted, 0.95)

    mask_95 = (cent >= hdi_95[0]) & (cent <= hdi_95[1])
    mask_68 = (cent >= hdi_68[0]) & (cent <= hdi_68[1])
    ax.fill_between(cent[mask_95], 0, pdf[mask_95], color='blue',   alpha=0.08)
    ax.fill_between(cent[mask_68], 0, pdf[mask_68], color='orange', alpha=0.25)

    def draw_vertical_fill(xval, color, label=None):
        idx = np.argmin(np.abs(cent - xval))
        ax.plot([cent[idx], cent[idx]], [0, pdf[idx]], color=color, linestyle='--', lw=1.5, label=label)

    draw_vertical_fill(median,    'red',    label='Median')
    draw_vertical_fill(hdi_68[0], 'orange', label='68% HDI')
    draw_vertical_fill(hdi_68[1], 'orange')
    draw_vertical_fill(hdi_95[0], 'blue',   label='95% HDI')
    draw_vertical_fill(hdi_95[1], 'blue')

    ax.plot(cent, pdf, color='k', lw=2.8)

    ymax = ax.get_ylim()[1]
    fmt = '.1f' if output == 'Vtan_micas' else '.0f'

    ax.text(median,    ymax*0.05, f'{median:{fmt}}',    color='red',    ha='right', va='bottom', fontsize=10, rotation=90)
    ax.text(hdi_68[0], ymax*0.02, f'{hdi_68[0]:{fmt}}', color='orange', ha='left',  va='bottom', fontsize=10, rotation=90)
    ax.text(hdi_68[1], ymax*0.02, f'{hdi_68[1]:{fmt}}', color='orange', ha='left',  va='bottom', fontsize=10, rotation=90)
    if output == 'Vtan_micas':
        ax.text(hdi_95[0], ymax*0.25, f'{hdi_95[0]:{fmt}}'.strip(), color='blue',   ha='right', va='bottom', fontsize=10, rotation=90)
        ax.text(hdi_95[1], ymax*0.25, f'{hdi_95[1]:{fmt}}'.strip(), color='blue',   ha='left',  va='bottom', fontsize=10, rotation=90)
    if output == 'Period':
        ax.text(hdi_95[0], ymax*0.14, f'{hdi_95[0]:{fmt}}'.strip(), color='blue',   ha='right', va='bottom', fontsize=10, rotation=90)
        ax.text(hdi_95[1], ymax*0.05, f'{hdi_95[1]:{fmt}}'.strip(), color='blue',   ha='left',  va='bottom', fontsize=10, rotation=90)
    if output == 'Semi-major_axis':
        ax.text(hdi_95[0], ymax*0.10, f'{hdi_95[0]:{fmt}}'.strip(), color='blue',   ha='right', va='bottom', fontsize=10, rotation=90)
        ax.text(hdi_95[1], ymax*0.05, f'{hdi_95[1]:{fmt}}'.strip(), color='blue',   ha='left',  va='bottom', fontsize=10, rotation=90)    

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized probability density')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=1, loc='upper right', fontsize=8)

panels = [
    # (output,            label, object)
    ('Semi-major_axis',   'A',   'A1'),
    ('Semi-major_axis',   'B',   'A2'),
    ('Period',            'C',   'A1'),
    ('Period',            'D',   'A2'),
    ('Vtan_micas',        'E',   'A1'),
    ('Vtan_micas',        'F',   'A2'),
]

fig, axes = plt.subplots(3, 2, figsize=(7.5, 7.5))

for ax, (output, label, obj) in zip(axes.flatten(), panels):
    plot_marginalised_panel(ax, output=output, object=obj, rng=np.random.default_rng(42))
    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    if label in ('A', 'B'):
        ax.set_title('a1' if obj == 'A1' else 'a2', fontsize=10)

fig.tight_layout(pad=1.5)
fig.subplots_adjust(left=0.12)
fig.savefig('R136_posteriors_3x2.pdf', dpi=300, bbox_inches='tight')
plt.show()
