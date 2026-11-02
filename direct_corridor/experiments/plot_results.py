#!/usr/bin/env python3
"""Render the measured halfwidth counterexample and fixed-section area results."""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial import polynomial as pol
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spline', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    colors = ['#2878b5', '#e08b2c', '#258c71', '#8b5ca6']
    for degree, label, color in [(3, 'Cubic minimum acceleration', colors[0]),
                                  (5, 'Quintic minimum jerk', colors[1])]:
        data = json.loads((args.spline/f'throat_degree{degree}.json').read_text())
        knots, values = np.array(data['knots']), np.array(data['values'])
        coeff = np.array(data['coefficients']).reshape(-1, degree+1, 3)
        for i, c in enumerate(coeff):
            t = np.linspace(0, knots[i+1]-knots[i], 120)
            axes[0].plot(knots[i]+t, pol.polyval(t, c[:, 1]), color=color,
                         label=label if i == 0 else None)
    for i in range(len(knots)-1):
        u = np.linspace(0, 1, 120)
        w = 10*u**3-15*u**4+6*u**5
        axes[0].plot(knots[i]+u, (1-w)*values[i, 1]+w*values[i+1, 1],
                     color=colors[2], ls='--', label='Quintic, zero node jets' if i == 0 else None)
    axes[0].scatter(knots, values[:, 1], color='#263449', s=25, zorder=4)
    axes[0].axhspan(-.23, 0, color='#fcdfdf', alpha=.6)
    axes[0].axhline(0, color='#b33a3a', lw=.8)
    axes[0].set(xlabel='Longitudinal parameter', ylabel='Section halfwidth',
                title='C2 continuity does not preserve positive halfwidth', ylim=(-.23, 1.02))
    axes[0].legend(fontsize=8, loc='upper center')
    cases = ['square', 'octagon', 'cut_corner']
    families = ['ellipse', 'superellipse4', 'minkowski2', 'minkowski4']
    labels = ['Ellipse', 'p=4 superellipse', '2-template sum', '4-template sum']
    rows = summary['geometry']['sections']['rows']
    x = np.arange(len(cases))
    for i, (family, label) in enumerate(zip(families, labels)):
        fractions = [100*next(r['fraction'] for r in rows if r['case']==case and r['family']==family)
                     for case in cases]
        bars = axes[1].bar(x+(i-1.5)*.19, fractions, .18, color=colors[i], label=label)
        axes[1].bar_label(bars, fmt='%.1f', fontsize=7, padding=2)
    axes[1].set(xticks=x, xticklabels=['Square', 'Octagon', 'Cut corner'],
                ylabel='Area / available polygon (%)', ylim=(0, 116),
                title='Same halfspaces and six orientation candidates')
    axes[1].legend(fontsize=8, loc='lower left', ncol=2)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Parameterization probes: fixed-section experiments', fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
