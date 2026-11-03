#!/usr/bin/env python3
"""Plot the measured tradeoff; no generation or optimization is run here."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    names = {1: 'RViz 1', 2: 'Perlin 10', 3: 'Maze 10',
             4: 'RViz 2', 5: 'Perlin 5', 6: 'Maze 4'}
    colors = {1: '#64748b', 4: '#64748b', 2: '#2563eb', 5: '#2563eb',
              3: '#059669', 6: '#059669'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax in axes:
        ax.axhline(1, color='#334155', lw=1, ls='--')
        ax.axvline(1, color='#334155', lw=1, ls='--')
        ax.grid(alpha=.18)
        ax.set_xlabel('Generation time / ellipse reference')
        ax.set_ylabel('Union volume / ellipse reference')
        ax.spines[['top', 'right']].set_visible(False)
    balanced = [r for r in data['final'] if r['variant'] == 'm4_smooth_sparse']
    dense = {r['case']: r for r in data['final'] if r['variant'] == 'm4_smooth'}
    for row in balanced:
        i = row['case']
        x = row['minkowski_median_ms']/row['ellipse_median_ms']
        y = row['minkowski_union_m3']/row['ellipse_union_m3']
        d = dense[i]
        dx = d['minkowski_median_ms']/d['ellipse_median_ms']
        dy = d['minkowski_union_m3']/d['ellipse_union_m3']
        axes[0].plot([x, dx], [y, dy], color=colors[i], alpha=.35, lw=1)
        axes[0].scatter(x, y, s=65, color=colors[i], zorder=3)
        axes[0].scatter(dx, dy, s=45, marker='^', facecolors='none', edgecolors=colors[i], zorder=3)
        offsets = {1: (-32, -14), 2: (5, -15), 4: (7, 7)}
        axes[0].annotate(names[i], (x, y), xytext=offsets.get(i, (4, 7)),
                         textcoords='offset points', fontsize=9)
    axes[0].set_xlim(.72, 1.25)
    axes[0].set_ylim(.74, 1.18)
    axes[0].fill_between([.72, 1], 1, 1.18, color='#dcfce7', alpha=.6)
    axes[0].set_title('Reference: ellipse at 0.4 m spacing')
    axes[0].legend(handles=[
        Line2D([], [], marker='o', ls='', color='#334155', label='Minkowski, 0.6 m'),
        Line2D([], [], marker='^', ls='', markerfacecolor='none', color='#334155', label='Minkowski, 0.4 m'),
    ], loc='lower right', frameon=False, fontsize=9)
    for row in data['matched_density']:
        i = row['case']
        x = row['minkowski_median_ms']/row['ellipse_median_ms']
        y = row['minkowski_union_m3']/row['ellipse_union_m3']
        axes[1].scatter(x, y, s=65, color=colors[i], zorder=3)
        axes[1].annotate(names[i], (x, y), xytext=(5, 6 if i != 1 else -13),
                         textcoords='offset points', fontsize=9)
    axes[1].set_xlim(.98, 1.23)
    axes[1].set_ylim(.80, 1.20)
    axes[1].set_title('Both methods at 0.6 m spacing')
    fig.suptitle('Minkowski corridors: scene-dependent volume / time tradeoffs', fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
