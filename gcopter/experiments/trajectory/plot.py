#!/usr/bin/env python3
"""Plot measured time and violation; failed outputs remain visibly marked."""
import json
from pathlib import Path
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

source=Path(sys.argv[1]);target=Path(sys.argv[2])
data=json.loads(source.read_text())
names=['Perlin A1','Perlin B1','Maze 1','Perlin A2','Perlin B2','Maze 2']
fig,axes=plt.subplots(2,1,figsize=(10,6.4),sharex=True,layout='constrained')
for method,color,offset,label in [('firi','#3b73b9',-.19,'FIRI union'),('hermite','#d57929',.19,'Hermite tube')]:
    times=[];violations=[];failed=[];low=[];high=[]
    for case in data['cases']:
        runs=[r for r in case['runs'] if r['method']==method and r['repeat']>=0]
        t=[r['optimize_ms']/1000 for r in runs];median=statistics.median(t)
        times.append(median);low.append(median-min(t));high.append(max(t)-median)
        violations.append(1000*runs[0]['max_corridor_error_m']);failed.append(not runs[0]['sampled_feasible'])
    x=[i+offset for i in range(len(names))]
    axes[0].bar(x,times,.36,color=color,label=label,yerr=[low,high],capsize=3)
    axes[1].bar(x,violations,.36,color=color)
    for axis,values in zip(axes,[times,violations]):
        for x0,y,bad in zip(x,values,failed):
            if bad:axis.plot(x0,y*1.12,'x',color='#b62222',markersize=7,markeredgewidth=1.7)
axes[0].set_ylabel('Optimization time (s, log scale)')
axes[0].set_yscale('log');axes[0].set_ylim(.3,40)
axes[0].set_title('Same map / path / initial trajectory — common backend\n128 integration intervals, 3200-iteration cap; 3 measured repeats',fontsize=12)
handles,labels=axes[0].get_legend_handles_labels()
axes[0].legend(handles+[Line2D([0],[0],marker='x',linestyle='',color='#b62222')],labels+['Fails sampled feasibility'],loc='upper right',fontsize=9)
axes[1].set_ylabel('Max corridor violation (mm, log scale)');axes[1].set_yscale('log')
axes[1].axhline(5,color='#b62222',linestyle='--',linewidth=1,label='5 mm tolerance')
axes[1].legend(loc='upper right',fontsize=9)
axes[1].set_ylim(.8,90);axes[1].set_xticks(range(len(names)),names)
for axis in axes:
    axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    axis.spines[['top','right']].set_visible(False)
fig.supxlabel('Failed/iteration-limited runs are included; this is not a native GCOPTER ranking.',fontsize=10)
target.parent.mkdir(parents=True,exist_ok=True);fig.savefig(target,dpi=160)
