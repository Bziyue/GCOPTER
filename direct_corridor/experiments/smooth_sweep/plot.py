#!/usr/bin/env python3
"""Render saved Bezier geometry; plots use the exact same input snapshot."""
import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--case', type=Path, required=True)
p.add_argument('--request', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
font_family=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({'font.family':font_family,'axes.unicode_minus':False,
                     'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
seed = np.loadtxt(a.request/'seed.xyz')
direction = seed[-1,:2]-seed[0,:2]
direction /= np.linalg.norm(direction)
rotation = np.array([[*direction,0],[-direction[1],direction[0],0],[0,0,1.]])
origin = np.array([*seed[0,:2],0])
transform = lambda points: (points-origin)@rotation.T
cloud = np.loadtxt(a.request.parent/'map_1.xyz')
cloud = transform(cloud)
seed = transform(seed)
theta = np.arange(40)*2*np.pi/40
circle = np.array([np.cos(theta),np.sin(theta)]).T

def geometry(name, low=0., high=1.):
    rows = np.loadtxt(a.case/f'{name}_controls.csv',delimiter=',',skiprows=1).reshape(-1,6,11)
    rings, centers = [], []
    for piece in rows:
        u0,u1 = piece[0,:2]
        for t in np.linspace(0,1,11,endpoint=False):
            u = u0+(u1-u0)*t
            if not low <= u <= high:
                continue
            basis = np.array([math.comb(5,j)*t**j*(1-t)**(5-j) for j in range(6)])
            v = basis@piece[:,2:]
            center = v[:3]
            shape = v[3:].reshape(2,3).T
            rings.append(transform(center+circle@shape.T))
            centers.append(transform(center))
    rings, centers = np.array(rings), np.array(centers)
    faces = np.stack([rings[:-1],np.roll(rings[:-1],-1,axis=1),
                      np.roll(rings[1:],-1,axis=1),rings[1:]],axis=2).reshape(-1,4,3)
    return rings,centers,faces

fig,axes = plt.subplots(2,1,figsize=(16,6),layout='constrained')
for ax,name,title in zip(axes,['baseline','smooth'],['修改前 · boundary-12','修改后 · smooth']):
    rings,centers,faces = geometry(name)
    footprints = np.unique(np.round(cloud[:,:2]/.25),axis=0)*.25
    ax.scatter(footprints[:,0],footprints[:,1],s=7,c='#8498ad',alpha=.13,marker='s',linewidth=0)
    ax.add_collection(PolyCollection(faces[:,:,:2],facecolor='#ef7d83',edgecolor='none',alpha=.22))
    ax.plot(seed[:,0],seed[:,1],color='#d8a126',lw=1,label='种子路径')
    ax.plot(centers[:,0],centers[:,1],color='#a7263a',lw=1.5,label='走廊中心')
    ax.set(xlim=(-3,seed[-1,0]+3),ylim=(-6,6),aspect='equal',title=title,ylabel='横向 / m')
    ax.grid(alpha=.15)
axes[-1].set_xlabel('沿起终点方向的投影 / m')
axes[0].legend(loc='upper right',ncol=2)
fig.suptitle('同一地图、同一种子路径：方向与共享导数平滑（俯视投影）',fontsize=16)
fig.savefig(a.output/'smooth_sweep_overview.png',dpi=150)
plt.close(fig)

fig = plt.figure(figsize=(15,5.8),layout='constrained')
data = [geometry(name,.46,.585) for name in ['baseline','smooth']]
all_points = np.concatenate([d[0].reshape(-1,3) for d in data])
lo,hi = all_points.min(0),all_points.max(0)
arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(seed,axis=0),axis=1))]
seed_detail=np.array([np.interp(np.linspace(.46,.585,100)*arc[-1],arc,seed[:,j]) for j in range(3)]).T
for i,((rings,centers,faces),title) in enumerate(zip(data,['修改前：截面折回，内表面堆叠','修改后：连续过渡，无局部折回'])):
    ax = fig.add_subplot(1,2,i+1,projection='3d')
    ax.add_collection3d(Poly3DCollection(faces,facecolors='#eb596b',edgecolors='none',alpha=.22,zsort='average'))
    for ring in rings[::22]:
        closed = np.vstack([ring,ring[0]])
        ax.plot(*closed.T,color='#a1283e',lw=.55,alpha=.8)
    ax.plot(*centers.T,color='#801e31',lw=2)
    ax.plot(*seed_detail.T,color='#d8a126',lw=1)
    ax.set(xlim=(lo[0]-.3,hi[0]+.3),ylim=(lo[1]-.3,hi[1]+.3),zlim=(0,5),
           xlabel='纵向 / m',ylabel='横向 / m',zlabel='高度 / m',title=title)
    ax.set_box_aspect((hi[0]-lo[0]+.6,hi[1]-lo[1]+.6,5))
    ax.view_init(elev=24,azim=-65)
    ax.grid(alpha=.12)
fig.suptitle('截图折叠区域的真实曲面；相同视角与比例',fontsize=16)
fig.savefig(a.output/'smooth_sweep_detail.png',dpi=180)
