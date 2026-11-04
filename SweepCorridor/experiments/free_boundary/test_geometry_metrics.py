#!/usr/bin/env python3
"""Analytic cylinder and reparameterization checks for offline metrics."""
import os,tempfile
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
from geometry_metrics import bending

with tempfile.TemporaryDirectory() as directory:
    for index,x in enumerate([np.linspace(-2,2,4),[-2,-4/3,-2/3,2]]):
        cp=np.zeros((4,12));cp[:,0]=3;cp[:,2]=1;cp[:,3]=x;cp[:,7]=2;cp[:,11]=2
        path=Path(directory)/f'cylinder_{index}.csv'
        np.savetxt(path,cp,delimiter=',',header='degree,u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21',comments='')
        q=bending(path)
        assert abs(q['surface']-4*np.pi)<1e-10,q
        assert abs(q['surface_area']-16*np.pi)<1e-10,q
        assert q['longitudinal']<1e-20,q
print('PASS: analytic cylinder curvature and nonlinear reparameterization')
