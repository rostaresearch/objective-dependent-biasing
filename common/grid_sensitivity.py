"""Refine the nine-well reference with fixed physical coarse-cell basin sets.

Re-evaluate the published analytic F, interpolate the deposited per-state bias
linearly (no reoptimization), and retain the same physical cells for A and B.
This is a discretization sensitivity test, not a new optimized protocol.
"""
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.special import expit
from exact_last_crossing import DATA, OUT, generator_grid, harmonic, boltzmann, law, tilt, mfpt, overlap_current


def potential(xs):
    X,Y=np.meshgrid(xs,xs)
    amp=6.
    amplitudes=np.array([.9,.3,.5,1.,.2,.4,.9,.9,.9])*amp
    x0=[1.12,1.,3.,4.15,4.,5.27,5.5,6.,1.]
    y0=[1.34,2.25,2.31,3.62,5.,4.14,4.5,1.52,5.]
    sx=[.5,.3,.4,2.,.9,1.,.3,.5,.5]
    sy=[.5,.3,1.,.8,.2,.3,1.,.6,.7]
    Z=np.zeros_like(X)+amp*4.184
    for a,x,y,wx,wy in zip(amplitudes,x0,y0,sx,sy):
        Z-=a*np.exp(-(X-x)**2/(2*wx**2)-(Y-y)**2/(2*wy**2))
    Z+=.3*amp*np.exp(-(X-np.pi)**2/(2*3.**2)-(Y-np.pi)**2/(2*.3**2))
    Z+=100*(expit(-5*(X+.7))+expit(5*(X-(2*np.pi+.7)))+
            expit(-5*(Y+.7))+expit(5*(Y-(2*np.pi+.7))))
    return (Z-Z.min())/.5981


if __name__=='__main__':
    d=loadmat(DATA/'rkhs_2d_data.mat'); xs=d['xs'].ravel()
    err=float(np.max(abs(potential(xs)-d['F'])))
    assert err < 1e-10, err
    coarse_coords=np.c_[d['X'].ravel(),d['Y'].ravel()]
    centres=[coarse_coords[np.argmin(np.linalg.norm(coarse_coords-d[k].ravel(),axis=1))] for k in ['start_xy','end_xy']]
    half=(xs[1]-xs[0])/2
    output=dict(potential_reproduction_max_error=err, basin_cell_centres=[v.tolist() for v in centres],
                basin_halfwidth=half, bias_interpolation='bilinear; no reoptimization', results=[])
    for n in [20,40,80,120]:
        x=np.linspace(0,2*np.pi,n); X,Y=np.meshgrid(x,x)
        coords=np.c_[X.ravel(),Y.ravel()]
        A,B=[np.flatnonzero(np.max(abs(coords-c),axis=1)<=half+1e-12) for c in centres]
        F=potential(x); K=generator_grid(F); pi=boltzmann(F.ravel()); q0=harmonic(K,A,B)
        ref=law(K,pi,A,B,q0); t0=mfpt(K,A,B)
        for name,key in [('MFPT_U6','u_mfpt_2'),('spectral_U6','u_spec_2')]:
            b=RegularGridInterpolator((xs,xs),d[key])(coords[:,::-1])
            kb=tilt(K,b); lb=law(kb,boltzmann(F.ravel()+b),A,B,q0)
            row=dict(n=n,protocol=name,n_A=len(A),n_B=len(B),
                     C_TS=float(np.minimum(ref['probability'],lb['probability']).sum()),
                     Omega_J=overlap_current(ref['current'],lb['current']),
                     speedup=t0/mfpt(kb,A,B), flux_residual=lb['capacity_relative_residual'])
            output['results'].append(row); print(row,flush=True)
    (OUT/'grid_sensitivity.json').write_text(json.dumps(output,indent=2)+'\n')
