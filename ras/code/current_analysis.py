"""Nonreversible TPT on directed DHAM estimates of complete saved blocks.

These are conditional fixed-generator diagnostics, not a reconstruction of
the nonstationary adaptive path law. No graph repair, reversible projection,
or artificial target state is permitted. Bootstrap failures are counted.
"""
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.linalg import solve, LinAlgWarning
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.io import savemat
from extract_periodic import gaussian,sha
from analyse_directed import KT

ROOT=Path(__file__).resolve().parents[1]
warnings.simplefilter('error',LinAlgWarning)

def current(K,A,B):
    n=len(K)
    # A common change of time unit leaves normalized currents/committors fixed.
    K=K/np.max(-np.diag(K))
    assert np.allclose(K.sum(1),0,atol=1e-12)
    # Subtraction-free GTH elimination avoids negative tiny stationary weights
    # from a constrained dense solve on strongly tilted generators.
    g=K.copy();np.fill_diagonal(g,0)
    for k in range(n-1):
        scale=g[k,k+1:].sum()
        if scale<=0: raise ValueError('GTH reducible support')
        g[k+1:,k]/=scale
        g[k+1:,k+1:]+=np.outer(g[k+1:,k],g[k,k+1:])
    pi=np.zeros(n);pi[-1]=1
    for k in range(n-2,-1,-1): pi[k]=pi[k+1:]@g[k+1:,k]
    pi/=pi.sum()
    if pi.min()<=0 or abs(pi@K).max()>1e-12: raise ValueError('stationary solve')
    interior=~(A|B)
    def comm(Q,a,b):
        q=np.zeros(n);q[b]=1
        q[interior]=solve(Q[np.ix_(interior,interior)],-Q[np.ix_(interior,b)].sum(1))
        if q.min()<-1e-9 or q.max()>1+1e-9: raise ValueError('committor range')
        if abs((Q@q)[interior]).max(initial=0)>1e-9: raise ValueError('committor residual')
        return np.clip(q,0,1)
    qp=comm(K,A,B)
    reverse=K.T*pi[None,:]/pi[:,None]
    qm=comm(reverse,B,A)
    F=pi[:,None]*qm[:,None]*K*qp[None,:];np.fill_diagonal(F,0)
    J=np.maximum(F-F.T,0); mass=J.sum()
    if not np.isfinite(mass) or mass<=0: raise ValueError('reactive current zero')
    return J/mass,pi,qp,qm

def generator(P):
    K=P.copy();np.fill_diagonal(K,0);np.fill_diagonal(K,-K.sum(1));return K

def prepare(records,es,ef,lag):
    ns,nf=len(es)-1,len(ef)-1;N=ns*nf
    centers=np.repeat((es[:-1]+es[1:])/2,nf)
    all_arrays=[]
    for z in records:
        C=np.zeros((N,N));D=C.copy();X=C.copy();V=np.zeros(N);occ=V.copy()
        for k,p in enumerate(z['parameters']):
            f=z['frames'][z['frames'][:,1]==k]
            if len(f)<=lag: continue
            s=np.clip(np.searchsorted(es,f[:,2],side='right')-1,0,ns-1)*nf+np.clip(np.searchsorted(ef,f[:,3],side='right')-1,0,nf-1)
            c=np.zeros_like(C);np.add.at(c,(s[:-lag],s[lag:]),1)
            v=gaussian(centers,p)
            C+=c;D+=c.sum(1)[:,None]*np.exp(-(v[None,:]-v[:,None])/(2*KT))
            np.add.at(V,s,f[:,4]);np.add.at(occ,s,1)
            # Actual force bias evaluated at both observed endpoint coordinates.
            dx=(gaussian(f[lag:,2],p)-gaussian(f[:-lag,2],p))/KT
            np.add.at(X,(s[:-lag],s[lag:]),dx)
        all_arrays.append((C,D,V,occ,X))
    return [np.array([a[k] for a in all_arrays]) for k in range(5)],centers

def estimate(arrays,weights,centers):
    C,D,V,occ,X=[np.tensordot(weights,a,axes=1) for a in arrays]
    N=len(C);active=(C.sum(0)+C.sum(1))>0;ix=np.flatnonzero(active)
    c=C[np.ix_(ix,ix)]
    n,_=connected_components(csr_matrix(c>0),directed=True,connection='strong')
    if n!=1: raise ValueError('directed support not strongly connected')
    A=centers[ix]<5;B=centers[ix]>=8
    if not A.any() or not B.any(): raise ValueError('empty physical A/B')
    raw=np.divide(C,D,out=np.zeros_like(C),where=D>0)[np.ix_(ix,ix)]
    P=raw/raw.sum(1)[:,None]
    K=generator(P)
    b=np.divide(V,occ,out=np.zeros_like(V),where=occ>0)[ix]/KT
    J,pi,q,qm=current(K,A,B)
    rate=np.exp(-(b[None,:]-b[:,None])/2)
    Kb=generator(K*rate)
    Jb,_,_,_=current(Kb,A,B)
    X=np.divide(X,C,out=np.zeros_like(X),where=C>0)[np.ix_(ix,ix)]
    # Directed edge-conditioned signed mean is a second fixed-generator model.
    Kr=generator(K*np.exp(-X/2))
    Jr,_,_,_=current(Kr,A,B)
    full=np.zeros((N,N)); full[np.ix_(ix,ix)]=J
    fullb=np.zeros((N,N));fullb[np.ix_(ix,ix)]=Jb
    return dict(omega_state=float(np.minimum(J,Jb).sum()),
        omega_edge_conditioned=float(np.minimum(J,Jr).sum()),
        D_state=float((J*abs(b[None,:]-b[:,None])/2).sum()),
        D_edge_conditioned=float((J*abs(X)/2).sum()),
        A_states=int(A.sum()),B_states=int(B.sum()),states=len(ix),
        max_backward_vs_one_minus_forward=float(abs(qm-(1-q)).max()),
        escape_probability_median=float(np.median(1-np.diag(P)))),full,fullb

def test():
    K=np.array([[-3.,2.,1.],[1.,-3.,2.],[2.,1.,-3.]])
    A=np.array([1,0,0],bool);B=np.array([0,0,1],bool)
    J,pi,q,qm=current(K,A,B)
    assert np.allclose(pi,1/3) and abs(q[1]-2/3)<1e-12 and abs(qm[1]-2/3)<1e-12
    assert abs(J.sum()-1)<1e-12
    assert np.max(abs(qm-(1-q)))>0.3
    Kr=K+K.T;np.fill_diagonal(Kr,0);Kr=generator(Kr)
    _,_,q,qm=current(Kr,A,B);assert np.allclose(qm,1-q)
    assert np.allclose(current(K*1e12,A,B)[0],J)

if __name__=='__main__':
    test(); rng=np.random.default_rng(20261025)
    records=[dict(np.load(p,allow_pickle=False)) for p in sorted((ROOT/'features').glob('*.npz'))]
    grid=np.load(ROOT/'directed_pre0_lag1.npz');es=grid['s_edges'];ef=grid['feature_edges']
    results={};data={}
    for lag in (1,2):
        arrays,centers=prepare(records,es,ef,lag)
        r,J,Jb=estimate(arrays,np.ones(73),centers)
        bootstrap=[];failures={};null=[];null_failed=0
        for it in range(200):
            weights=np.bincount(rng.integers(0,73,73),minlength=73)
            try: bootstrap.append(estimate(arrays,weights,centers)[0])
            except (ValueError,LinAlgWarning,np.linalg.LinAlgError) as e:
                failures[str(e)]=failures.get(str(e),0)+1
        for it in range(200):
            order=rng.permutation(73);wa=np.zeros(73);wb=wa.copy();wa[order[:36]]=1;wb[order[36:]]=1
            try:
                ja=estimate(arrays,wa,centers)[1];jb=estimate(arrays,wb,centers)[1]
                null.append(float(np.minimum(ja,jb).sum()))
            except (ValueError,LinAlgWarning,np.linalg.LinAlgError): null_failed+=1
        r.update(bootstrap_attempts=200,bootstrap_valid=len(bootstrap),bootstrap_failures=failures,
            bootstrap_values=bootstrap,split_half_attempts=200,split_half_valid=len(null),
            split_half_failures=null_failed,split_half_values=null,
            interval_note='Percentiles among estimable resamples only; failed draws remain counted. Not calibrated coverage.')
        if bootstrap:
            r['conditional_percentiles_5_50_95']={key:np.percentile([x[key] for x in bootstrap],[5,50,95]).tolist()
                for key in ('omega_state','omega_edge_conditioned','D_state','D_edge_conditioned')}
        if null:r['split_half_conditional_percentiles_5_50_95']=np.percentile(null,[5,50,95]).tolist()
        results[str(lag)]=r
        data[f'J0_lag{lag}']=J;data[f'Jb_lag{lag}']=Jb
        (ROOT/'current_results.json').write_text(json.dumps(dict(script_sha256=sha(__file__),
            counts='directed',source='complete recorded propagation blocks',results=results),indent=2))
        print(lag,{k:v for k,v in r.items() if k not in ('bootstrap_values','split_half_values')},flush=True)
    savemat(ROOT/'current_data.mat',dict(s_edges=es,feature_edges=ef,**data))
