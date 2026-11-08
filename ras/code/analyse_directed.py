"""Directed support and DHAM reconstruction audit. No automatic repair.

The primary sample is truncated at exact first passage. Complete recorded
propagations are a separately labelled support diagnostic, never a substitute
for the first-passage paths. A/B are physical frame definitions, not centres
of bins that straddle the boundaries.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from extract_periodic import gaussian, sha

OUT=Path(__file__).resolve().parents[1]
KT=0.001987204259*300

def support(C, A, B):
    visited=(C.sum(0)+C.sum(1))>0
    n, lab=connected_components(csr_matrix(C>0), directed=True, connection='strong')
    groups=[]
    for g in range(n):
        ix=np.flatnonzero((lab==g)&visited)
        if not len(ix): continue
        outside=np.flatnonzero(lab!=g)
        closed=not C[np.ix_(ix,outside)].any()
        groups.append(dict(states=len(ix),A=int(A[ix].sum()),B=int(B[ix].sum()),closed=closed))
    return dict(visited_states=int(visited.sum()),A_states=int((A&visited).sum()),
        B_states=int((B&visited).sum()),strong_components=groups,
        closed_components_containing_A_and_B=sum(g['closed'] and g['A']>0 and g['B']>0 for g in groups),
        asymmetric_count_max=float(abs(C-C.T).max()))

def counts_model(records,es,ef,lag,pre):
    ns,nf=len(es)-1,len(ef)-1; N=ns*nf
    C=np.zeros((N,N)); D=np.zeros((N,N)); V=np.zeros(N); occ=np.zeros(N)
    centers=np.repeat((es[:-1]+es[1:])/2,nf)
    n_pairs=0; sum_bias_error=0
    for z in records:
        f=z['frames']; keep=f[:,0]<=z['first_passage_step'] if pre else np.ones(len(f),bool)
        for k,p in enumerate(z['parameters']):
            x=f[(f[:,1]==k)&keep]
            if len(x)<=lag: continue
            si=np.clip(np.searchsorted(es,x[:,2],side='right')-1,0,ns-1)
            fi=np.clip(np.searchsorted(ef,x[:,3],side='right')-1,0,nf-1)
            st=si*nf+fi; c=np.zeros_like(C)
            np.add.at(c,(st[:-lag],st[lag:]),1)
            v=gaussian(centers,p)
            D+=c.sum(1)[:,None]*np.exp(-(v[None,:]-v[:,None])/(2*KT))
            C+=c; n_pairs+=len(x)-lag
            np.add.at(V,st,x[:,4]); np.add.at(occ,st,1)
    M=np.divide(C,D,out=np.zeros_like(C),where=D>0)
    P=np.divide(M,M.sum(1)[:,None],out=np.zeros_like(M),where=M.sum(1)[:,None]>0)
    A=centers<5; B=centers>=8
    r=support(C,A,B)
    r.update(n_pairs=n_pairs,lag_ps=lag,pre_first_passage=pre,
             row_sum_error=float(abs(P.sum(1)[C.sum(1)>0]-1).max()),
             no_symmetrization=True,no_pseudocounts=True)
    return r,C,P,A,B

def regression():
    C=np.array([[8.,2.,0.],[0.,8.,2.],[0.,0.,10.]])
    s=support(C,np.array([1,0,0],bool),np.array([0,0,1],bool))
    assert s['closed_components_containing_A_and_B']==0
    assert s['asymmetric_count_max']==2
    p=np.array([2.,3.,1.]); assert np.isclose(gaussian(3,p),2)
    # Source-first orientation: positive uphill bias suppresses forward counts.
    P0=np.array([[.8,.2],[.3,.7]]); v=np.array([0.,.4]); n=np.array([100.,120.])
    tilted=P0*np.exp(-(v[None,:]-v[:,None])/(2*KT))
    C=tilted*n[:,None]
    estimate=C/(n[:,None]*np.exp(-(v[None,:]-v[:,None])/(2*KT)))
    assert np.allclose(estimate,P0)
    assert np.allclose(P0.sum(1),1)

if __name__=='__main__':
    regression()
    files=sorted((OUT/'features').glob('*.npz')); assert len(files)==73
    records=[dict(np.load(p,allow_pickle=False)) for p in files]
    frames=np.vstack([z['frames'] for z in records])
    es=np.unique(np.r_[np.linspace(frames[:,2].min()-1e-6,frames[:,2].max()+1e-6,13),5.,8.])
    ef=np.linspace(frames[:,3].min()-1e-6,frames[:,3].max()+1e-6,9)
    results=[]
    for pre in (True,False):
        for lag in (1,2,4):
            r,C,P,A,B=counts_model(records,es,ef,lag,pre)
            np.savez_compressed(OUT/f'directed_pre{int(pre)}_lag{lag}.npz',C=C,P=P,A=A,B=B,s_edges=es,feature_edges=ef)
            results.append(r); print(json.dumps(r),flush=True)
    (OUT/'directed_support_audit.json').write_text(json.dumps(dict(
        grid='12 uniform s bins augmented with exact 5 and 8 A boundaries; 8 Pi-Q61 bins',
        script_sha256=sha(__file__),temperature_K=300,results=results,
        note='Stationary reactive current requires recurrent A/B support. No overlap is supplied when this gate fails.'),indent=2))
