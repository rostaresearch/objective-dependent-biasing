"""Independent reactive jump-chain sampling check of the last-crossing law."""
import json
import numpy as np
from exact_last_crossing import cases, generator_grid, harmonic, law, boltzmann, tilt, OUT


def sample(K, pi, A, B, q0, n_paths=10000, seed=40930):
    q=harmonic(K,A,B)
    dense=K.toarray(); np.fill_diagonal(dense,0)
    launch=pi[A,None]*dense[A]*q[None,:]
    ai,j=np.nonzero(launch)
    cdf0=np.cumsum(launch[ai,j]); cdf0/=cdf0[-1]
    destinations=[]; cdfs=[]
    for i in range(len(q)):
        dest=np.flatnonzero((dense[i]>0)&(q>0))
        weights=dense[i,dest]*q[dest]
        destinations.append(dest)
        cdf=np.cumsum(weights)
        cdfs.append(cdf/cdf[-1] if len(cdf) else cdf)
    cut=q0<.5
    ref=law(K,pi,A,B,q0)
    lookup={(int(i),int(j)):k for k,(i,j) in enumerate(zip(ref['i'],ref['j']))}
    counts=np.zeros(len(lookup),int)
    target=set(B); rng=np.random.default_rng(seed)
    jumps=[]
    for _ in range(n_paths):
        edge=int(np.searchsorted(cdf0,rng.random()))
        current=int(j[edge]); source=int(A[ai[edge]])
        last=lookup.get((source,current)) if cut[source] and not cut[current] else None
        steps=0
        while current not in target:
            k=int(np.searchsorted(cdfs[current],rng.random()))
            nxt=int(destinations[current][k])
            if cut[current] and not cut[nxt]:
                last=lookup[(current,nxt)]
            current=nxt; steps+=1
            if steps>1000000:
                raise RuntimeError('Unfinished validation trajectory')
        if last is None:
            raise RuntimeError('No last crossing recorded')
        counts[last]+=1; jumps.append(steps)
    empirical=counts/n_paths
    # Every bin should agree within six marginal binomial standard deviations,
    # with 1/n continuity allowance for very low-probability bins.
    sigma=np.sqrt(ref['probability']*(1-ref['probability'])/n_paths)
    max_scaled=float(np.max(abs(empirical-ref['probability'])/(sigma+1/n_paths)))
    assert max_scaled < 6
    return dict(n_paths=n_paths, seed=seed, probability_exact=ref['probability'].tolist(),
                probability_sampled=empirical.tolist(), histogram_TV=float(abs(empirical-ref['probability']).sum()/2),
                maximum_standardized_bin_error=max_scaled, mean_jumps=float(np.mean(jumps)))


if __name__=='__main__':
    name,F,xy,A,B,biases,_=next(cases()); K0=generator_grid(F); q0=harmonic(K0,A,B)
    results={}
    for index,(name,bias) in enumerate([('reference',np.zeros(F.size)),('MFPT_U6',biases['MFPT_U6']),('spectral_U6',biases['spectral_U6'])]):
        results[name]=sample(tilt(K0,bias),boltzmann(F.ravel()+bias),A,B,q0,seed=40930+index)
        print(name,'TV',results[name]['histogram_TV'],'max z',results[name]['maximum_standardized_bin_error'],flush=True)
    (OUT/'jump_chain_validation.json').write_text(json.dumps(results,indent=2)+'\n')
