"""Auditable 73-run feature extraction; no trajectory or source-file edits.

Keeps complete saved propagations AND an exact first-passage mask. No
interpolation, invented endpoint, count symmetrization or connectivity repair.
"""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path('/mnt/data7/KOMBI_Ras_trajectories/periodic_completed_20261023')
OUT = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def gaussian(x, p):
    a, b, c = np.asarray(p).reshape(3, -1)
    assert np.all(c > 0)
    return np.sum(a*np.exp(-0.5*((np.asarray(x)[...,None]-b)/c)**2), axis=-1)

def extract(row):
    import mdtraj as md
    run = Path(row['consolidated_path'])
    ident = str(row['analysis_id'])
    cfg = json.loads((run/'config_used.json').read_text())
    fp = json.loads((run/'first_passage.json').read_text())
    assert cfg['pbc'] and cfg['propagation_step'] == 10000
    assert cfg['dham_lag_steps'] == 1 and cfg['amp_bounds'] == [0.1,5.0]
    assert cfg['seed'] == row['seed'] and cfg['target_distance'] == 8
    top = md.load_topology(str(run/'solute.pdb'))
    mg = top.select('name MG'); ph = top.select('resname H2P and element P')
    q = [a.index for a in top.atoms if a.residue.chain.index == 0
         and a.residue.resSeq == 61 and a.name == 'NE2']
    assert len(mg)==len(ph)==len(q)==1
    frames=[]; pars=[]; sources=[]; found=None; max_cv=0.; max_v=0.
    for k in range(fp['n_props']):
        cp=run/'cv'/f'prop_{k:04d}.npz'
        tp=run/'traj'/f'solute_prop_{k:04d}.dcd'
        z=np.load(cp, allow_pickle=False)
        assert int(z['first_step'])==k*10000 and float(z['dt_ps'])==0.002
        s=z['s_A']; p=z['params']; v=z['V_openmm_kcal']
        assert len(s)==10000 and np.isfinite(s).all()
        err=abs(gaussian(s,p)-v)
        assert np.all(err<=0.001+0.0001*abs(v))
        max_v=max(max_v,float(err.max()))
        hit=np.flatnonzero(s>=8)
        if found is None and len(hit): found=k*10000+int(hit[0])+1
        tr=md.load_dcd(str(tp), top=top)
        assert len(tr)==20 and tr.unitcell_vectors is not None
        # Explicit periodic minimum-image distances (mdtraj nm -> Angstrom).
        dist=md.compute_distances(tr, [[ph[0],mg[0]],[ph[0],q[0]]],periodic=True)*10
        idx=np.arange(499,10000,500); steps=k*10000+idx+1
        max_cv=max(max_cv,float(abs(dist[:,0]-s[idx]).max()))
        assert max_cv<0.003
        frames.append(np.column_stack((steps, np.full(20,k), s[idx],dist[:,1],v[idx])))
        pars.append(p)
        sources.extend([{'path':str(cp),'sha256':sha(cp)}, {'path':str(tp),'sha256':sha(tp)}])
    assert found==fp['total_steps'] and abs(found*0.002-row['fpt_ps'])<1e-8
    x=np.vstack(frames); pre=x[:,0]<=found
    np.savez_compressed(OUT/'features'/f'{ident}.npz',frames=x,parameters=np.asarray(pars),
                        first_passage_step=found,source_id=ident)
    result=dict(id=ident,source=row,first_passage_ps=found*0.002,n_propagations=len(pars),
        saved_frames=len(x),first_passage_frames=int(pre.sum()),
        frames_at_or_beyond_8_before_first_passage=int(((x[:,2]>=8)&pre).sum()),
        frames_at_or_beyond_8_complete_propagations=int((x[:,2]>=8).sum()),
        last_prearrival_s=float(x[pre][-1,2]),max_cv_error_A=max_cv,max_bias_error_kcal=max_v,
        sources=sources,topology_sha256=sha(run/'solute.pdb'),config_sha256=sha(run/'config_used.json'))
    (OUT/'provenance'/f'{ident}.json').write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k not in ('sources','source')}

if __name__=='__main__':
    for name in ('features','provenance'): (OUT/name).mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    rows=[r for r in manifest['rows'] if r['update_ps']==20]
    assert len(rows)==73 and len({r['seed'] for r in rows})==73
    results=[]
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(extract, rows):
            results.append(r)
            print(f"extracted {len(results)}/73 {r['id']}",flush=True)
    times=np.array([r['first_passage_ps'] for r in results])
    output=dict(manifest_sha256=sha(ROOT/'manifest.json'),script_sha256=sha(__file__),
        n_runs=73,mean_ps=float(times.mean()),median_ps=float(np.median(times)),
        sem_ps=float(times.std(ddof=1)/np.sqrt(len(times))),runs=results)
    (OUT/'extraction_summary.json').write_text(json.dumps(output,indent=2))
