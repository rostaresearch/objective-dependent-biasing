"""Run a queue of Langevin jobs in order (each job checkpoints and resumes; see langevin.py).

Usage: python campaign.py JOBFILE.json THREADS
JOBFILE: list of {case, interp, dt, nwalk, seed, expected, max_hours}. 'expected' is the network
MFPT (it only sets the number of steps per kernel call).
"""
import json
import sys
import time

import numba

import langevin
from common import HERE

if __name__ == '__main__':
    jobs = json.loads(open(sys.argv[1]).read())
    numba.set_num_threads(int(sys.argv[2]))
    log = open(HERE / 'runs' / (sys.argv[1].replace('.json', '').split('/')[-1] + '.campaign.log'), 'a')
    for j in jobs:
        t = time.time()
        log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} start {json.dumps(j)}\n'); log.flush()
        r = langevin.run(j['case'], j['interp'], j['dt'], j['nwalk'], seed=j.get('seed', 1),
                         max_hours=j.get('max_hours', 1e9), expected=j.get('expected'), quiet=True,
                         lanes=j.get('lanes'), tag=j.get('tag', ''), target=j.get('target'))
        log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} end {json.dumps(r)} ({time.time() - t:.0f} s)\n'); log.flush()
