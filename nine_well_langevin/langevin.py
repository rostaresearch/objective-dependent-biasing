"""Overdamped Langevin first-passage times on the continuous nine-well potential, F + b.

    dx = -D grad(F + b) dt + sqrt(2 D) dW,   D = 1, F and b in kBT

Euler-Maruyama, reflecting walls at x, y in {0, 2pi} (the network generator has no-flux edges and
no periodic links). Each walker starts uniformly in the square A (the continuum limit of "uniform
over the A cells") and is stopped at its first step that lands in the square B. Each walker has
its own xoshiro256** stream seeded from (seed, walker id), so a walker's trajectory does not depend
on which thread runs it or on the number of threads.

Walkers run in lanes; Python hands a new walker id to a lane when its walker finishes, and the run
stops after N walkers have finished (no censoring). The state is checkpointed so a run can be resumed.

Usage:  python langevin.py CASE INTERP DT NWALKERS [--seed S] [--tag T] [--max-hours H]
    CASE in unbiased, MFPT_U3, spectral_U3, MFPT_U6, spectral_U6, free (free-diffusion test)
    INTERP in bilinear, bicubic, none
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numba
import numpy as np
from numba import njit, prange

from common import (CENTRE_A, CENTRE_B, D_COEF, HALF, HERE, L_BOX, bias_bicubic, bias_bilinear,
                    bicubic_coefficients, force_table, load_bias_table, potential_grad, table_grad)

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_U5 = np.uint64(5)
_U9 = np.uint64(9)


@njit(cache=True)
def _rotl(x, k):
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


@njit(cache=True)
def _splitmix(x):
    x = x + _GOLD
    z = x
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return x, z ^ (z >> np.uint64(31))


@njit(cache=True)
def _seed(s, seed, wid):
    x = np.uint64(seed) * _M2 + np.uint64(wid) * _M1 + _GOLD
    for k in range(4):
        x, z = _splitmix(x)
        s[k] = z


@njit(cache=True)
def _next(s):
    result = _rotl(s[1] * _U5, 7) * _U9
    t = s[1] << np.uint64(17)
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = _rotl(s[3], 45)
    return result


@njit(cache=True)
def _uniform(s):
    """Uniform on (0, 1]."""
    return (np.float64(_next(s) >> np.uint64(11)) + 1.) * 1.1102230246251565e-16


def _ziggurat_tables():
    """256-layer Marsaglia-Tsang ziggurat tables for 52-bit mantissas (as in numpy's legacy generator)."""
    r = 3.6541528853610088
    v = 0.00492867323399
    ki = np.zeros(256, np.uint64); wi = np.zeros(256); fi = np.zeros(256)
    m = 2.0 ** 52
    dn = tn = r
    q = v / np.exp(-.5 * dn * dn)
    ki[0] = np.uint64((dn / q) * m); ki[1] = np.uint64(0)
    wi[0] = q / m; wi[255] = dn / m
    fi[0] = 1.; fi[255] = np.exp(-.5 * dn * dn)
    for i in range(254, 0, -1):
        dn = np.sqrt(-2. * np.log(v / dn + np.exp(-.5 * dn * dn)))
        ki[i + 1] = np.uint64((dn / tn) * m)
        tn = dn
        fi[i] = np.exp(-.5 * dn * dn)
        wi[i] = dn / m
    return ki, wi, fi, r


ZIG_K, ZIG_W, ZIG_F, ZIG_R = _ziggurat_tables()
_MASK52 = np.uint64(0x000fffffffffffff)
_U255 = np.uint64(255)
_U1 = np.uint64(1)


@njit(cache=True)
def _zig_normal(s):
    while True:
        r = _next(s)
        idx = np.int64(r & _U255)
        r = r >> np.uint64(8)
        sign = r & _U1
        rabs = (r >> _U1) & _MASK52
        x = np.float64(rabs) * ZIG_W[idx]
        if sign == _U1:
            x = -x
        if rabs < ZIG_K[idx]:
            return x
        if idx == 0:
            while True:
                xx = -np.log(_uniform(s)) / ZIG_R
                yy = -np.log(_uniform(s))
                if yy + yy > xx * xx:
                    if ((rabs >> np.uint64(8)) & _U1) == _U1:
                        return -(ZIG_R + xx)
                    return ZIG_R + xx
        else:
            if (ZIG_F[idx - 1] - ZIG_F[idx]) * _uniform(s) + ZIG_F[idx] < np.exp(-.5 * x * x):
                return x


@njit(cache=True)
def _normal_pair(s, noise):
    """Two independent standard normals: noise 0 Box-Muller, 1 Marsaglia polar, 2 ziggurat."""
    if noise == 2:
        return _zig_normal(s), _zig_normal(s)
    if noise == 0:
        u1 = _uniform(s)
        u2 = _uniform(s)
        rad = np.sqrt(-2. * np.log(u1))
        return rad * np.cos(2. * np.pi * u2), rad * np.sin(2. * np.pi * u2)
    while True:
        u = 2. * _uniform(s) - 1.
        v = 2. * _uniform(s) - 1.
        q = u * u + v * v
        if 0. < q < 1.:
            f = np.sqrt(-2. * np.log(q) / q)
            return u * f, v * f


@njit(cache=True)
def _force(x, y, pot_on, bias_kind, tab, coef, tg):
    """pot_on: 0 no potential, 1 analytic gradient of F, 2 tabulated gradient of F."""
    gx = 0.
    gy = 0.
    if pot_on == 1:
        _, gx, gy = potential_grad(x, y)
    elif pot_on == 2:
        gx, gy = table_grad(tg, x, y)
    if bias_kind == 1:
        _, bx, by = bias_bilinear(tab, x, y)
        gx += bx
        gy += by
    elif bias_kind == 2:
        _, bx, by = bias_bicubic(coef, x, y)
        gx += bx
        gy += by
    return -gx, -gy


@njit(parallel=True, cache=True)
def advance(xy, nst, rng, state, wid, seed, src, tgt, pot_on, bias_kind, tab, coef, tg, noise, dt, nsteps, done):
    """Advance every lane by up to nsteps steps.

    state: 0 idle, 1 running, 2 start a new walker (id in wid) at the beginning of this call.
    On first passage into tgt the lane records done = 1, keeps the step count in nst, and goes idle.
    """
    nl = xy.shape[0]
    sq = np.sqrt(2. * D_COEF * dt)
    mu = D_COEF * dt
    for l in prange(nl):
        done[l] = 0
        if state[l] == 0:
            continue
        s = rng[l]
        if state[l] == 2:
            _seed(s, seed, wid[l])
            xy[l, 0] = src[0] + src[2] * (2. * _uniform(s) - 1.)
            xy[l, 1] = src[1] + src[3] * (2. * _uniform(s) - 1.)
            nst[l] = 0
            state[l] = 1
        x = xy[l, 0]
        y = xy[l, 1]
        n = nst[l]
        for k in range(nsteps):
            fx, fy = _force(x, y, pot_on, bias_kind, tab, coef, tg)
            n1, n2 = _normal_pair(s, noise)
            x += mu * fx + sq * n1
            y += mu * fy + sq * n2
            if x < 0.:
                x = -x
            elif x > L_BOX:
                x = 2. * L_BOX - x
            if y < 0.:
                y = -y
            elif y > L_BOX:
                y = 2. * L_BOX - y
            n += 1
            if abs(x - tgt[0]) <= tgt[2] and abs(y - tgt[1]) <= tgt[3]:
                done[l] = 1
                state[l] = 0
                break
        xy[l, 0] = x
        xy[l, 1] = y
        nst[l] = n


_TABLE = None


def setup(case, interp, potential='table'):
    global _TABLE
    tab = np.zeros((20, 20))
    coef = np.zeros((19, 19, 4, 4))
    src = np.array([CENTRE_A[0], CENTRE_A[1], HALF, HALF])
    tgt = np.array([CENTRE_B[0], CENTRE_B[1], HALF, HALF])
    pot_on, bias_kind = {'analytic': 1, 'table': 2}[potential], 0
    tg = np.zeros((2, 2, 2))
    if pot_on == 2:
        if _TABLE is None:
            _TABLE = force_table()
        tg = _TABLE
    if case == 'free':
        # free diffusion, start uniform on the line x = 4 (y uniform), absorb in the strip x <= 1:
        # T = ((L - 1)^2 - (L - 4)^2) / (2 D)
        pot_on = 0
        src = np.array([4., np.pi, 0., np.pi])
        tgt = np.array([.5, np.pi, .5, np.pi + 1.])
    elif case != 'unbiased':
        tab = load_bias_table(case)
        if interp == 'bilinear':
            bias_kind = 1
        elif interp == 'bicubic':
            bias_kind, coef = 2, bicubic_coefficients(tab)
        else:
            raise ValueError(interp)
    return pot_on, bias_kind, tab, coef, tg, src, tgt


def run(case, interp, dt, nwalk, seed=1, tag='', max_hours=1e9, lanes=None, expected=None, quiet=False,
        out_dir=None, potential='table', noise='polar', target=None):
    out_dir = out_dir or HERE / 'runs'
    os.makedirs(out_dir, exist_ok=True)
    name = f'{case}_{interp}_dt{dt:g}_s{seed}{tag}'
    ck = out_dir / (name + '.npz')
    logf = open(out_dir / (name + '.log'), 'a')
    pot_on, bias_kind, tab, coef, tg, src, tgt = setup(case, interp, potential)
    if target is not None:          # alternative target square (cx, cy, half-width x, half-width y)
        tgt = np.array(target, float)
    if case == 'free':
        tg = np.zeros((2, 2, 2))
    inoise = {'boxmuller': 0, 'polar': 1, 'ziggurat': 2}[noise]
    nthreads = numba.get_num_threads()
    lanes = lanes or 4 * nthreads
    # steps per call: aim for a small idle fraction (a finished lane idles for the rest of the call)
    exp_steps = (expected or 1e3) / dt
    nsteps = int(np.clip(exp_steps / 100, 1e4, 2e6))
    if ck.exists():
        z = np.load(ck)
        xy, nst, rng, state, wid = z['xy'], z['nst'], z['rng'], z['state'], z['wid']
        ids, steps = list(z['ids']), list(z['steps'])
        next_id, wall_prev = int(z['next_id']), float(z['wall'])
        lanes = len(state)
        logf.write(f'resumed: {len(ids)} done, next_id {next_id}\n')
    else:
        xy = np.zeros((lanes, 2)); nst = np.zeros(lanes, np.int64)
        rng = np.zeros((lanes, 4), np.uint64); state = np.zeros(lanes, np.int64); wid = np.zeros(lanes, np.int64)
        ids, steps, next_id, wall_prev = [], [], 0, 0.
    done = np.zeros(lanes, np.int64)
    t_start = time.time(); t_last = t_start; total_steps = 0
    meta = dict(case=case, interp=interp, dt=dt, seed=seed, nwalk=nwalk, lanes=lanes, nthreads=nthreads,
                steps_per_call=nsteps, D=D_COEF, src=src.tolist(), tgt=tgt.tolist(), potential=potential,
                noise=noise, table_n=int(tg.shape[0]) if pot_on == 2 else None)
    logf.write(json.dumps(meta) + '\n'); logf.flush()

    def save(final=False):
        np.savez(ck, xy=xy, nst=nst, rng=rng, state=state, wid=wid, ids=np.array(ids, np.int64),
                 steps=np.array(steps, np.int64), next_id=next_id, wall=wall_prev + time.time() - t_start,
                 dt=dt, meta=json.dumps(meta), final=final)

    while len(ids) < nwalk:
        for l in range(lanes):
            if state[l] == 0 and next_id < nwalk:
                wid[l] = next_id; state[l] = 2; next_id += 1
        running = int(np.sum(state > 0))
        advance(xy, nst, rng, state, wid, seed, src, tgt, pot_on, bias_kind, tab, coef, tg, inoise, dt, nsteps, done)
        total_steps += running * nsteps
        for l in np.flatnonzero(done):
            ids.append(int(wid[l])); steps.append(int(nst[l]))
        now = time.time()
        if now - t_last > 60 or len(ids) == nwalk:
            t_last = now
            fpt = np.array(steps) * dt
            msg = (f'{time.strftime("%H:%M:%S")} done {len(ids)}/{nwalk} mean {fpt.mean() if len(fpt) else np.nan:.5g} '
                   f'running {running} lane-steps/s {total_steps / (now - t_start):.3g} '
                   f'in-progress mean age {np.mean(nst[state > 0]) * dt if running else 0:.4g}')
            logf.write(msg + '\n'); logf.flush()
            if not quiet:
                print(msg, flush=True)
            save()
        if (now - t_start) / 3600 > max_hours:
            logf.write('stopped at max_hours\n'); break
    save(final=len(ids) >= nwalk)
    fpt = np.array(steps) * dt
    return dict(name=name, n=len(fpt), mean=float(fpt.mean()) if len(fpt) else None,
                wall=wall_prev + time.time() - t_start)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('case'); ap.add_argument('interp'); ap.add_argument('dt', type=float); ap.add_argument('nwalk', type=int)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--tag', default='')
    ap.add_argument('--max-hours', type=float, default=1e9); ap.add_argument('--expected', type=float, default=None)
    ap.add_argument('--threads', type=int, default=None); ap.add_argument('--lanes', type=int, default=None)
    ap.add_argument('--potential', default='table'); ap.add_argument('--noise', default='polar')
    a = ap.parse_args()
    if a.threads:
        numba.set_num_threads(a.threads)
    print(run(a.case, a.interp, a.dt, a.nwalk, a.seed, a.tag, a.max_hours, a.lanes, a.expected,
              potential=a.potential, noise=a.noise))
