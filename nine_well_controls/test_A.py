"""Unit tests for the reference-value helpers. Run: python -m unittest test_A.py"""
import unittest

import numpy as np

from common import System, systems, metrics, ess_stats, overlap_signed, tilt, boltzmann
from pathstats import (tables, sample_sets, seeds_for, path_moments, norm_signed, omega_rows,
                       cts_rows, clt_two_sample_cts, clt_two_sample_omega)


def small_system():
    F = np.array([[0., 2., 3., 2.5], [1.5, 4., 1., 2.], [2., 1., 3., 0.5]])
    r, c = np.meshgrid(np.arange(3), np.arange(4), indexing='ij')
    coords = np.c_[c.ravel(), r.ravel()].astype(float)
    return System('small', F, coords, np.array([0]), np.array([11]), {}, 'test')


class TestA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = systems()
        cls.small = small_system()

    def test_zero_bias_limits(self):
        for s in list(self.S.values()) + [self.small]:
            m = metrics(s, np.zeros(s.n))
            for k in ['mfpt_speedup', 'gap_speedup', 'C_TS', 'Omega_J', 'ess_frac']:
                self.assertAlmostEqual(m[k], 1.0, places=9, msg=(s.name, k))

    def test_gauge_invariance(self):
        s = self.S['nine_well']
        b = s.biases['spectral_U6']
        m1, m2 = metrics(s, b), metrics(s, b + 4.2)
        for k in ['mfpt_speedup', 'gap_speedup', 'C_TS', 'Omega_J', 'ess_frac', 'max_w', 'var_logw']:
            self.assertAlmostEqual(m1[k], m2[k], delta=1e-9*max(1, abs(m1[k])), msg=k)

    def test_omega_signed_equals_directed(self):
        s = self.S['nine_well']
        rng = np.random.default_rng(1)
        for _ in range(5):
            m = metrics(s, rng.uniform(-6, 6, s.n))
            self.assertAlmostEqual(m['Omega_J'], m['Omega_J_sparse'], places=12)

    def test_overlap_identity(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=50); x /= np.abs(x).sum()
        y = rng.normal(size=50); y /= np.abs(y).sum()
        direct = np.minimum(np.maximum(x, 0), np.maximum(y, 0)).sum() + np.minimum(np.maximum(-x, 0), np.maximum(-y, 0)).sum()
        self.assertAlmostEqual(overlap_signed(x, y), direct, places=14)

    def test_ess_closed_form(self):
        rng = np.random.default_rng(3)
        pi0 = rng.dirichlet(np.ones(30)); b = rng.uniform(-3, 3, 30)
        pib = pi0*np.exp(-b); pib /= pib.sum()
        w = pi0/pib
        e = ess_stats(pi0, pib)
        self.assertAlmostEqual(e['ess_frac'], (pib@w)**2/(pib@(w*w)), places=14)
        self.assertAlmostEqual(e['ess_frac'], 1/(1+np.sum((pi0-pib)**2/pib)), places=12)
        self.assertLessEqual(e['ess_frac'], 1.0)

    def test_sampler_reproducible_and_unbiased(self):
        s = self.small
        tab = tables(s, np.zeros(s.n))
        a = sample_sets(tab, seeds_for(1, 2, 3, n=8), 5000)
        b = sample_sets(tab, seeds_for(1, 2, 3, n=8), 5000)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
        # pooled 40000 paths: last-crossing law and net current agree with exact law
        pooled_last = a[1].sum(0); n = pooled_last.sum()
        se = np.sqrt(s.p0*(1-s.p0)/n)
        self.assertTrue(np.all(np.abs(pooled_last/n - s.p0) < 5*se + 1/n))
        x = norm_signed(a[0].sum(0))
        self.assertLess(np.abs(x - s.J0).sum()/2, 0.02)

    def test_path_moments_against_sampling(self):
        s = self.small
        tab = tables(s, np.zeros(s.n))
        mom = path_moments(s, tab, np.sign(s.J0))
        x = mom['mean']/np.abs(mom['mean']).sum()
        np.testing.assert_allclose(x, s.J0, atol=1e-12)
        net, last, steps = sample_sets(tab, seeds_for(5, n=200000), 1)
        np.testing.assert_allclose(net.mean(0), mom['mean'], atol=5*np.sqrt(mom['var'].max()/200000)+1e-3)
        big = mom['var'] > .05
        np.testing.assert_allclose(net.var(0)[big], mom['var'][big], rtol=.05)
        self.assertAlmostEqual(steps.mean()+1, mom['mean_jumps'], delta=5*np.sqrt(mom['mean_jumps']**2/200000)+.05)

    def test_clt_cts_against_multinomial(self):
        rng = np.random.default_rng(6)
        p = rng.dirichlet(np.ones(12))
        N = 20000
        c = rng.multinomial(N, p, size=4000)
        emp = 1 - cts_rows(c[0::2], c[1::2]).mean()
        self.assertAlmostEqual(emp, clt_two_sample_cts(p, N), delta=.05*clt_two_sample_cts(p, N))

    def test_clt_omega_against_sampling(self):
        s = self.small
        tab = tables(s, np.zeros(s.n))
        mom = path_moments(s, tab, np.sign(s.J0))
        N = 4000
        net, _, _ = sample_sets(tab, seeds_for(7, n=400), N)
        xs = norm_signed(net)
        emp = 1 - omega_rows(xs[0::2], xs[1::2]).mean()
        pred = clt_two_sample_omega(mom, N)
        self.assertAlmostEqual(emp, pred, delta=.1*pred)


if __name__ == '__main__':
    unittest.main()
