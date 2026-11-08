import unittest
import numpy as np
from scipy.sparse import csr_matrix
from exact_last_crossing import harmonic, law, tilt, boltzmann, generator_grid


class LastCrossingTests(unittest.TestCase):
    def test_triangle_closed_form(self):
        a,c,d=.05,1.,.02
        K=csr_matrix([[-a-c,a,c],[a,-a-d,d],[c,d,-c-d]])
        q0=harmonic(K,[0],[1])
        for b in [np.zeros(3),np.array([0.,-3.,2.]),np.array([0.,3.,-3.])]:
            kb=tilt(K,b); result=law(kb,boltzmann(b),[0],[1],q0)
            # Two competing successful routes: direct A->B and A->D->B.
            direct=float(kb[0,1]); via=float(kb[0,2]*kb[2,1]/(-kb[2,2]))
            np.testing.assert_allclose(result['probability'],np.array([direct,via])/(direct+via),rtol=1e-12)

    def test_time_rescaling_and_gauge_invariance(self):
        K=generator_grid(np.array([[0.,2.,1.],[1.,3.,0.]]))
        pi=boltzmann(np.array([0.,2.,1.,1.,3.,0.]))
        q=harmonic(K,[0],[5]); ref=law(K,pi,[0],[5],q)
        scaled=law(K*7,pi,[0],[5],q)
        np.testing.assert_allclose(scaled['probability'],ref['probability'],atol=1e-12)
        self.assertAlmostEqual(scaled['capacity']/ref['capacity'],7.)
        b=np.arange(6)*.1
        np.testing.assert_allclose(tilt(K,b).toarray(),tilt(K,b+12).toarray(),atol=1e-12)

    def test_one_dimensional_chain_is_trivial(self):
        K=generator_grid(np.array([0.,1.,3.,1.,0.]))
        q=harmonic(K,[0],[4])
        ref=law(K,boltzmann(np.array([0.,1.,3.,1.,0.])),[0],[4],q)
        self.assertEqual(len(ref['probability']),1)
        self.assertAlmostEqual(ref['probability'][0],1.)

    def test_nonreversible_no_symmetrization(self):
        # Clockwise and counterclockwise cycle rates differ, but pi is uniform.
        K=csr_matrix([[-4,3,0,1],[1,-4,3,0],[0,1,-4,3],[3,0,1,-4]])
        pi=np.ones(4)/4
        q=harmonic(K,[0],[2]); result=law(K,pi,[0],[2],q)
        self.assertGreater(np.max(abs(result['qminus']-(1-q))),.1)
        self.assertLess(result['capacity_relative_residual'],1e-12)


if __name__=='__main__':
    unittest.main()
