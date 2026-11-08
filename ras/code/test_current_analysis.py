import unittest
import numpy as np
from scipy.linalg import solve
from current_analysis import current,generator,test
from analyse_directed import regression,support

class RegressionTests(unittest.TestCase):
    def test_known_nonreversible_and_reversible(self):test()
    def test_directed_orientation_and_support(self):regression()
    def test_stationary_independent_dense_solve(self):
        rng=np.random.default_rng(7)
        K=generator(rng.uniform(.1,2,(9,9)))
        A=np.arange(9)==0;B=np.arange(9)==8
        J,pi,q,qm=current(K,A,B)
        m=K.T.copy();m[-1]=1;rhs=np.zeros(9);rhs[-1]=1
        self.assertTrue(np.allclose(pi,solve(m,rhs),atol=1e-12))
        self.assertAlmostEqual(J.sum(),1)
        self.assertTrue(np.allclose([q[0],q[8],qm[0],qm[8]],[0,1,1,0]))
    def test_current_time_unit_invariance(self):
        K=generator(np.array([[0.,3.,1.],[1.,0.,2.],[3.,1.,0.]]))
        A=np.array([True,False,False]);B=A[::-1]
        base=current(K,A,B)
        for factor in (1e-15,1e15):
            for x,y in zip(base,current(K*factor,A,B)):
                self.assertTrue(np.allclose(x,y,atol=1e-12))
    def test_overlap_total_variation_identity(self):
        x=np.array([.1,.2,.7]);y=np.array([.4,.4,.2])
        self.assertAlmostEqual(np.minimum(x,y).sum(),1-abs(x-y).sum()/2)
    def test_one_way_counts_not_repaired(self):
        C=np.array([[5.,2.,0.],[0.,4.,1.],[0.,0.,5.]])
        before=C.copy();r=support(C,np.array([1,0,0],bool),np.array([0,0,1],bool))
        self.assertTrue(np.array_equal(before,C))
        self.assertEqual(r['closed_components_containing_A_and_B'],0)

if __name__=='__main__':unittest.main(verbosity=2)
