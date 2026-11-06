import inspect
import unittest
import numpy as np
from tools.validate_portal_geometry_hypothesis import normalized_target,solve,edits,decompose,order_gt,angle_depth_oracle,metrics
from portallayout.geometry.candidate_local_portal_geometry import panorama_ray,bi_to_zind

class GeometryAttributionTests(unittest.TestCase):
    def test_target_coordinate_change_with_different_heights(self):
        theta=.6;s=1.3
        T=np.array([[s*np.cos(theta),-s*np.sin(theta),2],[s*np.sin(theta),s*np.cos(theta),-1],[0,0,1.]])
        p=np.array([.5,2.,1.]);ha=1.2;hb=1.8
        actual=normalized_target(T,ha,hb)@np.array([p[0]/hb,p[1]/hb,1.])
        expected=T@p;expected[:2]/=ha
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    def test_component_replacement_preserves_other_components(self):
        e=np.array([[1.,1.],[3.,1.]]);g=np.array([[4.,2.],[4.,6.]])
        out=edits(e,g);c,t,w=decompose(e);gc,gt,gw=decompose(g)
        dc,dt,dw=decompose(out['O_direction']);np.testing.assert_allclose(dc,c);self.assertAlmostEqual(dw,w);self.assertAlmostEqual(abs(dt@gt),1)
        mc,mt,mw=decompose(out['O_midpoint']);np.testing.assert_allclose(mc,gc);np.testing.assert_allclose(mt,t);self.assertAlmostEqual(mw,w)
        wc,wt,ww=decompose(out['O_width']);np.testing.assert_allclose(wc,c);np.testing.assert_allclose(wt,t);self.assertAlmostEqual(ww,gw)
        for old,new in zip(e,out['O_radial']):np.testing.assert_allclose(old/np.linalg.norm(old),new/np.linalg.norm(new))
    def test_GT_angle_depth_interpolation_seam(self):
        depth=np.full(256,2.5);gt=np.stack([bi_to_zind(np.array([panorama_ray(i)]))[0]*5 for i in [255,0]])
        e=angle_depth_oracle(depth,gt);np.testing.assert_allclose(np.linalg.norm(e,axis=1),2.5)
        np.testing.assert_allclose(e/2.5,gt/5)
        np.testing.assert_allclose(order_gt(gt[::-1],(255,0)),gt)
    def test_solver_signature_has_no_pose_truth(self):
        self.assertEqual(list(inspect.signature(solve).parameters),['ea','eb','ba','bb'])
    def test_perfect_geometry_with_opposite_room_sides(self):
        a=np.array([[1.,-1.],[1.,1.]])
        T=np.array([[1.,0.,2.],[0.,1.,0.],[0.,0.,1.]])
        b=a-np.array([2.,0.])
        ba=np.array([[-1.,-2.],[1.,-2.],[1.,2.],[-1.,2.]])
        bb=np.array([[-1.,-2.],[1.,-2.],[1.,2.],[-1.,2.]])
        pred,_=solve(a,b,ba,bb);np.testing.assert_allclose(pred,T,atol=1e-12)
        for ea,eb in [(a[::-1],b),(a,b[::-1]),(a[::-1],b[::-1])]:
            alt,_=solve(ea,eb,ba,bb);np.testing.assert_allclose(alt,T,atol=1e-12)
        self.assertTrue(metrics(pred,T,1.)['success'])
if __name__=='__main__':unittest.main()
