"""Analytic camera-geometry checks independent of learned weights."""
import numpy as np
from tools.salve_dav2_depth import bases, sphere, radial_from_faces, render_face


def test_radial_distance_recovers_constant_radius_sphere():
    size = 256
    y, x = np.meshgrid((np.arange(size)+.5)*2/size-1,(np.arange(size)+.5)*2/size-1,indexing='ij')
    face_z = (3.0 / np.sqrt(1+x*x+y*y)).astype(np.float32)
    radial = radial_from_faces([face_z]*6,128,256)
    np.testing.assert_allclose(radial,3.0,atol=.012)


def test_face_projection_preserves_direction_and_erp_seam():
    rays = sphere(256,512)
    rgb = ((rays+1)*100).astype(np.float32)
    for f,r,d in bases():
        face=render_face(rgb,(f,r,d),65)
        # At a pole the nearest ERP row is pi/(2H) away from the pole.
        np.testing.assert_allclose(face[32,32],(f+1)*100,atol=.7)
        np.testing.assert_allclose(np.cross(r,d),f,atol=1e-8)
