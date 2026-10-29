import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from repro.camera_preprocess import (
    build_camera, camera_frame, raw_to_virtual_affine,
    render_estimated, validate_camera, virtual_points,
)


class EstimatedCameraTests(unittest.TestCase):
    def fixture(self, direction=(.2, -.1, -1.)):
        rng=np.random.default_rng(83)
        raw=rng.normal(size=(1000,3))*np.array([5.,3.,1.])+[100.,-12.,230.]
        meta=dict(basis=Rotation.from_euler('xyz',[21.,-33.,14.],degrees=True).as_matrix(),
                  oriented_center=[70.,-30.,180.],normalization_radius=8.)
        fit=dict(rotation_row=Rotation.from_euler('xyz',[-17.,8.,41.],degrees=True).as_matrix(),
                 translation_row=[.1,-.04,.02],direction=direction,
                 score=.01,normal_strength=.8,alternative_angle_deg=2.,ambiguous=False)
        return raw,meta,fit

    def test_raw_and_virtual_projection_agree_with_nontrivial_transforms(self):
        raw,meta,fit=self.fixture()
        camera=build_camera(raw,meta,fit,.02)
        validation=validate_camera(raw,camera)
        self.assertLess(validation['projection_error_pixels'],1e-6)
        a,b=raw_to_virtual_affine(camera)
        np.testing.assert_allclose(raw@a+b,virtual_points(raw,camera),atol=1e-12)
        self.assertEqual(camera['status'],'ready')
        E=np.array(camera['world_to_camera_raw'])
        np.testing.assert_allclose(E@np.array(camera['camera_to_world_raw']),np.eye(4),atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(E[:3,:3]),1.,places=10)

    def test_up_axis_poles_keep_valid_frame_and_requested_side(self):
        for d in ([0,1,0],[0,-1,0],[0,0,1],[0,0,-1]):
            frame=camera_frame(np.array(d,float))
            np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-12)
            np.testing.assert_allclose(frame[:,2],d,atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(frame),1.)

    def test_uncertain_pose_is_blocked_before_reading_point_labels(self):
        raw,meta,fit=self.fixture()
        fit['alternative_angle_deg']=100.
        camera=build_camera(raw,meta,fit,.02)
        self.assertEqual(camera['status'],'needs_review')
        temp_parent=Path(__file__).resolve().parents[1]/'outputs/test_camera_tmp'
        temp_parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
            root=Path(directory)
            p=root/'fish/test/example/camera.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(camera))
            with self.assertRaisesRegex(ValueError,'requires review'):
                render_estimated('fish','example',None,None,root,root/'renders')

    def test_poor_fit_and_weak_direction_are_not_silently_accepted(self):
        raw,meta,fit=self.fixture();fit['score']=.05;fit['normal_strength']=.1
        camera=build_camera(raw,meta,fit,.02)
        self.assertEqual(camera['status'],'needs_review')
        self.assertIn('weak_surface_direction',camera['reasons'])
        self.assertIn('partial_template_fit_too_large',camera['reasons'])


if __name__=='__main__':unittest.main()
