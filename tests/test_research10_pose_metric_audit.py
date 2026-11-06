import numpy as np

from portallayout.geometry.floor_geometry import matrix
from tools.research10_pose_metric_audit import metric_pose_error


def test_metric_translation_uses_anchor_scale_and_keeps_rotation():
    a = matrix([5., 3., .2, np.log(2.)])
    truth = matrix([1., 2., .3, 0.])
    b = a @ truth
    pred = truth.copy(); pred[0, 2] += .2
    result = metric_pose_error(pred, a, b, 2.)
    assert abs(result['translation_m'] - .8) < 1e-10
    assert result['rotation_deg'] < 1e-10
    assert result['correct_50cm_10deg'] is False


def test_missing_metric_scale_does_not_manufacture_meters():
    assert metric_pose_error(np.eye(3), np.eye(3), np.eye(3), None) is None
