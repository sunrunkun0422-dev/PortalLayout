"""Statistical regressions for the artifact-only evidence dossier."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('evidence', Path(__file__).resolve().parents[1]/'tools/build_method_advantage_evidence.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def floor(house, errors, iou=None):
    return dict(house_id=house, metrics={'overall_iou': iou},
                pose_errors=[dict(translation_cm=v) for v in errors])


def test_pooled_median_is_not_mean_of_floor_medians():
    rows = [floor('a',[0,1,2]),floor('b',[100])]
    assert evidence.aggregate(rows,'translation_error_cm') == 1.5


def test_missing_metrics_are_not_zero():
    assert evidence.aggregate([floor('a',[],None)],'translation_error_cm') is None
    assert evidence.aggregate([floor('a',[],None),floor('b',[],.8)],'overall_iou') == .8


def test_house_pairing_and_missing_report():
    result=evidence.bootstrap_delta({'a':.2,'b':None,'c':.8},{'a':.3,'b':.5})
    assert result['houses']==['a']
    assert result['excluded_houses']==['b','c']
    assert result['delta']==pytest.approx(.1)
    assert result['ci95']==pytest.approx([.1,.1])


def test_cluster_bootstrap_preserves_constant_effect():
    a=[floor('a',[1,2],.2),floor('a',[3],.4),floor('b',[4],.5)]
    b=[floor('a',[11,12],.3),floor('a',[13],.5),floor('b',[14],.6)]
    for metric,delta in [('translation_error_cm',10),('overall_iou',.1)]:
        result=evidence.floor_bootstrap(a,b,metric)
        assert result['delta']==pytest.approx(delta)
        assert result['ci95']==pytest.approx([delta,delta])
        assert result['valid_replicates']==2000


def test_cluster_bootstrap_rejects_different_houses():
    with pytest.raises(ValueError,match='house sets differ'):
        evidence.floor_bootstrap([floor('a',[1])],[floor('b',[1])],'translation_error_cm')


def test_duplicate_samples_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        evidence.unique(['one','one'],'pair')


def test_no_valid_replicates_stays_na():
    result=evidence.floor_bootstrap([floor('a',[])],[floor('a',[])],'translation_error_cm')
    assert result['delta'] is None and result['ci95'] is None
    assert result['valid_replicates']==0
