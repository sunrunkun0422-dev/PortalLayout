import pytest

from tools.research10_objectives_job import training_spec, verify_report


def test_diagnostic_plan_is_bounded_and_selects_only_on_selection_metric(tmp_path):
    for objective in ('assignment', 'positive_margin'):
        for stage, steps in (('smoke', '200'), ('diagnostic', '5000')):
            spec = training_spec(objective, stage, *[tmp_path/str(i) for i in range(5)])
            argv = spec['argv']
            assert spec['purpose'] == 'diagnostic'
            assert argv[argv.index('--pair-steps')+1] == steps
            assert argv[argv.index('--pair-objective')+1] == objective
            assert argv[argv.index('--selection-metric')+1] == 'cross_room_ap'


def test_missing_or_nonfinite_gradient_cannot_unlock_diagnostic():
    report = {'training_status': 'smoke_trained', 'steps': 200, 'parameters_changed': True,
              'skipped_batches': 2, 'gradient_norms_first_step': {'encoder': 1., 'matcher': 2.}}
    verify_report(report, 'smoke')
    report['gradient_norms_first_step']['encoder'] = float('nan')
    with pytest.raises(ValueError): verify_report(report, 'smoke')
