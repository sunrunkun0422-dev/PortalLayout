from unittest.mock import patch

import pytest

from tools import run_floor_layout


class CacheReached(Exception):
    pass


@pytest.mark.parametrize('command,extra', [
    ('prepare', []),
    ('train', ['--stage', 'pairs', '--mode', 'smoke']),
])
def test_dataset_option_reaches_cache_opening(tmp_path, command, extra):
    args = [command, '--dataset', str(tmp_path/'dataset'), '--prediction-cache',
            str(tmp_path/'cache'), '--output', str(tmp_path/'out'), '--device', 'cpu'] + extra
    with patch('portallayout.utils.frozen_opening_cache.FrozenOpeningCache', side_effect=CacheReached), \
         patch('portallayout.datasets.floor_auxiliary_supervision.house_protocol', return_value={}):
        with pytest.raises(CacheReached):
            run_floor_layout.main(args)


def test_evaluate_accepts_dataset_argument_before_checking_predictions(tmp_path, capsys):
    args = ['evaluate', '--dataset', str(tmp_path/'dataset'), '--predictions',
            str(tmp_path/'missing'), '--output', str(tmp_path/'out'), '--device', 'cpu']
    with pytest.raises(SystemExit):
        run_floor_layout.main(args)
    error = capsys.readouterr().err
    assert 'no floor predictions found' in error
    assert '--portallayout.datasets' not in error
