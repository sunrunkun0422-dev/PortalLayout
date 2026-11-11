"""Behavioral checks for migration compatibility and repository resources."""
import importlib
from pathlib import Path
import pickle
import subprocess
import sys
from unittest.mock import patch


def test_legacy_modules_share_identity_and_patches():
    for old, new in [
        ('models.cross_scene_contracts', 'portallayout.models.cross_scene_contracts'),
        ('utils.floor_geometry', 'portallayout.geometry.floor_geometry'),
        ('utils.floor_training', 'portallayout.engine.floor_training'),
        ('dataset.zind_floor_dataset', 'portallayout.datasets.zind_floor_dataset'),
        ('loss.grad_loss', 'portallayout.losses.grad_loss'),
    ]:
        assert importlib.import_module(old) is importlib.import_module(new)
    canonical = importlib.import_module('portallayout.geometry.floor_geometry')
    sentinel = object()
    with patch('utils.floor_geometry.polygon', sentinel):
        assert canonical.polygon is sentinel


def test_historical_pickle_global_resolves_to_canonical_class():
    # Protocol 0 GLOBAL opcode reproduces a saved class reference without
    # modifying the real class or relying on a newly serialized module name.
    from portallayout.models.cross_scene_contracts import CoordinateFrameSpec
    assert pickle.loads(b'cmodels.cross_scene_contracts\nCoordinateFrameSpec\n.') is CoordinateFrameSpec


def test_legacy_import_first_in_fresh_process():
    result = subprocess.run([sys.executable, '-c',
        'from utils import floor_geometry as from_old; '
        'import utils.floor_geometry as old; '
        'import portallayout.geometry.floor_geometry as new; '
        'assert old is new and from_old is new; '
        'import models.modules as old_models; '
        'import portallayout.models.modules as new_models; '
        'assert old_models is new_models'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_repository_paths_survive_module_move():
    from portallayout.engine.floor_observation_pipeline import REPO_ROOT
    from portallayout.engine.single_view_prediction import REPO_ROOT as SINGLE_ROOT
    root = Path(__file__).resolve().parents[1]
    assert REPO_ROOT == SINGLE_ROOT == root
    for old, new in [('src/config', 'configs/layout'), ('src/dataset', 'data'), ('src/output', 'outputs')]:
        assert (root / old).resolve() == (root / new).resolve()
