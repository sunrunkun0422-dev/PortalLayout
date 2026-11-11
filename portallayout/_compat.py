"""Resolve historical import paths to the same canonical module objects.

Legacy pickle class paths and external scripts may still import `models` or
`utils`. No implementation is copied or imported twice under different names.
"""
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

MODULES = json.loads(Path(__file__).with_name('_legacy_modules.json').read_text())


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target
        self.spec = None

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self.spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        canonical = MODULES.get(fullname)
        if canonical is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(canonical))


_FINDER = _AliasFinder()


def alias_package(name):
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    module = importlib.import_module(MODULES[name])
    # `from utils import floor_geometry` needs the moved child even when no
    # earlier `import utils.floor_geometry` has populated the package attribute.
    if name == 'utils' and not hasattr(module, '__getattr__'):
        def resolve_moved_child(child):
            target = MODULES.get(name + '.' + child)
            if target is None:
                raise AttributeError(child)
            return importlib.import_module(target)
        module.__getattr__ = resolve_moved_child
    sys.modules[name] = module
    return module
