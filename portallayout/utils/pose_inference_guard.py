"""Fail closed if a prediction-only stage opens annotation or supervision files."""
import builtins
import io
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def inference_guard():
    def guarded(original):
        def call(path,*args,**kwargs):
            if isinstance(path,(str,Path)):
                p=Path(path)
                if 'labels' in p.parts or p.name in ('opening_weak.json','supervision.json','observations_gt.jsonl','floor_gt.json','portal_links.jsonl'):
                    raise AssertionError('prediction attempted supervision access: '+str(p))
            return original(path,*args,**kwargs)
        return call
    with patch('builtins.open',side_effect=guarded(builtins.open)),patch('io.open',side_effect=guarded(io.open)):
        yield
