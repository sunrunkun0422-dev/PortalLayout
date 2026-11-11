"""Run the official pretrained verifier on the official four-image fixture.

This is a module smoke check, NOT an end-to-end reconstruction or benchmark.
Run with the SALVe environment's Python; no training or pretrained downloads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salve-root', type=Path, default=Path(__file__).resolve().parents[2] / 'salve')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.salve_root.resolve()
    sys.path.insert(0, str(root))
    import imageio.v2 as imageio
    import torch
    from salve.models.early_fusion import EarlyFusionCEResnet
    from salve.train_utils import get_val_test_transform

    torch.set_num_threads(4)
    checkpoint = root / 'checkpoints/salve/1200ffbe47d836557d88fef052952337.pth'
    digest = hashlib.md5()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != checkpoint.stem:
        raise RuntimeError('Official checkpoint MD5 mismatch')
    cfg = SimpleNamespace(modalities=['ceiling_rgb_texture', 'floor_rgb_texture'],
                          resize_h=234, resize_w=234, train_h=224, train_w=224)
    model = torch.nn.DataParallel(EarlyFusionCEResnet(152, False, 2, cfg))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
    model.eval()
    paths = sorted((root / 'tests/test_data/Renderings/gt_alignment_approx/1208').glob('*.jpg'))
    assert len(paths) == 4
    assert all('_ceiling_rgb_' in p.name for p in paths[:2])
    assert all('_floor_rgb_' in p.name for p in paths[2:])
    tensors = get_val_test_transform(cfg)(*[imageio.imread(p) for p in paths])
    with torch.inference_mode():
        logits = model(*[t.unsqueeze(0) for t in tensors], None, None)
        probabilities = logits.softmax(dim=1)[0]
    assert torch.isfinite(probabilities).all()
    result = {'scope': 'official_fixture_verifier_smoke_only', 'end_to_end_reproduced': False,
              'salve_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
              'checkpoint_md5': digest.hexdigest(), 'strict_checkpoint_load': True,
              'torch_version': torch.__version__, 'device': 'cpu',
              'inputs': [str(p) for p in paths], 'logits': logits[0].tolist(),
              'probabilities': probabilities.tolist(), 'predicted_class': int(probabilities.argmax()),
              'fixture_label': 1, 'benchmark_metrics': None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
