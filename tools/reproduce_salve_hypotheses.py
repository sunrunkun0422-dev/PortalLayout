"""Generate SALVe hypotheses for one explicit building using predicted W/D/O."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--salve-root', type=Path, default=Path(__file__).resolve().parents[2] / 'salve')
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--predictions-root', type=Path, required=True)
    p.add_argument('--building-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not (len(args.building_id) == 4 and args.building_id.isdigit()):
        p.error('building-id must contain four digits')
    root = args.salve_root.resolve()
    raw = args.raw_root.resolve()
    predictions = args.predictions_root.resolve()
    annotation = raw / args.building_id / 'zind_data.json'
    if not annotation.is_file():
        p.error(f'Missing annotation: {annotation}')
    if not list((predictions / 'horizon_net' / args.building_id).glob('*.json')):
        p.error('Missing predicted MHNet W/D/O; GT substitution is not permitted')
    if not (predictions / 'vanishing_angle' / f'{args.building_id}.json').is_file():
        p.error('Missing vanishing-angle JSON; run the official split script first')
    if args.output.exists():
        p.error('Use a new output directory to avoid mixing different runs')
    sys.path.insert(0, str(root))
    from scripts.export_alignment_hypotheses import export_single_building_wdo_alignment_hypotheses
    export_single_building_wdo_alignment_hypotheses(
        hypotheses_save_root=str(args.output.resolve()), building_id=args.building_id,
        json_annot_fpath=str(annotation), raw_dataset_dir=str(raw),
        use_inferred_wdos_layout=True, mhnet_predictions_data_root=str(predictions))
    files = list(args.output.rglob('*.json'))
    if not files:
        raise RuntimeError('No hypotheses were generated')
    print(json.dumps({'building_id': args.building_id, 'json_count': len(files),
                      'end_to_end_reproduced': False}))


if __name__ == '__main__':
    main()
