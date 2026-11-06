#!/usr/bin/env python3
"""Create an offline-only, reproducible review pack for R06.

The candidate universe uses annotation-geometry proxies, not visual labels.
Nothing produced here is input to a model or a substitute for human review.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from portallayout.datasets.zind_floor_dataset import read_json, read_jsonl, safe_path, iter_floor_pairs


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rank(seed, *values):
    return hashlib.sha256(json.dumps([seed, *values], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def select_candidates(records, seed, count):
    """House-round-robin with independent deterministic order within each house."""
    by_house = defaultdict(list)
    seen = set()
    for r in records:
        key = (r['floor_uid'], r['pair_id'])
        if key in seen:
            raise ValueError('duplicate floor/pair in sampling universe')
        seen.add(key)
        by_house[r['house_id']].append(r)
    for rows in by_house.values():
        rows.sort(key=lambda r: (rank(seed, r['floor_uid'], r['pair_id']), r['pair_id']))
    houses = sorted(by_house, key=lambda h: (rank(seed, 'house', h), h))
    selected = []
    round_index = 0
    while len(selected) < count:
        found = False
        for house in houses:
            rows = by_house[house]
            if round_index < len(rows):
                selected.append(rows[round_index])
                found = True
                if len(selected) == count:
                    break
        if not found:
            break
        round_index += 1
    return selected


def validate_cohort(cohort, floor_samples):
    roles = cohort['roles']
    expected = ('fit', 'selection', 'calibration', 'audit')
    sets = {k: set(roles[k]) for k in expected}
    if any(sets[a] & sets[b] for i, a in enumerate(expected) for b in expected[i+1:]):
        raise ValueError('cohort role house overlap')
    rows = cohort['cohort']
    seen = set()
    for row in rows:
        floor, house, role = row['floor_uid'], row['house_id'], row['role']
        if role not in sets or house not in sets[role] or floor in seen:
            raise ValueError('cohort row role, house or floor invalid')
        seen.add(floor)
        sample = floor_samples.get(floor)
        if not sample or sample['official_split'] != 'train' or sample['house_id'] != house:
            raise ValueError('cohort floor not in official train or house differs')
    if {r['house_id'] for r in rows} != set.union(*sets.values()):
        raise ValueError('cohort rows do not cover role houses')
    return rows


def projection_index(records):
    index = {}
    for r in records:
        for group in r.get('interfaces_by_layout', {}).values():
            for x in group:
                projection = x.get('projection') or {}
                uid = x.get('interface_observation_id')
                if uid and projection.get('endpoint_u') and projection.get('top_v') and projection.get('bottom_v'):
                    index[uid] = {'endpoint_u': projection['endpoint_u'], 'top_v': [projection['top_v'][0], projection['top_v'][-1]],
                                  'bottom_v': [projection['bottom_v'][0], projection['bottom_v'][-1]],
                                  'wraps_seam': bool(projection.get('wraps_seam'))}
    return index


def source_hashes(dataset, raw, cohort, selected, out):
    """Hash the complete sampling source and every presented image."""
    dataset_files = {'dataset_meta.json', 'samples/all_panoramas.jsonl'}
    for row in cohort['cohort']:
        dataset_files.add(row['input_manifest'])
        for name in ('floor_gt.json', 'observations_gt.jsonl', 'portal_links.jsonl'):
            dataset_files.add('labels/' + row['floor_uid'] + '/' + name)
    originals = sorted({r['image_' + side + '_relative'] for r in selected for side in ('a', 'b')})
    thumbs = sorted({r['image_' + side] for r in selected for side in ('a', 'b')})
    return {
        'dataset': {p: digest_file(safe_path(dataset, p)) for p in sorted(dataset_files)},
        'original_rgb': {p: digest_file(safe_path(raw, p)) for p in originals},
        'review_thumbnails': {p: digest_file(safe_path(out, p)) for p in thumbs},
    }


def enumerate_universe(dataset, cohort, seed):
    floor_samples = {r['floor_uid']: r for r in read_jsonl(dataset / 'samples/all_panoramas.jsonl')}
    rows = validate_cohort(cohort, floor_samples)
    universe, counters = [], Counter()
    for floor_row in rows:
        floor, house, role = floor_row['floor_uid'], floor_row['house_id'], floor_row['role']
        sample = floor_samples[floor]
        manifest = read_json(safe_path(dataset, sample['input_manifest']))
        obs_rgb = {o['observation_id']: o['rgb']['relative_path'] for o in manifest['observations']}
        label_dir = safe_path(dataset, 'labels/' + floor)
        obs_gt = list(read_jsonl(label_dir / 'observations_gt.jsonl'))
        projections = projection_index(obs_gt)
        for pair in iter_floor_pairs(label_dir, manifest['observation_order']):
            counters['all_unordered_pairs'] += 1
            if pair['relation_scope'] != 'different_complete':
                counters['excluded_same_complete_room'] += 1
                continue
            counters['cross_room_pairs'] += 1
            if not pair['pose_gt_available']:
                counters['excluded_no_gt_pose'] += 1
                continue
            links = [x for x in pair['portal_correspondences'] if x['type'] in ('door', 'opening')]
            if not links:
                counters['excluded_no_nonambiguous_door_or_opening_proxy'] += 1
                continue
            mapped = [x for x in links if x['source_interface_a'] in projections and x['source_interface_b'] in projections]
            if not mapped:
                counters['excluded_no_projection'] += 1
                continue
            link = min(mapped, key=lambda x: (rank(seed, pair['pair_id'], x['physical_interface_uid']), x['physical_interface_uid']))
            a, b = pair['observation_a'], pair['observation_b']
            record = {'case_id': rank(seed, floor, pair['pair_id'])[:24], 'pair_id': pair['pair_id'],
                      'house_id': house, 'floor_uid': floor, 'role': role,
                      'observation_a': a, 'observation_b': b, 'image_a_relative': obs_rgb[a], 'image_b_relative': obs_rgb[b],
                      'opening_proxy': {'physical_interface_uid': link['physical_interface_uid'], 'kind': link['type'],
                          'source_interface_a': link['source_interface_a'], 'source_interface_b': link['source_interface_b'],
                          'projection_a': projections[link['source_interface_a']], 'projection_b': projections[link['source_interface_b']],
                          'status': 'annotation_geometry_proxy_unreviewed'},
                      'source': 'offline_gt_geometry_proxy', 'human_review': None}
            universe.append(record)
            counters['eligible_pairs'] += 1
    return universe, counters


def prepare(args):
    dataset, raw, out = args.dataset.resolve(), args.raw_root.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('output already exists; use a new directory')
    cohort_path = args.cohort.resolve()
    cohort = read_json(cohort_path)
    records, counts = enumerate_universe(dataset, cohort, args.seed)
    selected = select_candidates(records, args.seed, args.initial + args.reserve)
    if len(selected) < args.initial + args.reserve:
        raise ValueError('insufficient eligible pairs for requested initial plus reserve: %d' % len(selected))
    for i, row in enumerate(selected):
        row['sample_index'] = i + 1
        row['batch'] = 'initial' if i < args.initial else 'reserve'
    # Refuse nonexistent or out-of-root images before creating the output.
    for r in selected:
        for side in ('a', 'b'):
            p = safe_path(raw, r['image_' + side + '_relative'])
            if not p.is_file():
                raise FileNotFoundError(str(p))
    out.mkdir(parents=True)
    images = out / 'images'
    images.mkdir()
    image_map = {}
    for row in selected:
        for side in ('a', 'b'):
            relative = row['image_' + side + '_relative']
            if relative not in image_map:
                name = hashlib.sha256(relative.encode()).hexdigest()[:24] + '.jpg'
                path = images / name
                with Image.open(safe_path(raw, relative)) as img:
                    img.convert('RGB').resize((1024, 512), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS).save(path, quality=88)
                image_map[relative] = 'images/' + name
            row['image_' + side] = image_map[relative]
    identity = {'schema': 'r06-review-pack.v1', 'seed': args.seed, 'initial_count': args.initial,
                'reserve_count': args.reserve, 'cohort_sha256': digest_file(cohort_path),
                'dataset_meta_sha256': digest_file(dataset / 'dataset_meta.json'),
                'code_sha256': {Path(__file__).name: digest_file(Path(__file__)),
                                'research10_r06_review.html': digest_file(ROOT / 'tools/research10_r06_review.html')},
                'python_version': sys.version.split()[0], 'pillow_version': pillow_version,
                'cohort_file': str(cohort_path), 'dataset_root': str(dataset), 'raw_root': str(raw),
                'sampling_rule': 'eligible cross-room physical door/opening geometry proxy and valid GT pose; round-robin houses; sha256 order',
                'data_use': 'offline_human_review_only', 'source': 'official_train_only',
                'candidate_counts': dict(counts), 'universe_houses': len({r['house_id'] for r in records}),
                'selected_case_counts_by_role': dict(Counter(r['role'] for r in selected)),
                'selected_house_counts_by_role': {role: len({r['house_id'] for r in selected if r['role'] == role})
                                                  for role in sorted({r['role'] for r in selected})},
                'selected_houses_by_batch': {b: len({r['house_id'] for r in selected if r['batch'] == b}) for b in ('initial','reserve')},
                'created_at': datetime.now().astimezone().isoformat(), 'human_reviews_completed': 0}
    # Pack identifier depends only on source cohort, seed, and ordered sample IDs, not creation time.
    pack_id = rank(args.seed, identity['cohort_sha256'], *(r['case_id'] for r in selected))
    identity['pack_id'] = pack_id
    payload = {'pack_id': pack_id, 'cases': selected}
    sources = source_hashes(dataset, raw, cohort, selected, out)
    (out / 'source_hashes.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    identity['source_hashes_sha256'] = digest_file(out / 'source_hashes.json')
    (out / 'sampling.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'cases.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    html = (ROOT / 'tools' / 'research10_r06_review.html').read_text(encoding='utf-8')
    html = html.replace('__PACK_PAYLOAD__', json.dumps(payload, ensure_ascii=False).replace('</', '<\\/'))
    (out / 'index.html').write_text(html, encoding='utf-8')
    (out / 'README.md').write_text('> 修改记录：' + datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z') + ' - 创建 R06 人工核验包。\n\n'
        '# R06 跨房门后结构人工核验\n\n打开 `index.html`。黄色线是标注几何投影代理，不是人工结论；先判断共享开口，再分别判断两个方向的可见性、遮挡和目标归属。'
        '若可用，在对应来源图片上依次点击标记墙地边界点；点坐标为归一化全景坐标。选择其他样本前先保存当前方向。'
        '完成后填写审核者并导出 JSON；本地浏览器存储仅为方便继续，导出文件须妥善保存。\n\n'
        '初始200对与预留200对均已冻结；预留仅在初始样本支持不足时启用并如实报告。所有标签初始待审。'
        '本包使用离线标注几何筛选和投影，不得作为模型推理输入；不等同于像素可见性标签。'
        'fit、selection、calibration、audit 必须按房屋隔离使用；校准和复核标签不能用于训练或选型。\n', encoding='utf-8')
    print(json.dumps({'status': 'prepared_pending_human_review', 'output': str(out), 'pack_id': pack_id,
                      'initial': args.initial, 'reserve': args.reserve, 'eligible': len(records),
                      'houses_initial': identity['selected_houses_by_batch']['initial']}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--initial', type=int, default=200)
    p.add_argument('--reserve', type=int, default=200)
    args = p.parse_args()
    if args.initial < 1 or args.reserve < 0:
        p.error('invalid sample counts')
    prepare(args)


if __name__ == '__main__':
    main()
